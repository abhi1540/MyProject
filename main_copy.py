from langchain.document_loaders.base import BaseLoader
from abc import ABC
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union,Iterable
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
import numpy as np
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.memory import ConversationBufferMemory
from langchain.llms import LlamaCpp
from langchain.chains import ConversationChain, LLMChain, ConversationalRetrievalChain, RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import MongoDBChatMessageHistory
from langchain.vectorstores.pgvector import PGVector
import queue
from langchain.callbacks.manager import AsyncCallbackManager
import threading
from langchain.callbacks import AsyncIteratorCallbackHandler
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.callbacks.base import BaseCallbackHandler
import requests
import uvicorn
from langchain.schema import HumanMessage
import asyncio
from queue import SimpleQueue
from langchain.schema import LLMResult
from threading import Thread
import gradio as gr

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# callback = AsyncIteratorCallbackHandler()

job_done = object()  # signals the processing is done

class StreamingGradioCallbackHandler(BaseCallbackHandler):
    """Callback handler - works with LLMs that support streaming."""

    def __init__(self, q: SimpleQueue):
        self.q = q

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        while not self.q.empty():
            try:
                self.q.get(block=False)
            except SimpleQueue.empty:
                continue

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.q.put(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.q.put(job_done)

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        self.q.put(job_done)
        
# Initializes the LLM
q = SimpleQueue()


llm = LlamaCpp(model_path='D:/Workspace/PrivateGPTLangchain/models/llm/mistral-7b-v0.1.Q4_0.gguf',
        n_ctx=10000,
        max_tokens=250,
        n_batch=64,
        n_threads=max(os.cpu_count() // 2, 1),
        n_threads_batch=max(os.cpu_count() // 2, 1),
        callbacks=[StreamingGradioCallbackHandler(q)],
        n_gpu_layers=10)

prompt = "Act like a knowledgeable professional, only answer once, and always limit your answers to the document content only. Never make up answers. If you do not have the answer, state that the data is not contained in your knowledge base and stop your response."

embeddings = HuggingFaceEmbeddings(model_name=r"D:\Workspace\DocChat\models\embedding\all-mpnet-base-v2", model_kwargs={"device": 'cuda'})
CONNECTION_STRING = "postgresql+psycopg2://postgres:admin@localhost:5432/vector_db"
COLLECTION_NAME = 'document_vector'
db = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)
    
chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',retriever=db.as_retriever())

    
# Set up chat history and streaming for Gradio Display
def process_question(question):
    chat_history = []
    full_query = f"{prompt} {question}"
    result = chain({"question": full_query, "chat_history": chat_history})
    return result["answer"]


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""


def streaming_chat(history):
    user_input = history[-1][0]
    thread = Thread(target=process_question, args=(user_input,))
    thread.start()
    history[-1][1] = ""
    while True:
        next_token = q.get(block=True)  # Blocks until an input is available
        if next_token is job_done:
            break
        history[-1][1] += next_token
        yield history
    thread.join()


# Creates A gradio Interface
with gr.Blocks() as demo:
    Langchain = gr.Chatbot(label="Langchain Response", height=500)
    Question = gr.Textbox(label="Question")
    Question.submit(add_text, [Langchain, Question], [Langchain, Question]).then(
        streaming_chat, Langchain, Langchain
    )
demo.queue(default_concurrency_limit=10)

app = FastAPI()
# CORS configuration
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Request(BaseModel):
    question: str



class StreamingResponse(BaseModel):
    results: str


@app.post("/predict", response_model=StreamingResponse)
async def predict_api(request: Request):
    results = process_question(request.question)
    return StreamingResponse(results, media_type="text/event-stream")


app = gr.mount_gradio_app(app, demo, path="/")