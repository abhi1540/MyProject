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
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.llms import LlamaCpp
from langchain.chains import ConversationChain, LLMChain, ConversationalRetrievalChain, RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
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
from llama_cpp import Llama
from queue import SimpleQueue
from langchain.schema import LLMResult
from threading import Thread
import gradio as gr

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

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

# template = """Use the following pieces of information to answer the user's question.
#     If you don't know the answer, just say that you don't know, don't try to make up an answer.

#     Context: {context}
#     Question: {question}

#     Only return the helpful answer below and nothing else.
#     Helpful answer:
#     """

    # prompt = PromptTemplate(input_variables=["context",  "question"], template=template)
embeddings = HuggingFaceEmbeddings(model_name=r"D:\Workspace\DocChat\models\embedding\all-mpnet-base-v2", model_kwargs={"device": 'cuda'})
CONNECTION_STRING = "postgresql+psycopg2://postgres:admin@localhost:5432/vector_db"
COLLECTION_NAME = 'document_vector'
db = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)

    # llm = Llama(model_path='D:/Workspace/PrivateGPTLangchain/models/llm/mistral-7b-v0.1.Q4_0.gguf',
    #         n_ctx=10000,
    #         max_tokens=250,
    #         n_batch=512,
    #         n_threads=max(os.cpu_count() // 2, 1),
    #         n_threads_batch=max(os.cpu_count() // 2, 1),
    #         callbacks=[callback],
    #         n_gpu_layers=-1)
    # Define prompts and initialize conversation chain
    
chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',retriever=db.as_retriever())

    
    # qa = RetrievalQA.from_chain_type(
    #         llm=llm,
    #         chain_type="stuff",
    #         retriever=db.as_retriever(),
    #         return_source_documents=True,
    #         #chain_type_kwargs={'prompt': prompt}
    # )
    # task = asyncio.create_task(
    #     qa.ainvoke(query)
    #     #qa(query)
    # )
    # task = asyncio.create_task(
    #     qa.ainvoke(query)
    # )
    # try:
    #     async for token in callback.aiter():
    #         yield token
    # except Exception as e:
    #     print(f"Caught exception: {e}")
    # finally:
    #     callback.done.set()

    # await task
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

# class ChainRequest(BaseModel):
#     message: str
    

# app = FastAPI()
# # CORS configuration
# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# @app.post("/chain")
# async def _chain(request: ChainRequest):
#     gen = llm_thread(request.message)
#     return StreamingResponse(gen, media_type="text/event-stream")


if __name__ == '__main__':
    demo.queue().launch()
    
# docker load -i D:\Software\Docker_Images\pgvector-containerimage.tar
# (base) D:\Software\Docker_Images>docker run --name pgvector-image -e POSTGRES_PASSWORD=admin -p 5432:5432 -d ankane/pgvector:latest
# 77e30db6c3b97855737d01fac20f5fa70c30c73fac3a28f5744b588ab443be8e
# docker start 77e30db6c3b97855737d01fac20f5fa70c30c73fac3a28f5744b588ab443be8e

 #   uvicorn.run('main:app', host='0.0.0.0', port=8010, reload=True)
    
    # embeddings = HuggingFaceEmbeddings(model_name=r"D:\Workspace\DocChat\models\embedding\all-mpnet-base-v2", model_kwargs={"device": 'cuda'})

    # CONNECTION_STRING = "postgresql+psycopg2://postgres:admin@localhost:5432/vector_db"
    # COLLECTION_NAME = 'document_vector'
    # db = PGVector(
    #     collection_name=COLLECTION_NAME,
    #     connection_string=CONNECTION_STRING,
    #     embedding_function=embeddings,
    # )
    # # template = """You are helpful information giving QA System and make sure you don't answer anything 
    # # # not related to following context. You are always provide useful information & details available in the given context. Use the following pieces of context to answer the question at the end. 
    # # # If you don't know the answer, just say that you don't know, don't try to make up an answer. 

    # # # {context}

    # # # Question: {question}
    # # # Helpful Answer:"""
    # template = """Use the following pieces of information to answer the user's question.
    # If you don't know the answer, just say that you don't know, don't try to make up an answer.

    # Context: {context}
    # Question: {question}

    # Only return the helpful answer below and nothing else.
    # Helpful answer:
    # """

    # prompt = PromptTemplate(
    # input_variables=["context",  "question"], template=template)
    # connection_string = f"mongodb://localhost:27017/"
    # history = MongoDBChatMessageHistory(
    #     connection_string=connection_string, session_id="user_id"
    # )
    # # Initialize the ConversationBufferMemory
    # memory = ConversationBufferMemory(
    #     memory_key="chat_history",          # Ensure this matches the key used in chain's prompt template
    #     chat_memory=history,   # Pass the RedisChatMessageHistory instance
    #     return_messages=True,          # Does your prompt template expect a string or a list of Messages?
    #     k = 5,
    #     output_key='answer'
    # )
    # llm = LlamaCpp(model_path='D:/Workspace/PrivateGPTLangchain/models/llm/mistral-7b-v0.1.Q4_0.gguf',
    #            n_ctx=10000,
    #            max_tokens=250,
    #            n_batch=512,
    #            callbacks=[StreamingStdOutCallbackHandler()],
    #            n_gpu_layers=-1)
    # # qa = RetrievalQA.from_chain_type(
    # # llm=llm,
    # # chain_type="stuff",
    # # retriever=db.as_retriever(),
    # # return_source_documents=True,
    # # memory=memory,
    # # chain_type_kwargs={'prompt': prompt}
    # # )
    # qa = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     memory=memory,
    #     chain_type="stuff",
    #     retriever=db.as_retriever(),
    #     return_source_documents=True,
    #     combine_docs_chain_kwargs={"prompt": prompt},
    #     verbose=False)
    # while True:
    #     query = input()
    #     result = qa({"question": query})
    #     result