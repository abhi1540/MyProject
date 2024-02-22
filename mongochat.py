from fastapi import FastAPI, HTTPException, Cookie, Request
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

app = FastAPI()

# MongoDB connection settings
MONGODB_URL = "mongodb://192.168.56.1:27017/"
DB_NAME = "chat_app"
USER_COLLECTION_NAME = "useres"
CHAT_COLLECTION_NAME = "chat_history"


class MongoDBManager:
    """Manager class to handle MongoDB operations."""

    def __init__(self):
        self.client = AsyncIOMotorClient(MONGODB_URL)
        self.db = self.client[DB_NAME]

    def get_user_collection(self):
        return self.db[USER_COLLECTION_NAME]

    def get_chat_collection(self):
        return self.db[CHAT_COLLECTION_NAME]


mongo_manager = MongoDBManager()


@app.post("/users/")
async def create_user(user_data: dict):
    user_collection = mongo_manager.get_user_collection()
    print(user_collection)
    result = await user_collection.insert_one(user_data)
    print(result)
    created_user_id = str(result.inserted_id)
    return {"user_id": created_user_id}



@app.post("/messages/")
async def send_message(session_id: str, user_id: str, message: dict):
    chat_collection = mongo_manager.get_chat_collection()
    message["session_id"] = session_id
    message["user_id"] = user_id
    await chat_collection.insert_one(message)
    return {"message": "Message sent successfully"}


@app.get("/messages/")
async def get_messages(session_id: str, user_id: str):
    chat_collection = mongo_manager.get_chat_collection()
    messages = await chat_collection.find({"session_id": session_id, "user_id": user_id}).to_list(length=1000)
    messages[0].pop('_id')
    if not messages:
        raise HTTPException(status_code=404, detail="No messages found")
    return messages[0]


@app.delete("/messages/")
async def clear_messages(session_id: str, user_id: str):
    chat_collection = mongo_manager.get_chat_collection()
    await chat_collection.delete_many({"session_id": session_id, "user_id": user_id})
    return {"message": "Messages cleared successfully"}



(base) C:\Users\Abhisek>docker run -d -p 27017:27017 --name example-mongo -v mongo-data:/data/db -e MONGODB_INITDB_ROOT_USERNAME=mongo -e MONGODB_INITDB_ROOT_PASSWORD=mongo mongo:latest
d6b16cf52c25b496933e28ec1ddf075455c2d4a15a4420abc6ea4484c50a218a

docker logs example-mongo --follow
docker exec -it example-mongo mongosh
show dbs
show collections
db.chat_history.find()

(base) C:\Users\Abhisek>docker run -d -p 27017:27017 --name example-mongo -v mongo-data:/data/db mongo:latest
721998cf25cc50da208ffab862ee76e19c70fae71ddf9c8f1bb392be2ae782ff
