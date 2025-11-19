from pymongo import MongoClient
from dotenv import load_dotenv
from fastapi import HTTPException
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

# Connect to Mongo Atlas
try:
    client = MongoClient(MONGO_URI)
    print( "Connection Succeded")
except Exception as e:
    print("❌ ERROR:", str(e))
    raise HTTPException(status_code=500, detail="Internal Server Error")

db = client["AI_Analytics"]
users_collection = db["Users"]
project_collection = db["Projects"]
conversations_collection = db["Conversations"]
datasets_collection = db["datasets"]
charts_collection = db["charts"]
