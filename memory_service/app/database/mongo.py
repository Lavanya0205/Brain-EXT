from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)

db = client["digital_brain"]  # your database name
vector_collection = db["vector_memory"]  # collection for embeddings