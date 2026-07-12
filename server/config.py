import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Get project root directory (DocuMind/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Shared storage path under root directory (storage/)
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

# Specific paths inside the main storage folder
CHROMA_DB_PATH = os.path.join(STORAGE_DIR, "chroma_db")
UPLOAD_FOLDER = os.path.join(STORAGE_DIR, "uploaded_pdfs")


COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
