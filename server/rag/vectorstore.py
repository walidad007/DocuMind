import os

from langchain_chroma import Chroma
from server.rag.embeddings import get_embedding_model
from server.config import CHROMA_DB_PATH as DB_PATH, COLLECTION_NAME
from server.logger import logger

# Automatically create the chroma_db folder structure if missing
os.makedirs(DB_PATH, exist_ok=True)


def create_vectorstore(chunks):
    """Create a new vector database from text chunks."""
    # Get the embedding model to convert text into vectors
    embeddings = get_embedding_model()

    # Initialize Chroma database instance
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
        embedding_function=embeddings,
    )

    # Clean the old collection first to avoid duplicate data during testing
    vectorstore.delete_collection()

    # Re-initialize a fresh collection instance
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
        embedding_function=embeddings,
    )

    # Add text chunks and their generated embeddings to the database
    vectorstore.add_documents(chunks)

    # Track how many items were successfully stored
    count = vectorstore._collection.count()
    logger.info(
        "Vectorstore created | path=%s | collection=%s | count=%d",
        DB_PATH,
        COLLECTION_NAME,
        count,
    )

    return vectorstore

def load_vectorstore():
    """Load the existing vector store database from disk memory."""
    embeddings = get_embedding_model()

    # Connect to the saved database instance
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
        embedding_function=embeddings,
    )

    # Check the total document count currently saved in the database
    count = vectorstore._collection.count()
    logger.info(
        "Vectorstore loaded | path=%s | collection=%s | count=%d",
        DB_PATH,
        COLLECTION_NAME,
        count,
    )

    return vectorstore

def clear_vectorstore():
    """Delete the entire Chroma collection to clear the knowledge base."""
    embeddings = get_embedding_model()

    # Connect to the existing vector store
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
        embedding_function=embeddings,
    )

    # Physically delete the database collection from disk
    vectorstore.delete_collection()
    logger.info("===== VECTORSTORE CLEARED =====")


