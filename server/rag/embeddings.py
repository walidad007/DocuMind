from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from server.config import EMBEDDING_MODEL
from server.logger import logger

def get_embedding_model():
    """Load and initialize the HuggingFace BGE embedding model."""
    logger.info("Initializing embedding model: %s", EMBEDDING_MODEL)

    try:
        # Initialize the HuggingFace BGE model configuration
        embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL)
        
        logger.info("Embedding model loaded successfully.")
        return embeddings

    except Exception as e:
        # Catch and log any errors if the model fails to download or load
        logger.error("Failed to load embedding model: %s", str(e))
        raise e