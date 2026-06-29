from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from server.config import EMBEDDING_MODEL

def get_embedding_model():
    """
    Load embedding model.
    """

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return embeddings
