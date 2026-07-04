from server.rag.vectorstore import load_vectorstore
from server.logger import logger
def get_retriever():
    """Convert the loaded Vectorstore into a search retriever interface."""
    logger.info("Initializing document retriever...")

    # Load the existing database database instance
    vectorstore = load_vectorstore()

    # Configure the retriever to fetch top 3 most relevant context chunks (k=3)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    logger.info("Retriever initialized successfully with k=3.")
    return retriever
