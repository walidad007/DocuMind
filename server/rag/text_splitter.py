from langchain_text_splitters import RecursiveCharacterTextSplitter
from server.config import CHUNK_SIZE, CHUNK_OVERLAP
from server.logger import logger

def split_documents(documents):
    """Split long documents into smaller chunks for the RAG pipeline."""

    # Initialize the splitter with dynamic configurations (size and overlap)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # Execute the splitting process on the incoming documents
    chunks = splitter.split_documents(documents)

    # testing
    logger.info("Total chunks generated: %d", len(chunks))
    
    # Log a small preview of the very first chunk for debugging verification
    if chunks:
        logger.debug("First chunk preview: %s", chunks[0].page_content[:200])

    return chunks