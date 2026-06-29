from langchain_text_splitters import RecursiveCharacterTextSplitter
from server.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents):
    """
    Split documents into chunks.
    """

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    chunks = splitter.split_documents(documents)

    # testing
    print(f"Total chunks: {len(chunks)}")

    if chunks:
        print(chunks[0].page_content)

    return chunks
