from fastapi import FastAPI, UploadFile, File
from server.rag.chat_service import ask_question
from server.rag.pdf_loader import save_uploaded_pdfs, load_pdfs, clear_uploaded_pdfs
from server.rag.text_splitter import split_documents
from server.rag.vectorstore import (
    create_vectorstore,
    clear_vectorstore,
)
from typing import List
from server.logger import logger

app = FastAPI()

logger.info("MY MAIN.PY LOADED")

@app.get("/")
def home():
    """Root endpoint to check if the API is alive and running."""
    return {"message": "RAG Chatbot API Running"}


@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload multiple PDFs, process them into chunks, and save to Vector DB."""

    logger.info("Upload endpoint triggered. Files received: %d", len(files))


    # Log each filename for incoming request verification
    for f in files:
        logger.debug("Processing incoming file: %s", f.filename)
    
    # Step 1: Save files to disk memory asynchronously
    await save_uploaded_pdfs(files)

    # Step 2: Load the raw documents using LangChain loader
    documents = load_pdfs()
    logger.info("Document loaded successfully. Count: %d", len(chunks))
    
    # Step 3: Split the documents into smaller text pieces
    chunks = split_documents(documents)
    logger.info("Chunks created successfully. Count %d", len(chunks))

    # Step 4: Store chunks into the vector database
    vectorstore = create_vectorstore(chunks)
    lgger.info("Document successfully indexed in vector store. Database total: %d",vectorstore._collection.count())
    

    return {
        "status": "success",
        "message": "PDF processed successfully",
    }


# ==========================================================
# Clear Knowledge Base
#
# Removes:
# 1. All uploaded PDFs
# 2. All vector embeddings
#
# Allows user to start with a fresh dataset
# ==========================================================


@app.post("/clear-kb")
def clear_kb():
    """Wipe out the knowledge base by deleting local files and vector DB."""
    logger.info("Clear Knowledge Base endpoint triggered.")
    
    # Remove physical files from the storage folder
    clear_uploaded_pdfs()

    # Remove data collections from the vector database
    clear_vectorstore()

    logger.info("Knowledge Base cleared successfully.")
    return {"status": "success", "message": "Knowledge Base cleared successfully"}


@app.post("/chat")
def chat(query: str):
    """Answer user questions based on the uploaded data context."""
    logger.info("Chat endpoint triggered with query: %s", query)

    # Get answer and sources from chat service
    result = ask_question(query)

    # Safety check if the service returns an empty or None response
    if result is None:
        logger.warning("Chat service returned None for query: %s", query)
        return {
            "query": query,
            "answer": "No relevant information found.",
            "sources": [],
        }

    logger.info("Chat response generated successfully.")
    return {"query": query, "answer": result["answer"], "sources": result["sources"]}
