import logging
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from server.rag.chat_service import ask_question
from server.rag.pdf_loader import save_uploaded_pdfs, load_pdfs, clear_uploaded_pdfs
from server.rag.text_splitter import split_documents
from server.rag.vectorstore import create_vectorstore, clear_vectorstore

# Initialize logger
logger = logging.getLogger(__name__)

app = FastAPI()

logger.info("MAIN.PY APP INITIALIZED")


@app.get("/")
def home():
    """Root endpoint to check if the API is alive and running."""
    return {"message": "RAG Chatbot API Running"}


@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload multiple PDFs, process them into chunks, and save to Vector DB."""
    logger.info("Upload endpoint triggered. Files received: %d", len(files))

    try:
        # Step 1: Save files to disk memory asynchronously
        await save_uploaded_pdfs(files)

        # Step 2: Load the raw documents using LangChain loader
        documents = load_pdfs()
        logger.info("Documents loaded successfully. Count: %d", len(documents))

        # Step 3: Split the documents into smaller text pieces
        chunks = split_documents(documents)
        logger.info("Chunks created successfully. Count: %d", len(chunks))

        # Step 4: Store chunks into the vector database
        # (Yahan try-except laga diya taake model download ya DB error pakda ja sake)
        logger.info("Starting embedding generation and vector storage...")
        vectorstore = create_vectorstore(chunks)
        
        count = vectorstore._collection.count()
        logger.info("Documents successfully indexed. Database total: %d", count)

        return {
            "status": "success",
            "message": "PDF processed successfully",
        }

    except Exception as e:
        # Pura error detail backend logs/terminal mein print karein
        logger.error("CRITICAL ERROR DURING UPLOAD PIPELINE: %s", str(e), exc_info=True)
        
        # Proper JSON error bhein frontend ko taake JSONDecodeError na aaye
        raise HTTPException(status_code=500, detail=f"Backend Error: {str(e)}")


@app.post("/clear-kb")
def clear_kb():
    """Wipe out the knowledge base by deleting local files and vector DB."""
    logger.info("Clear Knowledge Base endpoint triggered.")
    clear_uploaded_pdfs()
    clear_vectorstore()
    logger.info("Knowledge Base cleared successfully.")
    return {"status": "success", "message": "Knowledge Base cleared successfully"}


@app.post("/chat")
def chat(query: str):
    """Answer user questions based on the uploaded data context."""
    logger.info("Chat endpoint triggered with query: %s", query)
    result = ask_question(query)

    if result is None:
        logger.warning("Chat service returned None for query: %s", query)
        return {
            "query": query,
            "answer": "No relevant information found.",
            "sources": [],
        }

    return {
        "query": query, 
        "answer": result["answer"], 
        "sources": result["sources"]
    }