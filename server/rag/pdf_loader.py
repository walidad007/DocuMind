from server.logger import logger
import os
from langchain_community.document_loaders import PyPDFLoader

# Calculate absolute path from this file's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "storage", "uploaded_pdfs")

# Create the folder automatically if it does not exist yet
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Print path at startup so you can verify
logger.info("===== PDF LOADER INIT =====")
logger.info("UPLOAD_FOLDER resolved to: %s", UPLOAD_FOLDER)
logger.info("Folder exists: %s", os.path.exists(UPLOAD_FOLDER))


async def save_uploaded_pdfs(files):
    """Save incoming PDF files from the user request to local storage."""

    for file in files:
        # Create the full path where the file will be saved
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        logger.info("saving file: %s", file.filename)
        logger.debug("full path: %s", file_path)

        # Read the file content asynchronously (without blocking the server)
        content = await file.read()

        # Safety check: Skip if the file is empty or corrupted
        if not content:
            logger.error("File is empty: %s", file.filename)
            continue

        # Open the file in 'write-binary' mode and save the bytes
        with open(file_path, "wb") as f:
            f.write(content)

        # Verify file actually saved on disk
        if os.path.exists(file_path):
            logger.info(
                "SUCCESS — %s (%d bytes)", file.filename, os.path.getsize(file_path)
            )
        else:
            logger.error("File NOT found after saving: %s", file_path)

    return "PDFs uploaded successfully"


def load_pdfs():
    """Read all PDFs from the folder and parse them into LangChain documents."""

    documents = []

    logger.info("===== PDF LOADER =====")
    logger.info("Reading from: %s", UPLOAD_FOLDER)
    logger.debug("Files found: %s", os.listdir(UPLOAD_FOLDER))

    # Loop through all files inside the upload folder
    for filename in os.listdir(UPLOAD_FOLDER):
        # Process only PDF files
        if filename.endswith(".pdf"):
            path = os.path.join(UPLOAD_FOLDER, filename)

            logger.info("Loading: %s", path)

            # Initialize LangChain's PDF loader and extract text content
            loader = PyPDFLoader(path)
            docs = loader.load()

            logger.debug("Pages loaded: %d", len(docs))

            # Add the extracted pages to our main documents list
            documents.extend(docs)

    logger.info("Total documents loaded: %d", len(documents))

    return documents


# ==========================================================
# Delete all uploaded PDF files
# Used when user clears Knowledge Base
# ==========================================================


def clear_uploaded_pdfs():
    """Delete all PDF files from the upload folder to clear the knowledge base."""
    logger.info("===== CLEARING PDF FOLDER =====")

    # Loop through the folder and delete each PDF file one by one
    for filename in os.listdir(UPLOAD_FOLDER):

        if filename.endswith(".pdf"):

            file_path = os.path.join(UPLOAD_FOLDER, filename)

            os.remove(file_path)

            logger.info("Deleted: %s", filename)

    print("All uploaded PDFs removed.")
