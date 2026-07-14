from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from server.config import GROQ_API_KEY
from server.logger import logger



def build_rag_chain(retriever):
    """Assemble the final RAG (Retrieval-Augmented Generation) pipeline."""
    logger.info("Starting RAG chain building process...")

    # Initialize the high-performance Llama-3.3 LLM using ChatGroq
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

    # Combine the LLM and Retriever into a complete Question-Answering (QA) chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True # Keeps track of which PDF page the answer came from
    )

    logger.info("RAG chain built successfully and ready for queries.")
    return qa_chain
