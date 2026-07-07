from server.rag.rag_chain import build_rag_chain
from server.rag.retriever import get_retriever
from server.logger import logger

def ask_question(query):
    """
    Ask a question from the RAG chatbot.
    Retrieves relevant document chunks from the vector store
    and generates an answer using the LLM.
    """

    # Step 1: Create a fresh retriever instance to get latest database state
    retriever = get_retriever()

    # Step 2: Search the vector store for chunks relevant to the query
    retrieved_docs = retriever.invoke(query)

    # Step 3: Log retrieval info to track how many chunks matched
    logger.info("========== RETRIEVED DOCS ==========")
    logger.info("Retrieved Chunks count: %d", len(retrieved_docs))

    # Step 4: If no relevant chunks found, return early with a clear message
    # This prevents the LLM from giving a generic hallucinated answer
    if not retrieved_docs:
        logger.warning("No relevant chunks found for query: %s", query)
        return {
            "answer": "I could not find relevant information in the uploaded documents. Please make sure you have uploaded a PDF and try again.",
            "sources": [],
        }

    # Step 5: Log a clean preview of each retrieved chunk for debugging verification
    for doc in retrieved_docs:
        logger.debug("Chunk Content Preview: %s", doc.page_content[:500])

    # Step 6: Build a fresh RAG chain to make sure it picks up newly uploaded docs
    qa_chain = build_rag_chain()

    # Step 7: Run the query through the RAG chain (retriever + LLM combined)
    result = qa_chain.invoke({"query": query})

    # Step 8: Extract source metadata (page number, filename) from retrieved docs
    sources = []
    for doc in result["source_documents"]:
        page = doc.metadata.get("page", "Unknown")
        sources.append(
            {
                # Page numbers in PDF are 0-indexed, so add 1 for display
                "page": page + 1 if isinstance(page, int) else page,
                "source": doc.metadata.get("source", ""),
            }
        )

    # Step 9: Return the final answer and source references
    logger.info("Successfully generated answer from LLM.")
    return {"answer": result["result"], "sources": sources}