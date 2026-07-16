from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from server.config import GROQ_API_KEY
from server.logger import logger


# --------------------------------------------------------------
# Custom prompt to control LLM behavior
# Without this, LLM may answer from its own training data
# With this, LLM is forced to answer ONLY from uploaded documents
# --------------------------------------------------------------
prompt_template = """You are a helpful assistant.
Answer the question based only on the context below.
If the answer is not in the context, say "I don't know based on the uploaded documents."

Context:
{context}

Question: {question}
Answer:"""

# Convert raw string into a LangChain PromptTemplate object
# {context} = retrieved chunks from ChromaDB
# {question} = user's query from the UI
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# --------------------------------------------------------------

def build_rag_chain(retriever):
    """Assemble the final RAG (Retrieval-Augmented Generation) pipeline."""
    logger.info("Starting RAG chain building process...")

    # Initialize Groq LLM — fast inference, LLaMA 3.3 70B model
    # API key comes from config.py which loads it from .env file
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

    # Connect three components into one pipeline:
    # retriever → finds relevant chunks from ChromaDB
    # llm       → generates answer from those chunks
    # prompt    → controls how LLM uses the chunks
    # return_source_documents → keeps page number + filename for citations
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    logger.info("RAG chain built successfully and ready for queries.")
    return qa_chain