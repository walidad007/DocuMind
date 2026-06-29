from langchain_chroma import Chroma
from server.rag.embeddings import get_embedding_model
from server.config import CHROMA_DB_PATH as DB_PATH, COLLECTION_NAME



def create_vectorstore(chunks):
    embeddings = get_embedding_model()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
        embedding_function=embeddings,
    )

    # Testing phase
    vectorstore.delete_collection()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
        embedding_function=embeddings,
    )

    vectorstore.add_documents(chunks)

    # print("Chunks stored:", len(chunks))
    # print("DB count:", vectorstore._collection.count())

    # ===============Temporarily==============
    print("\n===== CREATE VECTORSTORE =====")
    print("DB_PATH =", DB_PATH)
    print("COLLECTION =", COLLECTION_NAME)
    print("COUNT =", vectorstore._collection.count())

    return vectorstore


# ==========================================================
# Clear existing Chroma collection
# Used when user wants a fresh Knowledge Base
# ==========================================================


def clear_vectorstore():
    """
    Delete the entire Chroma collection.

    This removes all embeddings and vectors
    stored in the database.
    """

    embeddings = get_embedding_model()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
        embedding_function=embeddings,
    )

    vectorstore.delete_collection()

    print("\n===== VECTORSTORE CLEARED =====")


# ===============Temporarily==============


def load_vectorstore():
    embeddings = get_embedding_model()

    print("\n===== LOAD VECTORSTORE =====")
    print("DB_PATH =", DB_PATH)
    print("COLLECTION =", COLLECTION_NAME)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
        embedding_function=embeddings,
    )

    count = vectorstore._collection.count()

    print("COUNT =", count)

    return vectorstore
