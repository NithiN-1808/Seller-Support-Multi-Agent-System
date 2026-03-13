"""
rag/retriever.py
Retrieval logic over the ChromaDB vector store.
"""

from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document

CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "seller_central_faq"


def get_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(model="mistral")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )


def retrieve(query: str, k: int = 4) -> list[Document]:
    """Retrieve top-k relevant chunks for a query."""
    vs = get_vectorstore()
    return vs.similarity_search(query, k=k)


def retrieve_with_scores(query: str, k: int = 4) -> list[tuple[Document, float]]:
    """Retrieve top-k chunks with similarity scores."""
    vs = get_vectorstore()
    return vs.similarity_search_with_score(query, k=k)