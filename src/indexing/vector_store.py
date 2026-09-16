import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from ..config import get_settings

_embeddings = None  # cached so the model is only loaded into memory once per process


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        settings = get_settings()
        _embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return _embeddings


def build_vector_store(documents: list[Document]) -> FAISS:
    """Builds a fresh FAISS index from documents and saves it to disk."""
    if not documents:
        raise ValueError("No documents provided — nothing to index.")
    settings = get_settings()
    store = FAISS.from_documents(documents, get_embeddings())
    os.makedirs(settings.vector_store_dir, exist_ok=True)
    store.save_local(settings.vector_store_dir)
    return store


def load_vector_store() -> FAISS | None:
    """Loads a previously saved FAISS index, or None if none exists yet."""
    settings = get_settings()
    index_file = os.path.join(settings.vector_store_dir, "index.faiss")
    if not os.path.exists(index_file):
        return None

    return FAISS.load_local(
        settings.vector_store_dir, get_embeddings(), allow_dangerous_deserialization=True
    )


def add_documents(store: FAISS, documents: list[Document]) -> FAISS:
    """Adds new documents to an existing index (e.g. a newly uploaded PDF) and re-saves."""
    settings = get_settings()
    store.add_documents(documents)
    store.save_local(settings.vector_store_dir)
    return store


def get_or_build_vector_store(documents: list[Document]) -> FAISS:
    """Loads the existing index if present, otherwise builds one from the given documents."""
    store = load_vector_store()
    if store is not None:
        return store
    return build_vector_store(documents)
