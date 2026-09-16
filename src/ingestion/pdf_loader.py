import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ..config import get_settings


def load_pdfs(uploads_dir: str | None = None) -> list[Document]:
    """Load every PDF in the uploads directory into LangChain Documents,
    one Document per page, tagged with source metadata."""
    settings = get_settings()
    uploads_dir = uploads_dir or settings.uploads_dir

    if not os.path.isdir(uploads_dir):
        return []

    documents: list[Document] = []
    for file_name in os.listdir(uploads_dir):
        if not file_name.lower().endswith(".pdf"):
            continue
        file_path = os.path.join(uploads_dir, file_name)
        try:
            pages = PyPDFLoader(file_path).load()
        except Exception as e:
            print(f"[pdf_loader] failed to load {file_name}: {e}")
            continue
        for page in pages:
            page.metadata["source_type"] = "pdf"
            page.metadata["file_name"] = file_name
        documents.extend(pages)

    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split loaded documents into overlapping chunks sized for embedding + retrieval."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def load_and_chunk_pdfs(uploads_dir: str | None = None) -> list[Document]:
    return chunk_documents(load_pdfs(uploads_dir))
