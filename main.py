import argparse
import uuid

from src.config import get_settings
from src.ingestion.pdf_loader import load_and_chunk_pdfs
from src.ingestion.sheets_loader import load_sheet_as_documents
from src.indexing.vector_store import build_vector_store, load_vector_store
from src.pipeline.rag_pipeline import RAGAssistant


def build_index():
    """Loads all PDFs (and the Google Sheet, if configured) and builds the FAISS index."""
    settings = get_settings()
    print(f"Loading PDFs from {settings.uploads_dir} ...")
    pdf_chunks = load_and_chunk_pdfs()
    print(f"  -> {len(pdf_chunks)} chunks")

    sheet_docs = []
    if settings.google_sheet_id:
        print("Loading Google Sheet ...")
        sheet_docs = load_sheet_as_documents()
        print(f"  -> {len(sheet_docs)} rows")

    all_docs = pdf_chunks + sheet_docs
    if not all_docs:
        print("No documents found — add PDFs to data/uploads/ or set GOOGLE_SHEET_ID.")
        return None

    store = build_vector_store(all_docs)
    print(f"Index built with {len(all_docs)} chunks -> saved to {settings.vector_store_dir}")
    return store


def chat_loop():
    store = load_vector_store()
    if store is None:
        print("No index found. Run with --build first.")
        return

    assistant = RAGAssistant(store)
    session_id = str(uuid.uuid4())
    print(f"Chat session {session_id[:8]} started. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        result = assistant.ask(question, session_id=session_id)
        print(f"\nAssistant: {result['answer']}\n")
        sources = {d.metadata.get("file_name", d.metadata.get("source_type")) for d in result["source_documents"]}
        if sources:
            print(f"  (sources: {', '.join(sorted(sources))})\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="(Re)build the vector index from data/uploads/")
    parser.add_argument("--chat", action="store_true", help="Start an interactive chat session")
    args = parser.parse_args()

    if args.build:
        build_index()
    elif args.chat:
        chat_loop()
    else:
        print("Usage: python main.py --build   (index your documents)")
        print("       python main.py --chat    (start chatting)")
