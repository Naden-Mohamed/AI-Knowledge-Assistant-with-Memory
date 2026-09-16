import gspread
from google.oauth2.service_account import Credentials
from langchain_core.documents import Document

from ..config import get_settings

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


def _get_client() -> gspread.Client:
    settings = get_settings()
    creds = Credentials.from_service_account_file(
        settings.google_service_account_json, scopes=SCOPES
    )
    return gspread.authorize(creds)


def load_sheet_as_documents(sheet_id: str | None = None, worksheet_index: int = 0) -> list[Document]:
    """Reads a Google Sheet and converts each row into one Document.
    Each row's columns become 'field: value' lines so the text stays
    readable and searchable after embedding — a raw comma-joined row
    loses the column meaning once it's just embedded text."""
    settings = get_settings()
    sheet_id = sheet_id or settings.google_sheet_id
    if not sheet_id:
        return []

    client = _get_client()
    worksheet = client.open_by_key(sheet_id).get_worksheet(worksheet_index)
    records = worksheet.get_all_records()  # list of dicts, keyed by header row

    documents: list[Document] = []
    for i, row in enumerate(records):
        row_text = "\n".join(f"{key}: {value}" for key, value in row.items() if str(value).strip())
        if not row_text.strip():
            continue
        documents.append(
            Document(
                page_content=row_text,
                metadata={"source_type": "google_sheet", "sheet_id": sheet_id, "row_index": i},
            )
        )

    return documents
