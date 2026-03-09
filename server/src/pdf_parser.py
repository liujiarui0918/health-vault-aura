from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader


class PDFParseError(Exception):
    pass


def extract_pdf_text(pdf_bytes: bytes) -> tuple[str, int]:
    if not pdf_bytes:
        raise PDFParseError("Uploaded file is empty.")

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception as exc:
        raise PDFParseError(f"Failed to read PDF file: {exc}") from exc

    page_count = len(reader.pages)
    texts: list[str] = []
    for page in reader.pages:
        try:
            texts.append((page.extract_text() or "").strip())
        except Exception:
            texts.append("")

    text = "\n\n".join(t for t in texts if t)
    if not text.strip():
        raise PDFParseError("No extractable text found in PDF. Only text-based PDFs are supported.")

    return text, page_count
