from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .ark_client import build_ark_client
from .pdf_parser import PDFParseError, extract_pdf_text
from .summarizer import summarize_pdf_text

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


class ParsedIndicator(BaseModel):
    id: str
    name: str
    category: str
    value: str
    unit: str
    referenceRange: str
    status: str
    instrument: str = ""


class ParseMeta(BaseModel):
    model: str = ""
    char_count: int = 0
    chunk_count: int = 0
    page_count: int = 0
    filename: str = ""
    max_file_size_mb: int = 0
    ark_base_url: str = ""


class ParseHealthFileResponse(BaseModel):
    fileName: str
    contentType: str
    indicatorCount: int = Field(ge=0)
    indicators: list[ParsedIndicator]
    meta: ParseMeta

app = FastAPI(
    title="vital-key-chain server",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "vital-key-chain server is running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/health/parse", response_model=ParseHealthFileResponse)
async def parse_health_file(file: UploadFile = File(...)) -> ParseHealthFileResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename detected.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(pdf_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 20MB.")

    try:
        text, page_count = extract_pdf_text(pdf_bytes)
    except PDFParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        client = build_ark_client()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        result = summarize_pdf_text(client, text, filename=file.filename)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI analysis failed: {exc}") from exc

    meta = result.get("meta")
    if isinstance(meta, dict):
        meta["page_count"] = page_count
        meta["filename"] = file.filename
        meta["max_file_size_mb"] = MAX_FILE_SIZE // (1024 * 1024)
        meta["ark_base_url"] = os.getenv("ARK_BASE_URL", "https://api.tu-zi.com/v1")

    return ParseHealthFileResponse.model_validate(result)
