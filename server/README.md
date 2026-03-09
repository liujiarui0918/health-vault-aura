# vital-key-chain Server

A FastAPI-based Python backend service for parsing health examination reports. Users upload a PDF report, and the service uses an AI model to extract structured health indicators and return them as JSON.

## Directory Structure

```text
server/
├── .env              # Local environment variables (not committed to git)
├── .env.example      # Environment variable template
├── requirements.txt  # Python dependencies
├── README.md
└── src/
    ├── __init__.py
    ├── main.py        # FastAPI app entry point & routes
    ├── ark_client.py  # AI client (Doubao model)
    ├── pdf_parser.py  # PDF text extraction
    └── summarizer.py  # LLM analysis → structured JSON
```

## Quick Start

### 1. Configure Environment Variables

```bash
cd server
cp .env.example .env
```

Edit `.env` and fill in your real API key:

```env
ARK_API_KEY=your_api_key_here
ARK_BASE_URL=https://api.tu-zi.com/v1
ARK_MODEL=doubao-seed-1-6-250615
```

### 2. Install Dependencies

```bash
# Recommended: use a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

python3 -m pip install -r requirements.txt
```

### 3. Start the Server

```bash
uvicorn src.main:app --reload --port 8000
```

The API will be available at `http://127.0.0.1:8000`.

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service status check |
| `GET` | `/health` | Health check, returns `{"status": "ok"}` |
| `POST` | `/api/health/parse` | Upload a PDF and receive structured health indicators |

### POST /api/health/parse

**Request**: `multipart/form-data`, field name `file`, `.pdf` files only, max 20MB.

**Response Example**:

```json
{
  "fileName": "report.pdf",
  "contentType": "application/pdf",
  "indicatorCount": 3,
  "indicators": [
    {
      "id": "hba1c",
      "name": "HbA1c",
      "category": "Lab Results",
      "value": "6.8",
      "unit": "%",
      "referenceRange": "4.0-5.6",
      "status": "high",
      "instrument": ""
    }
  ],
  "meta": {
    "model": "doubao-seed-1-6-250615",
    "char_count": 1234,
    "chunk_count": 1,
    "page_count": 2,
    "filename": "report.pdf"
  }
}
```

**Error Codes**:

| Status | Reason |
|--------|--------|
| 400 | Empty file, non-PDF format, or no extractable text in PDF |
| 413 | File exceeds 20MB |
| 500 | Server configuration error (e.g. API key not set) |
| 502 | AI model call failed |

## Notes

- Only **text-based PDFs** are supported; scanned (image-based) PDFs are not currently supported
- Long documents are automatically split into chunks (6000 chars each) and processed separately, with results merged
- AI API call timeout is 60 seconds
