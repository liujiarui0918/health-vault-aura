# Vital Key Chain

Vital Key Chain is a privacy-first health data product prototype built with a React frontend and a FastAPI backend.

The current product flow lets a user:

1. Connect a browser wallet.
2. Upload a text-based health report PDF.
3. Parse the report into structured JSON with an AI-assisted backend.
4. Review the extracted health indicators in a user-facing results page.
5. Continue into an on-chain publishing decision flow.
6. Choose which summary-level records should go on-chain.
7. Connect a wallet and complete a presentation-ready on-chain confirmation flow.

The UI is optimized for demos and product reviews, while the backend is functional enough to parse real PDF reports and return normalized JSON.

## Product Highlights

- Wallet-based sign in and sign up flow.
- PDF upload and health report parsing.
- Automatic JSON normalization and malformed field repair.
- User-facing health data review screen with categories, flagged values, and JSON export.
- On-chain review flow with selectable records, privacy messaging, wallet connection, and success receipt.
- Frontend and backend can run locally with a simple two-process setup.

## Stack

### Frontend

- React 18
- TypeScript
- Vite
- Tailwind CSS
- shadcn/ui
- React Router
- TanStack Query

### Backend

- FastAPI
- Python 3
- OpenAI-compatible client
- pypdf

## Repository Structure

```text
vital-key-chain/
├── src/
│   ├── components/           # shared UI and layouts
│   ├── contexts/             # wallet context
│   ├── hooks/                # wallet + UI hooks
│   ├── pages/                # product pages
│   └── test/                 # frontend tests
├── server/
│   ├── src/
│   │   ├── main.py           # FastAPI app and routes
│   │   ├── pdf_parser.py     # PDF text extraction
│   │   ├── summarizer.py     # AI parsing + JSON normalization
│   │   └── ark_client.py     # API client setup
│   ├── requirements.txt
│   └── README.md             # backend-specific notes
├── public/
├── package.json
└── README.md
```

## Main User Flows

### 1. Wallet login

Relevant files:

- `src/pages/Login.tsx`
- `src/pages/Signup.tsx`
- `src/hooks/use-wallet.ts`
- `src/contexts/WalletContext.tsx`

The app supports browser-wallet-based account connection. Wallet state is stored locally and shared through a React context.

### 2. Health report upload and review

Relevant files:

- `src/pages/HealthDataUpload.tsx`
- `server/src/main.py`
- `server/src/pdf_parser.py`
- `server/src/summarizer.py`

The upload page accepts PDF files, sends them to the FastAPI backend, receives structured JSON, repairs malformed response shapes, and renders the results in a user-facing review interface.

### 3. On-chain decision and publishing flow

Relevant files:

- `src/pages/HealthDataOnchain.tsx`
- `src/App.tsx`

After reviewing the parsed report, the user can continue to an on-chain review page, choose which records should be published, connect a wallet, and see a final success receipt with published items, protected fields, wallet address, transaction hash, and timestamp.

## Local Development

### Prerequisites

- Node.js 18+
- npm
- Python 3.11+ recommended
- A browser wallet such as MetaMask for the wallet flow

### Frontend setup

```bash
npm install
```

Start the frontend:

```bash
npm run dev -- --host 127.0.0.1 --port 8080
```

The frontend will be available at:

```text
http://127.0.0.1:8080
```

### Backend setup

```bash
cd server
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env
```

Add your API configuration to `server/.env`:

```env
ARK_API_KEY=your_api_key_here
ARK_BASE_URL=https://api.tu-zi.com/v1
ARK_MODEL=doubao-seed-1-6-flash-250828
```

Start the backend:

```bash
cd server
.venv/bin/uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

The backend will be available at:

```text
http://127.0.0.1:8000
```

### Run the full app

Terminal 1:

```bash
cd server
.venv/bin/uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

Terminal 2:

```bash
npm run dev -- --host 127.0.0.1 --port 8080
```

Then open:

```text
http://127.0.0.1:8080/health-data
```

To test the on-chain page directly:

```text
http://127.0.0.1:8080/health-data/onchain
```

## Environment Variables

### Frontend

- `VITE_API_BASE_URL`
  - Optional.
  - If omitted, the frontend uses the local Vite proxy and calls `/api/...`.

- `VITE_PROXY_TARGET`
  - Optional.
  - Defaults to `http://127.0.0.1:8000`.

### Backend

- `ARK_API_KEY`
  - Required for AI parsing.

- `ARK_BASE_URL`
  - Optional.
  - Defaults to `https://api.tu-zi.com/v1`.

- `ARK_MODEL`
  - Optional.
  - Controls which OpenAI-compatible model is used for parsing.

## Frontend Scripts

```bash
npm run dev
npm run build
npm run test
```

## Backend API

### Health check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

### Parse health report

```http
POST /api/health/parse
Content-Type: multipart/form-data
```

Request field:

- `file`: PDF file

Example response:

```json
{
  "fileName": "report.pdf",
  "contentType": "application/pdf",
  "indicatorCount": 3,
  "indicators": [
    {
      "id": "glucose",
      "name": "Glucose",
      "category": "Lab Results",
      "value": "5.8",
      "unit": "mmol/L",
      "referenceRange": "3.9-6.1",
      "status": "normal",
      "instrument": ""
    }
  ],
  "meta": {
    "model": "doubao-seed-1-6-flash-250828",
    "char_count": 1234,
    "chunk_count": 1,
    "page_count": 1,
    "filename": "report.pdf",
    "max_file_size_mb": 20,
    "ark_base_url": "https://api.tu-zi.com/v1"
  }
}
```

## Design Intent

This repository is not just an API demo. The frontend aims to look close to a real product:

- The health-data page is built for user review rather than raw developer inspection.
- The on-chain page is built for presentation and stakeholder walkthroughs.
- Product copy, layout, and visual hierarchy are optimized for demo readiness.

## Current Constraints

- Only text-based PDFs are supported. Image-only scanned PDFs are not supported yet.
- The on-chain flow is presentation-oriented and uses fixed frontend data for predictable output.
- Some wallet-related pages still contain mixed product and prototype behavior.
- `src/index.css` currently emits a non-blocking `@import` order warning during build.

## Recommended Next Steps

- Add persistent storage for uploaded reports and parsed records.
- Replace hard-coded on-chain selections with backend-backed records.
- Add a real save-to-vault action after report review.
- Add authenticated session handling around wallet login.
- Add end-to-end tests for upload, parse, and on-chain flows.

## Verification

The current branch has been validated with:

```bash
npm run build
npm run test
python3 -m compileall server/src
```

## License

No license file is currently included in this repository.
