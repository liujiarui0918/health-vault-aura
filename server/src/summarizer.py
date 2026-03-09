from __future__ import annotations

import json
import re
from typing import cast

from openai import OpenAI

from .ark_client import get_model_name, run_text_prompt

MAX_CHARS_PER_CHUNK = 6000
AI_TIMEOUT_SECONDS = 60


def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in text.split("\n\n"):
        p = paragraph.strip()
        if not p:
            continue

        p_len = len(p) + 2
        if current and current_len + p_len > max_chars:
            chunks.append("\n\n".join(current))
            current = [p]
            current_len = len(p)
        else:
            current.append(p)
            current_len += p_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _extract_json(text: str) -> dict[str, object]:
    text = text.strip()
    try:
        return cast(dict[str, object], json.loads(text))
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("Model response is not valid JSON.")
    return cast(dict[str, object], json.loads(match.group(0)))


def _build_prompt(text: str, filename: str) -> str:
    return f"""You are a professional medical data analysis assistant specializing in processing medical records, health examination reports, and other medical documents.
Please carefully read the following PDF content and extract all medical indicator information.

Strictly output in the following JSON format without any additional explanation:

{{
    "fileName": "{filename}",
    "contentType": "application/pdf",
    "indicatorCount": <total number of indicators>,
    "indicators": [
        {{
            "id": "<indicator ID in lowercase English, e.g. hba1c>",
            "name": "<indicator name, e.g. HbA1c>",
            "category": "<category, e.g. Lab Results, Blood Test, Imaging, etc.>",
            "value": "<value>",
            "unit": "<unit>",
            "referenceRange": "<reference range>",
            "status": "<status: normal/high/low>",
            "instrument": "<testing instrument or method>"
        }}
    ]
}}

Requirements:
1. Extract all recognizable medical indicators (including blood tests, biochemical indicators, imaging results, etc.)
2. indicatorCount must equal the length of the indicators array
3. Use empty string "" for any missing fields
4. Determine status by comparing value against reference range: normal/high/low
5. Output JSON only, no other text

PDF content:
{text}"""


def summarize_pdf_text(client: OpenAI, text: str, filename: str = "unknown.pdf") -> dict[str, object]:
    model = get_model_name()
    chunks = chunk_text(text)

    all_indicators: list[dict[str, object]] = []
    for chunk in chunks:
        prompt = _build_prompt(chunk, filename)
        response_text = run_text_prompt(client, prompt, model=model, timeout=AI_TIMEOUT_SECONDS)
        try:
            parsed = _extract_json(response_text)
            indicators = parsed.get("indicators", [])
            if isinstance(indicators, list):
                typed = cast(list[dict[str, object]], indicators)
                all_indicators.extend(typed)
        except ValueError:
            # Skip failed chunk without affecting other chunks
            continue

    return {
        "fileName": filename,
        "contentType": "application/pdf",
        "indicatorCount": len(all_indicators),
        "indicators": all_indicators,
        "meta": {
            "model": model,
            "char_count": len(text),
            "chunk_count": len(chunks),
        },
    }
