from __future__ import annotations

import json
import re
from typing import cast

from openai import OpenAI

from .ark_client import get_model_name, run_text_prompt

MAX_CHARS_PER_CHUNK = 6000
AI_TIMEOUT_SECONDS = 60
DEFAULT_CATEGORY = "Lab Results"
DEFAULT_CONTENT_TYPE = "application/pdf"
STATUS_MAP = {
    "normal": "normal",
    "正常": "normal",
    "高": "high",
    "high": "high",
    "偏高": "high",
    "low": "low",
    "低": "low",
    "偏低": "low",
    "abnormal": "abnormal",
    "异常": "abnormal",
}
RESULT_LIST_KEYS = ("indicators", "items", "results", "data", "records")
NAME_KEYS = ("name", "indicatorName", "label", "title", "item")
ID_KEYS = ("id", "indicatorId", "slug", "code")
CATEGORY_KEYS = ("category", "type", "group", "section")
VALUE_KEYS = ("value", "result", "measurement", "finding")
UNIT_KEYS = ("unit", "units")
REFERENCE_RANGE_KEYS = ("referenceRange", "reference_range", "reference", "range")
STATUS_KEYS = ("status", "flag", "statusFlag", "resultFlag")
INSTRUMENT_KEYS = ("instrument", "method", "testMethod", "device")


def _find_json_block(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    object_match = re.search(r"\{[\s\S]*\}", text)
    if object_match:
        return object_match.group(0)

    array_match = re.search(r"\[[\s\S]*\]", text)
    if array_match:
        return array_match.group(0)

    raise ValueError("Model response is not valid JSON.")


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


def _extract_json(text: str) -> object:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    return json.loads(_find_json_block(text))


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
4. Determine status by comparing value against reference range: normal/high/low. If comparison is impossible, use abnormal or "".
5. Output JSON only, no other text
6. For Chinese reports, keep the original indicator names but normalize status to English
7. Prefer category values such as Lab Results, Imaging / Reports, Vitals, Conditions & Diagnoses
8. Must output in English

PDF content:
{text}"""


def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _slugify(value: str) -> str:
    lowered = value.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or "indicator"


def _normalize_status(value: str) -> str:
    normalized = STATUS_MAP.get(value.strip().lower())
    if normalized:
        return normalized

    raw = value.strip()
    if raw in STATUS_MAP:
        return STATUS_MAP[raw]
    return raw.lower() if raw else ""


def _pick_value(source: dict[str, object], keys: tuple[str, ...]) -> object:
    for key in keys:
        if key in source:
            return source[key]
    return ""


def _coerce_result_object(result: object) -> dict[str, object]:
    if isinstance(result, dict):
        return result
    if isinstance(result, list):
        return {"indicators": result}
    return {}


def _extract_indicator_list(result: dict[str, object]) -> list[object]:
    for key in RESULT_LIST_KEYS:
        candidate = result.get(key)
        if isinstance(candidate, list):
            return candidate
    return []


def _normalize_indicator_item(item: dict[str, object], index: int, seen_ids: dict[str, int]) -> dict[str, str]:
    name = _stringify(_pick_value(item, NAME_KEYS)) or f"Indicator {index}"
    base_id = _stringify(_pick_value(item, ID_KEYS)) or _slugify(name)
    seen_ids[base_id] = seen_ids.get(base_id, 0) + 1
    indicator_id = base_id if seen_ids[base_id] == 1 else f"{base_id}-{seen_ids[base_id]}"

    return {
        "id": indicator_id,
        "name": name,
        "category": _stringify(_pick_value(item, CATEGORY_KEYS)) or DEFAULT_CATEGORY,
        "value": _stringify(_pick_value(item, VALUE_KEYS)),
        "unit": _stringify(_pick_value(item, UNIT_KEYS)),
        "referenceRange": _stringify(_pick_value(item, REFERENCE_RANGE_KEYS)),
        "status": _normalize_status(_stringify(_pick_value(item, STATUS_KEYS))),
        "instrument": _stringify(_pick_value(item, INSTRUMENT_KEYS)),
    }


def normalize_parse_result(
    result: dict[str, object],
    *,
    filename: str,
    content_type: str = DEFAULT_CONTENT_TYPE,
) -> dict[str, object]:
    result = _coerce_result_object(result)
    raw_indicators = _extract_indicator_list(result)
    seen_ids: dict[str, int] = {}
    indicators: list[dict[str, str]] = []

    if isinstance(raw_indicators, list):
        for index, item in enumerate(raw_indicators, start=1):
            if not isinstance(item, dict):
                continue
            indicators.append(_normalize_indicator_item(item, index, seen_ids))

    meta = result.get("meta")
    normalized_meta = meta if isinstance(meta, dict) else {}

    return {
        "fileName": _stringify(result.get("fileName") or result.get("file_name")) or filename,
        "contentType": _stringify(result.get("contentType") or result.get("content_type")) or content_type,
        "indicatorCount": len(indicators),
        "indicators": indicators,
        "meta": normalized_meta,
    }


def summarize_pdf_text(client: OpenAI, text: str, filename: str = "unknown.pdf") -> dict[str, object]:
    model = get_model_name()
    chunks = chunk_text(text)

    all_indicators: list[dict[str, object]] = []
    for chunk in chunks:
        prompt = _build_prompt(chunk, filename)
        response_text = run_text_prompt(client, prompt, model=model, timeout=AI_TIMEOUT_SECONDS)
        try:
            parsed = _extract_json(response_text)
            result = _coerce_result_object(parsed)
            indicators = _extract_indicator_list(result)
            if isinstance(indicators, list):
                typed = cast(list[dict[str, object]], indicators)
                all_indicators.extend(typed)
        except ValueError:
            # Skip failed chunk without affecting other chunks
            continue

    return normalize_parse_result(
        {
            "fileName": filename,
            "contentType": DEFAULT_CONTENT_TYPE,
            "indicatorCount": len(all_indicators),
            "indicators": all_indicators,
            "meta": {
                "model": model,
                "char_count": len(text),
                "chunk_count": len(chunks),
            },
        },
        filename=filename,
    )
