from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

_ = load_dotenv()


def build_ark_client() -> OpenAI:
    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        raise ValueError("ARK_API_KEY is not set. Please configure it in server/.env.")
    base_url = os.getenv("ARK_BASE_URL", "https://api.tu-zi.com/v1")
    return OpenAI(base_url=base_url, api_key=api_key)


def get_model_name() -> str:
    return os.getenv("ARK_MODEL", "doubao-seed-1-6-flash-250828")


def run_text_prompt(
    client: OpenAI,
    prompt: str,
    model: Optional[str] = None,
    timeout: Optional[float] = 60.0,
) -> str:
    response = client.chat.completions.create(
        model=model or get_model_name(),
        messages=[
            {"role": "user", "content": prompt}
        ],
        timeout=timeout,
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        }
    )
    return response.choices[0].message.content or ""
