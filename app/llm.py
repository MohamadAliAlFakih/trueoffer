# app/llm.py
# Groq API wrapper — single call function

import json
import os
from pathlib import Path

from groq import Groq

from app.schemas import GroqError, ParseError

# Model sourced from env with fallback constant (never hardcode the key)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def call_groq(system_prompt: str, user_message: str, *, json_mode: bool = True) -> str:
    """
    Call Groq API and return the raw text response.

    Raises:
        GroqError: if the API call fails for any reason
        ParseError: if json_mode=True and response is not valid JSON
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise GroqError("GROQ_API_KEY environment variable not set")

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            response_format={"type": "json_object"} if json_mode else {"type": "text"},
        )
        text = response.choices[0].message.content or ""
    except Exception as exc:
        raise GroqError(f"Groq API call failed: {exc}") from exc

    if json_mode:
        try:
            json.loads(text)   # validate parseable; caller receives raw text
        except json.JSONDecodeError as exc:
            raise ParseError(f"Groq returned non-JSON: {text[:200]}") from exc

    return text


def load_prompt(name: str) -> str:
    """Load a prompt from prompts/{name}.txt relative to project root."""
    root = Path(__file__).resolve().parent.parent
    path = root / "prompts" / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()
