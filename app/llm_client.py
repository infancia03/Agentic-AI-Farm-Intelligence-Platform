"""
AgriFarm — LLM Client
OpenRouter wrapper with automatic free-model fallback.
Primary model switched to meta-llama/llama-3.3-70b-instruct:free
which is ~5x faster than Nemotron on the free tier.
"""

from __future__ import annotations
import os
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

_client: OpenAI | None = None

# Ordered by speed on free tier — fastest first
# Change LLM_MODEL in .env to override the primary
FREE_MODELS = [
    os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
    "mistralai/mistral-7b-instruct:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "meta-llama/llama-3.1-405b-instruct:free",
]

# Smaller token limits for quick calls — reduces latency significantly
QUICK_MAX_TOKENS  = 512
DETAIL_MAX_TOKENS = 1024


def get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.getenv("OPENROUTER_API_KEY", "")
        if not key:
            raise ValueError("OPENROUTER_API_KEY not set. Add it to your .env file.")
        _client = OpenAI(
            api_key=key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            timeout=90,
            default_headers={
                "HTTP-Referer": "https://github.com/agrifarm",
                "X-Title":      "AgriFarm Intelligence Platform",
            },
        )
    return _client


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=6), reraise=True)
def chat(
    messages:    list[dict],
    model:       str | None = None,
    temperature: float = 0.3,
    max_tokens:  int   = DETAIL_MAX_TOKENS,
    system:      str | None = None,
    tools:       list | None = None,
    tool_choice: str  = "auto",
) -> dict:
    """
    Chat call with free-model fallback.
    Returns dict with 'content' and 'tool_calls' keys.
    """
    client = get_client()
    if system:
        messages = [{"role": "system", "content": system}] + messages

    models_to_try = ([model] if model else []) + FREE_MODELS
    last_err = None

    for m in models_to_try:
        try:
            kwargs: dict = dict(
                model=m,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if tools:
                kwargs["tools"]       = tools
                kwargs["tool_choice"] = tool_choice

            logger.debug(f"LLM → {m}")
            resp = client.chat.completions.create(**kwargs)
            msg  = resp.choices[0].message
            content = msg.content or ""
            logger.debug(f"LLM ← {len(content)} chars from {m}")
            return {
                "content":    content,
                "tool_calls": msg.tool_calls or [],
                "role":       "assistant",
                "model_used": m,
            }
        except Exception as e:
            logger.warning(f"Model {m} failed: {e} — trying next")
            last_err = e

    raise RuntimeError(f"All free models failed. Last error: {last_err}")


def quick_ask(
    prompt: str,
    system: str = "You are a helpful agricultural AI.",
    max_tokens: int = QUICK_MAX_TOKENS,
) -> str:
    """Single-turn, lower token limit for faster responses."""
    result = chat(
        messages=[{"role": "user", "content": prompt}],
        system=system,
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return result["content"]


def detailed_ask(
    prompt: str,
    system: str = "You are a helpful agricultural AI.",
) -> str:
    """Single-turn, full token limit for detailed responses."""
    result = chat(
        messages=[{"role": "user", "content": prompt}],
        system=system,
        temperature=0.3,
        max_tokens=DETAIL_MAX_TOKENS,
    )
    return result["content"]
