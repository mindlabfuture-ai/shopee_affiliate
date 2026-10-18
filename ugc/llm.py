"""Shared Claude client and structured-output call."""
from __future__ import annotations

import os
from typing import TypeVar

import anthropic
from pydantic import BaseModel

from . import config

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


def get_client() -> anthropic.Anthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise LLMError("ANTHROPIC_API_KEY is not set. Export it in the shell that starts the app.")
    return anthropic.Anthropic()


def image_block(b64: str) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}


def parse(client, schema: type[T], system: str, content: list[dict], max_tokens: int = 8000) -> T:
    resp = client.messages.parse(
        model=config.CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
        output_format=schema,
    )
    if resp.stop_reason == "refusal":
        raise LLMError(f"Claude declined the request: {getattr(resp, 'stop_details', None)}")
    if resp.parsed_output is None:
        raise LLMError(f"no structured output (stop_reason={resp.stop_reason})")
    return resp.parsed_output
