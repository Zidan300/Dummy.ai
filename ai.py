"""Synchronous Phase 1 Ollama client."""

from __future__ import annotations

import logging
import threading

import ollama


logger = logging.getLogger(__name__)
MODEL = "gemma3:4b"


class AIError(RuntimeError):
    """An Ollama request failed."""


def ask_dummy(
    prompt: str,
    on_token=None,
    cancel_event: threading.Event | None = None,
) -> str:
    """Ask Gemma once without response streaming (streaming is Phase 2)."""
    if cancel_event is not None and cancel_event.is_set():
        return ""

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Dummy, a fast local voice assistant. "
                        "Be concise and natural. Usually answer in 1 or 2 short "
                        "sentences. No markdown. No unnecessary explanation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            stream=False,
            options={
                "temperature": 0.2,
                "top_k": 20,
                "top_p": 0.8,
                "num_predict": 80,
            },
        )
    except Exception as exc:
        raise AIError(f"Ollama request failed: {exc}") from exc

    if cancel_event is not None and cancel_event.is_set():
        return ""

    try:
        if hasattr(response, "message"):
            text = response.message.content
        else:
            text = response["message"]["content"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise AIError(f"Ollama returned an invalid response: {exc}") from exc

    answer = str(text).strip()
    if on_token and answer:
        # Keep the old callback API usable, but invoke it once because Phase 1
        # intentionally does not stream Gemma output.
        on_token(answer)
    return answer
