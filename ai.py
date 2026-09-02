"""Ollama streaming and voice-oriented sentence buffering."""

from __future__ import annotations

import logging
import re
import threading

import ollama


logger = logging.getLogger(__name__)
MODEL = "gemma3:4b"
MIN_SENTENCE_CHARS = 8
_SENTENCE_END = re.compile(r"[.!?]+(?:[\"')\]]+)?(?=\s|$)")


class AIError(RuntimeError):
    """An Ollama request failed."""


class SentenceBuffer:
    """Turn arbitrary streamed token chunks into useful speech-sized sentences."""

    def __init__(self, minimum_chars: int = MIN_SENTENCE_CHARS) -> None:
        self._buffer = ""
        self._minimum_chars = minimum_chars

    def add(self, token: str) -> list[str]:
        self._buffer += token
        return self._extract_complete()

    def finish(self) -> list[str]:
        remaining = self._buffer.strip()
        self._buffer = ""
        return [remaining] if remaining else []

    def _extract_complete(self) -> list[str]:
        sentences: list[str] = []
        while True:
            matches = list(_SENTENCE_END.finditer(self._buffer))
            if not matches:
                break

            match = matches[0]
            candidate = self._buffer[:match.end()].strip()
            if not candidate:
                self._buffer = self._buffer[match.end():]
                continue

            # Keep one-word acknowledgements attached to the next sentence so
            # Piper does not receive choppy fragments such as "Yes." or "Okay.".
            if not self._is_sensible(candidate):
                combined_match = next(
                    (later for later in matches[1:] if self._is_sensible(self._buffer[:later.end()].strip())),
                    None,
                )
                if combined_match is None:
                    break
                match = combined_match
                candidate = self._buffer[:match.end()].strip()

            sentences.append(candidate)
            self._buffer = self._buffer[match.end():]
        return sentences

    def _is_sensible(self, sentence: str) -> bool:
        words = re.findall(r"[A-Za-z0-9']+", sentence)
        return len(sentence) >= self._minimum_chars or len(words) >= 2


def _chunk_text(chunk) -> str:
    try:
        if hasattr(chunk, "message"):
            return str(chunk.message.content or "")
        return str(chunk.get("message", {}).get("content", "") or "")
    except (AttributeError, TypeError):
        raise AIError("Ollama returned an invalid stream chunk") from None


def stream_dummy(
    prompt: str,
    on_token=None,
    cancel_event: threading.Event | None = None,
) -> str:
    """Stream Gemma output and forward each token as soon as it arrives."""
    if cancel_event is not None and cancel_event.is_set():
        return ""

    response_stream = None
    full_response = ""
    try:
        response_stream = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Dummy, a fast local voice assistant. "
                        "Answer in 1 to 3 short sentences. No markdown, no lists, "
                        "and no unnecessary explanation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            stream=True,
            options={
                "temperature": 0.2,
                "top_k": 20,
                "top_p": 0.8,
                "num_predict": 80,
            },
        )

        for chunk in response_stream:
            if cancel_event is not None and cancel_event.is_set():
                break
            token = _chunk_text(chunk)
            if not token:
                continue
            full_response += token
            if on_token:
                on_token(token)
    except AIError:
        raise
    except Exception as exc:
        raise AIError(f"Ollama streaming request failed: {exc}") from exc
    finally:
        # The Ollama iterator owns an HTTP response. Closing it on cancellation
        # releases the connection without killing any Python thread.
        close = getattr(response_stream, "close", None)
        if close:
            try:
                close()
            except Exception:
                logger.debug("Could not close Ollama response stream", exc_info=True)

    return full_response.strip()


def ask_dummy(
    prompt: str,
    on_token=None,
    cancel_event: threading.Event | None = None,
) -> str:
    """Compatibility wrapper; Phase 2 generation is streamed."""
    return stream_dummy(prompt, on_token=on_token, cancel_event=cancel_event)
