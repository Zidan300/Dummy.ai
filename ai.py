"""Ollama streaming and voice-oriented sentence buffering."""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Sequence

import ollama

from context import ConversationTurn


logger = logging.getLogger(__name__)
MODEL = "gemma3:4b"
MIN_SENTENCE_CHARS = 8
_SENTENCE_END = re.compile(r"[.!?]+(?:[\"')\]]+)?(?=\s|$)")

SYSTEM_PROMPT = """You are Dummy, a personal local voice assistant.

Be intelligent, calm, confident, natural, factual, and concise. Be lightly
witty only when it fits. Answer the user's actual question immediately and put
the most useful information first. Think first, then use the minimum language
needed for a complete and useful answer. Do not become vague just to be short.
Never invent facts. If you are uncertain, say so briefly.

Response guidance:
- Simple factual questions usually get one informative sentence.
- Normal questions usually get one to three sentences.
- Technical explanations may use two to five sentences when needed.
- Requests for detail, a deep dive, a full explanation, or step-by-step help
  may be longer, but stay focused.
- For definitions, give the definition and why it matters.
- For why questions, give the reason and consequence.
- For how questions, give the essential steps unless a full tutorial is asked.
- For comparisons, state the key difference first and recommend when useful.
- For yes/no questions, start with Yes or No and briefly explain.
- For technical topics, use accurate terms and explain them plainly.

Speak in plain natural language. Do not repeat the user's question. Do not
use markdown, bullet points, headings, emojis, or code fences in normal spoken
answers. Avoid filler such as Sure, Absolutely, Of course, I'd be happy to,
That's a great question, Well, In conclusion, and As an AI. Do not apologize
unless you made an actual mistake. Do not add a repeated conclusion or end
with a routine offer to help. Never claim to have opened an app, changed a
setting, searched the web, used a tool, or completed an action unless a real
tool did it.

Conversation history, when present, is only for resolving references such as
it, that, they, the first one, and follow-up questions. It comes before the
current user message and must not override that message. For "what did I just
say?", use the most recent earlier user message in the history, not the
question itself. For "what did you say?", use the most recent earlier Dummy
reply. If the needed context is unavailable, say so honestly."""


class AIError(RuntimeError):
    """An Ollama request failed."""


def build_messages(
    user_message: str,
    history: Sequence[ConversationTurn] | None = None,
) -> list[dict[str, str]]:
    """Build a compact Ollama chat payload from bounded session history."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or ():
        messages.append({"role": "user", "content": turn.user})
        messages.append({"role": "assistant", "content": turn.assistant})
    messages.append({"role": "user", "content": user_message})
    return messages


def clean_for_speech(text: str) -> str:
    """Remove common formatting artifacts while preserving natural wording."""
    cleaned = text.replace("\r", "\n")
    cleaned = re.sub(r"```(?:[A-Za-z0-9_+-]+)?", " ", cleaned)
    cleaned = re.sub(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", "", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "").replace("`", "")
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", cleaned)
    cleaned = re.sub(r"(^|\s)[#>*-]+(?=\s)", r"\1", cleaned)
    cleaned = cleaned.translate(str.maketrans({
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": "-",
        "–": "-",
        "•": ",",
        "→": " ",
    }))
    cleaned = re.sub(r"[^\w\s.,!?;:'\"()/%+$&-]", " ", cleaned, flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned).strip()


def unsupported_command_response(command: str) -> str:
    """Return a factual response until Phase 6 tools exist."""
    normalized = re.sub(r"[.!?,;:]+$", "", command.strip())
    words = normalized.split()
    if words and words[0].lower() == "please":
        words = words[1:]
    if len(words) >= 2 and words[0].lower() in {"open", "launch"}:
        target = " ".join(words[1:]).strip()
        if target.lower().startswith("the "):
            target = target[4:]
        return f"{target.capitalize()} control isn't available yet."
    return "I can't execute that command yet."


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
    history: Sequence[ConversationTurn] | None = None,
) -> str:
    """Stream Gemma output and forward each token as soon as it arrives."""
    if cancel_event is not None and cancel_event.is_set():
        return ""

    response_stream = None
    full_response = ""
    try:
        response_stream = ollama.chat(
            model=MODEL,
            messages=build_messages(prompt, history),
            stream=True,
            options={
                "temperature": 0.18,
                "top_k": 20,
                "top_p": 0.8,
                "num_predict": 128,
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
    history: Sequence[ConversationTurn] | None = None,
) -> str:
    """Compatibility wrapper; Phase 2 generation is streamed."""
    return stream_dummy(
        prompt,
        on_token=on_token,
        cancel_event=cancel_event,
        history=history,
    )
