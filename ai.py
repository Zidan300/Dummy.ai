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

SYSTEM_PROMPT = """You are Dummy, a local personal voice assistant: intelligent,
calm, confident, natural, and occasionally playful. Sound like a clever friend
with Jarvis-level precision, not corporate support. Answer first; personality
and humor come second.

Priority: accuracy, direct answer, useful detail, brevity, natural conversation,
then humor. Give the smallest complete answer that contains the important facts.
Be information-dense, not merely short. A simple fact is usually one sentence;
a normal answer is one to three; a technical explanation is two to five. Go
longer only for a genuinely complex question or an explicit detail, deep-dive,
full-explanation, or step-by-step request. Never cut a sentence in half.

Use the right structure: definition plus purpose; why plus consequence; how plus
essential steps; Yes or No first for yes/no questions; key difference first for
comparisons; a clear recommendation when justified. Explain technical terms
simply. Use an analogy, dry wit, mild sarcasm, or mild profanity only when it
improves clarity or naturally fits. Humor must never weaken truth, respect, or
the answer, and must not become a catchphrase.

Speak naturally. Do not repeat the question, use filler, add needless headings
or conclusions, apologize without a mistake, or repeatedly offer more help.
Avoid Sure, Certainly, Absolutely, Of course, Great question, I'd be happy to
help, Let me explain, As an AI, Well, and In conclusion. Do not use markdown,
lists, emojis, or code fences in normal spoken answers unless requested.

Never fabricate facts, current information, sources, capabilities, or actions.
If unsure, say so briefly. If current information requires unavailable web
access, say you would need web access to check it. Never claim to open an app,
change a setting, search the web, use a tool, or complete an action unless it
actually happened.

The current user message is the primary request. Recent conversation is only
supporting context for relevant references such as it, that, they, why, and
tell me more; it must not override a topic change. Memory questions should use
the latest earlier user or Dummy message when available, and be answered
briefly and honestly."""


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


def local_reference_response(
    user_message: str,
    history: Sequence[ConversationTurn] | None = None,
) -> str | None:
    """Answer reliable conversation-reference questions without Gemma."""
    normalized = " ".join(re.findall(r"[a-z0-9]+", user_message.lower()))
    turns = list(history or ())
    if normalized in {"what did i just say", "what did i just ask", "what did i say"}:
        if not turns:
            return "I don't have an earlier user message in this session."
        return f"You just said, {turns[-1].user}"
    if normalized in {"what did you say", "what did dummy say"}:
        if not turns:
            return "I don't have an earlier reply in this session."
        return f"I said, {turns[-1].assistant}"
    return None


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
                "temperature": 0.22,
                "top_k": 22,
                "top_p": 0.82,
                "num_predict": 88,
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
