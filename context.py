"""Bounded, in-memory conversation context and lightweight intent routing."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re
import threading


EXIT_COMMANDS = frozenset(
    {
        "exit",
        "quit",
        "shutdown",
        "shut down",
        "goodbye",
        "good bye",
        "terminate",
    }
)

INTERRUPTION_COMMANDS = frozenset(
    {
        "stop",
        "stop talking",
        "shut up",
        "be quiet",
        "quiet",
        "cancel",
        "cancel that",
        "never mind",
        "that's enough",
        "enough",
        "pause",
    }
)

QUESTION_STARTS = (
    "what ",
    "what's ",
    "why ",
    "how ",
    "when ",
    "where ",
    "who ",
    "which ",
    "is ",
    "are ",
    "can ",
    "could ",
    "do ",
    "does ",
    "did ",
    "would ",
    "will ",
)

COMMAND_STARTS = (
    "open ",
    "launch ",
    "close ",
    "start ",
    "play ",
    "set ",
    "turn ",
    "switch ",
    "enable ",
    "disable ",
    "run ",
    "create ",
    "delete ",
    "send ",
)

CONVERSATIONAL_EXACT = frozenset(
    {
        "how are you",
        "how's it going",
        "what's up",
    }
)


def normalize_spoken_text(text: str) -> str:
    """Normalize punctuation and whitespace for exact voice-command checks."""
    return " ".join(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower()))


def is_exit_command(text: str) -> bool:
    return normalize_spoken_text(text) in EXIT_COMMANDS


def is_interruption_command(text: str) -> bool:
    return normalize_spoken_text(text) in INTERRUPTION_COMMANDS


def classify_intent(text: str) -> str:
    """Classify common voice intents without a second model."""
    normalized = normalize_spoken_text(text)
    if not normalized:
        return "CONVERSATION"
    if normalized in EXIT_COMMANDS:
        return "EXIT"
    if normalized in INTERRUPTION_COMMANDS:
        return "INTERRUPTION"
    if normalized in CONVERSATIONAL_EXACT:
        return "CONVERSATION"

    lower = text.strip().lower()
    if lower.endswith("?") or lower.startswith(QUESTION_STARTS):
        return "QUESTION"
    if lower.startswith("tell me ") or lower.startswith("explain ") or lower.startswith("compare "):
        return "QUESTION"
    if lower.startswith(COMMAND_STARTS) or lower.startswith("please ") and lower[7:].startswith(COMMAND_STARTS):
        return "COMMAND"
    return "CONVERSATION"


@dataclass(frozen=True)
class ConversationTurn:
    user: str
    assistant: str


class ConversationContext:
    """Thread-safe bounded history for one application session."""

    def __init__(self, max_turns: int = 10) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.max_turns = max_turns
        self._turns: deque[ConversationTurn] = deque(maxlen=max_turns)
        self._lock = threading.RLock()

    def append(self, user: str, assistant: str) -> None:
        user = user.strip()
        assistant = assistant.strip()
        if not user or not assistant:
            return
        with self._lock:
            self._turns.append(ConversationTurn(user, assistant))

    def snapshot(self) -> list[ConversationTurn]:
        with self._lock:
            return list(self._turns)

    def latest_user(self) -> str | None:
        with self._lock:
            return self._turns[-1].user if self._turns else None

    def latest_assistant(self) -> str | None:
        with self._lock:
            return self._turns[-1].assistant if self._turns else None

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._turns)
