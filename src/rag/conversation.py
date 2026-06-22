"""
src/rag/conversation.py
Sliding-window conversation memory manager for multi-turn RAG chat.

Stores (user, assistant) turn pairs, trims to a configurable window,
and serialises to/from JSON for session persistence.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Turn:
    user:      str
    assistant: str
    metadata:  dict = field(default_factory=dict)   # latency, sources, etc.

    def to_dict(self) -> dict:
        return asdict(self)


class ConversationMemory:
    """
    Sliding-window conversation memory.

    Usage::

        mem = ConversationMemory(max_turns=6)
        mem.add("What is anemia?", "Anemia means low hemoglobin.")
        mem.add("How is it treated?", "With iron supplements.")

        print(mem.format())          # formatted string for prompt injection
        print(mem.as_messages())     # list of {role, content} dicts
        mem.save("session.json")     # persist
        mem.load("session.json")     # restore
    """

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self._turns: List[Turn] = []

    # ── mutators ─────────────────────────────

    def add(
        self,
        user_message: str,
        assistant_message: str,
        metadata: Optional[dict] = None,
    ) -> None:
        self._turns.append(Turn(
            user=user_message,
            assistant=assistant_message,
            metadata=metadata or {},
        ))
        # Trim to window
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns:]

    def clear(self) -> None:
        self._turns.clear()

    # ── accessors ─────────────────────────────

    @property
    def turns(self) -> List[Turn]:
        return list(self._turns)

    @property
    def is_empty(self) -> bool:
        return len(self._turns) == 0

    def last_k(self, k: int) -> List[Turn]:
        return self._turns[-k:] if k < len(self._turns) else list(self._turns)

    # ── formatting ────────────────────────────

    def format(self, k: Optional[int] = None) -> str:
        """Return conversation as a plain-text string for prompt injection."""
        turns = self.last_k(k) if k else self._turns
        if not turns:
            return "(No prior conversation)"
        lines = []
        for t in turns:
            lines.append(f"User: {t.user}")
            lines.append(f"Assistant: {t.assistant}")
        return "\n".join(lines)

    def as_messages(self, k: Optional[int] = None) -> List[dict]:
        """Return as OpenAI-style message list [{role, content}, ...]."""
        turns = self.last_k(k) if k else self._turns
        messages = []
        for t in turns:
            messages.append({"role": "user",      "content": t.user})
            messages.append({"role": "assistant",  "content": t.assistant})
        return messages

    # ── persistence ───────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([t.to_dict() for t in self._turns], f, indent=2)

    def load(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            return
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        self._turns = [Turn(**d) for d in data]
        # Apply window limit after load
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns:]

    # ── dunder ────────────────────────────────

    def __len__(self) -> int:
        return len(self._turns)

    def __repr__(self) -> str:
        return f"ConversationMemory(turns={len(self._turns)}, max={self.max_turns})"
