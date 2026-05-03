from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Protocol

from .types import MemoryItem, MemoryWriteResult

ENTRY_DELIMITER = "\n§\n"

_THREAT_PATTERNS = [
    re.compile(r"ignore\s+(previous|all|above|prior)\s+instructions", re.I),
    re.compile(r"system\s+prompt\s+override", re.I),
    re.compile(r"cat\s+[^\n]*(\.env|credentials|\.netrc|\.npmrc)", re.I),
]
_INVISIBLE_CHARS = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u202e"}


class MemoryProvider(Protocol):
    def snapshot(self) -> dict[str, str]:
        ...

    def search(self, query: str, limit: int = 5) -> list[MemoryItem]:
        ...

    def write(self, item: MemoryItem) -> MemoryWriteResult:
        ...


class MemoryStore:
    def __init__(self, memory_dir: str | Path, memory_char_limit: int = 4000, user_char_limit: int = 2500):
        self.memory_dir = Path(memory_dir)
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        self.memory_entries: list[str] = []
        self.user_entries: list[str] = []
        self._snapshot = {"memory": "", "user": ""}

    def load(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_entries = self._read_entries(self.memory_dir / "MEMORY.md")
        self.user_entries = self._read_entries(self.memory_dir / "USER.md")
        self._snapshot = {
            "memory": self._render("Project/environment memory", self.memory_entries),
            "user": self._render("User preferences", self.user_entries),
        }

    def snapshot(self) -> dict[str, str]:
        return dict(self._snapshot)

    def read_live(self, target: str = "memory") -> list[str]:
        return list(self._entries(target))

    def search(self, query: str, limit: int = 5) -> list[MemoryItem]:
        hits: list[MemoryItem] = []
        for target in ("memory", "user"):
            for entry in self._entries(target):
                if query.lower() in entry.lower():
                    hits.append(MemoryItem(target=target, content=entry))  # type: ignore[arg-type]
                    if len(hits) >= limit:
                        return hits
        return hits

    def write(self, item: MemoryItem) -> MemoryWriteResult:
        message = self.add(item.target, item.content)
        return MemoryWriteResult(target=item.target, action="add", message=message, snapshot_changed=False)

    def add(self, target: str, content: str) -> str:
        content = content.strip()
        if not content:
            raise ValueError("memory content must not be empty")
        self._scan(content)
        entries = self._entries(target)
        if content not in entries:
            entries.append(content)
            self._trim(target)
            self._write_entries(self._path(target), entries)
        return f"Added {target} memory entry. New memories are durable; the current prompt snapshot remains unchanged."

    def replace(self, target: str, match: str, content: str) -> str:
        self._scan(content)
        entries = self._entries(target)
        hits = [idx for idx, value in enumerate(entries) if match in value]
        if len(hits) != 1:
            raise ValueError(f"replace requires exactly one match, found {len(hits)}")
        entries[hits[0]] = content.strip()
        self._trim(target)
        self._write_entries(self._path(target), entries)
        return f"Replaced {target} memory entry. Current prompt snapshot remains unchanged."

    def remove(self, target: str, match: str) -> str:
        entries = self._entries(target)
        hits = [idx for idx, value in enumerate(entries) if match in value]
        if len(hits) != 1:
            raise ValueError(f"remove requires exactly one match, found {len(hits)}")
        removed = entries.pop(hits[0])
        self._write_entries(self._path(target), entries)
        return f"Removed {target} memory entry: {removed[:120]}"

    def _entries(self, target: str) -> list[str]:
        if target == "user":
            return self.user_entries
        if target == "memory":
            return self.memory_entries
        raise ValueError("target must be 'memory' or 'user'")

    def _path(self, target: str) -> Path:
        return self.memory_dir / ("USER.md" if target == "user" else "MEMORY.md")

    def _limit(self, target: str) -> int:
        return self.user_char_limit if target == "user" else self.memory_char_limit

    def _trim(self, target: str) -> None:
        entries = self._entries(target)
        limit = self._limit(target)
        while len(ENTRY_DELIMITER.join(entries)) > limit and entries:
            entries.pop(0)

    @staticmethod
    def _read_entries(path: Path) -> list[str]:
        if not path.exists():
            return []
        return [item.strip() for item in path.read_text(encoding="utf-8").split(ENTRY_DELIMITER) if item.strip()]

    @staticmethod
    def _write_entries(path: Path, entries: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name, dir=str(path.parent), text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(ENTRY_DELIMITER.join(entries))
                if entries:
                    fh.write("\n")
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    @staticmethod
    def _render(title: str, entries: list[str]) -> str:
        if not entries:
            return ""
        rendered = "\n".join(f"- {entry}" for entry in entries)
        return f"## {title}\n{rendered}"

    @staticmethod
    def _scan(content: str) -> None:
        for char in _INVISIBLE_CHARS:
            if char in content:
                raise ValueError(f"blocked memory content: invisible character U+{ord(char):04X}")
        for pattern in _THREAT_PATTERNS:
            if pattern.search(content):
                raise ValueError("blocked memory content: prompt-injection or exfiltration pattern")
