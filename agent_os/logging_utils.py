from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .storage import utcnow_iso

_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[^'\"\s]+"),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
]


def redact(value: Any) -> Any:
    if isinstance(value, str):
        text = value
        for pattern in _SECRET_PATTERNS:
            text = pattern.sub(lambda m: m.group(0).split("=", 1)[0] + "=<redacted>" if "=" in m.group(0) else "<redacted>", text)
        return text
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


class JSONLLogger:
    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def write(self, *, session_id: str, run_id: str, event_type: str, message: str,
              payload: dict[str, Any] | None = None) -> None:
        path = self.log_dir / f"{session_id}.jsonl"
        record = {
            "created_at": utcnow_iso(),
            "session_id": session_id,
            "run_id": run_id,
            "event_type": event_type,
            "message": message,
            "payload": redact(payload or {}),
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
