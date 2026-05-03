from __future__ import annotations

import hashlib
import json
from typing import Any

from .types import ToolResult, json_safe


def stable_json(value: Any) -> str:
    """Render model-visible structured content deterministically."""
    return json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def prompt_fingerprint(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def tool_schema_fingerprint(tools: list[dict[str, Any]]) -> str:
    return hashlib.sha256(stable_json(tools).encode("utf-8")).hexdigest()


def model_tool_result_payload(tool_name: str, result: ToolResult) -> dict[str, Any]:
    """Return a stable model-visible tool result.

    Runtime-only details such as latency, wall-clock timestamps, run ids, and
    storage ids should stay in events/SQLite. The model only needs a stable
    result envelope so repeated equivalent calls produce equivalent context.
    """
    metadata = dict(result.metadata or {})
    if "truncated" not in metadata and result.content.endswith("\n...[truncated]"):
        metadata["truncated"] = True
    payload: dict[str, Any] = {
        "ok": result.success,
        "tool": tool_name,
        "status": "ok" if result.success else "error",
        "summary": result.summary,
        "content": result.content,
        "artifacts": result.artifacts,
        "metadata": metadata,
        "source": {
            "tool": tool_name,
            "origin": "current_session",
        },
        "error": result.error,
    }
    if result.data is not None:
        payload["data"] = result.data
    return payload


def model_tool_result_content(tool_name: str, result: ToolResult) -> str:
    return stable_json(model_tool_result_payload(tool_name, result))
