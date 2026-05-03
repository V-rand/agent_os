from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from .storage import utcnow_iso

MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ChatMessage:
    role: MessageRole
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            data["content"] = self.content
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None:
            data["tool_calls"] = self.tool_calls
        return data


@dataclass(slots=True)
class ToolResult:
    success: bool
    content: str
    data: Any = None
    error: str | None = None
    latency_seconds: float = 0.0
    summary: str | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, content: str, data: Any = None, latency_seconds: float = 0.0) -> "ToolResult":
        return cls(success=True, content=content, data=data, latency_seconds=latency_seconds)

    @classmethod
    def fail(cls, error: str, data: Any = None, latency_seconds: float = 0.0) -> "ToolResult":
        return cls(success=False, content=error, data=data, error=error, latency_seconds=latency_seconds)

    def to_payload(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "content": self.content,
            "data": self.data,
            "error": self.error,
            "latency_seconds": self.latency_seconds,
            "summary": self.summary,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_prompt_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_hit_rate_known: bool = False

    @property
    def cache_hit_rate(self) -> float | None:
        if not self.cache_hit_rate_known or self.prompt_tokens <= 0:
            return None
        return self.cached_prompt_tokens / self.prompt_tokens


@dataclass(slots=True)
class MemoryItem:
    target: Literal["memory", "user"]
    content: str


@dataclass(slots=True)
class MemoryWriteResult:
    target: Literal["memory", "user"]
    action: str
    message: str
    snapshot_changed: bool = False


@dataclass(slots=True)
class CompressionResult:
    summary: str
    coverage_end_message_id: int | None
    protected_message_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelResponse:
    content: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    raw: Any = None


@dataclass(slots=True)
class RunEvent:
    type: str
    message: str
    session_id: str
    run_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "message": self.message,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "payload": json_safe(self.payload),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunEvent":
        return cls(
            type=str(data["type"]),
            message=str(data["message"]),
            session_id=str(data["session_id"]),
            run_id=str(data["run_id"]),
            payload=dict(data.get("payload") or {}),
            created_at=str(data.get("created_at") or utcnow_iso()),
        )


@dataclass(slots=True)
class RunResult:
    session_id: str
    run_id: str
    content: str
    messages: list[dict[str, Any]]
    usage: Usage
    iterations: int


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [json_safe(v) for v in value]
        return str(value)
