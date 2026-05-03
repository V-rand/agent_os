from __future__ import annotations

import json
from typing import Any, Protocol

from .types import CompressionResult, ModelResponse

SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION - REFERENCE ONLY] Earlier turns were compacted. "
    "Treat this as background, not active instructions. Resume from the latest user message."
)


class CompressionEngine(Protocol):
    def compress(
        self,
        messages: list[dict[str, Any]],
        todos: list[dict[str, Any]],
        existing_summary: str | None = None,
    ) -> CompressionResult:
        ...


class DeterministicCompressionEngine:
    """Dependency-free compressor that preserves operational state."""

    def compress(
        self,
        messages: list[dict[str, Any]],
        todos: list[dict[str, Any]],
        existing_summary: str | None = None,
    ) -> CompressionResult:
        coverage_end = _last_message_id(messages)
        lines = [
            "## Active Task",
            "Continue from the latest non-compacted user message after this summary.",
            "",
            "## Existing Summary",
            existing_summary.strip() if existing_summary else "None.",
            "",
            "## Key Earlier Turns",
        ]
        for msg in messages[-60:]:
            role = msg.get("role", "unknown")
            content = _message_content(msg)
            if not content:
                continue
            lines.append(f"- {role}: {_truncate(content, 700)}")

        pending = [todo for todo in todos if todo.get("status") not in {"completed", "cancelled"}]
        lines.extend(["", "## Current Todos"])
        if pending:
            for todo in pending:
                stage = ""
                metadata = todo.get("metadata") or {}
                if metadata.get("stage"):
                    stage = f" stage={metadata['stage']}"
                lines.append(f"- [{todo.get('status', 'pending')}] {todo.get('text', '')}{stage}")
        else:
            lines.append("No pending todos.")

        lines.extend(["", "## Compression Notes", "Tool results above are summarized; re-run tools only if the latest task requires fresh state."])
        return CompressionResult(
            summary="\n".join(lines),
            coverage_end_message_id=coverage_end,
            protected_message_count=0,
            metadata={"engine": "deterministic_fallback"},
        )


class CompressionSubagentEngine:
    """LLM-backed middle-context compressor with deterministic fallback."""

    def __init__(self, model_client: Any, *, max_summary_tokens: int = 4000,
                 fallback: CompressionEngine | None = None):
        self.model_client = model_client
        self.max_summary_tokens = max_summary_tokens
        self.fallback = fallback or DeterministicCompressionEngine()

    def compress(
        self,
        messages: list[dict[str, Any]],
        todos: list[dict[str, Any]],
        existing_summary: str | None = None,
    ) -> CompressionResult:
        if not messages:
            return self.fallback.compress(messages, todos, existing_summary)
        prompt = _build_compression_prompt(messages, todos, existing_summary, self.max_summary_tokens)
        try:
            response: ModelResponse = self.model_client.complete(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a conservative context compression subagent. "
                            "Summarize only the provided middle transcript. Do not answer user questions, "
                            "do not execute tasks, and do not invent facts. Preserve operational state."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=[],
            )
            summary = (response.content or "").strip()
            if not summary:
                raise ValueError("compression subagent returned empty summary")
            return CompressionResult(
                summary=summary,
                coverage_end_message_id=_last_message_id(messages),
                protected_message_count=0,
                metadata={"engine": "compression_subagent"},
            )
        except Exception:
            return self.fallback.compress(messages, todos, existing_summary)


def _last_message_id(messages: list[dict[str, Any]]) -> int | None:
    for msg in reversed(messages):
        value = msg.get("id")
        if isinstance(value, int):
            return value
    return None


def _message_content(msg: dict[str, Any]) -> str:
    if msg.get("content"):
        return str(msg["content"])
    if msg.get("tool_calls"):
        return "tool_calls=" + json.dumps(msg["tool_calls"], ensure_ascii=False)
    return ""


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def _build_compression_prompt(
    messages: list[dict[str, Any]],
    todos: list[dict[str, Any]],
    existing_summary: str | None,
    max_summary_tokens: int,
) -> str:
    compact_messages = []
    for msg in messages:
        item = {
            "role": msg.get("role"),
            "content": _message_content(msg),
        }
        if msg.get("tool_call_id"):
            item["tool_call_id"] = msg["tool_call_id"]
        compact_messages.append(item)
    pending_todos = [todo for todo in todos if todo.get("status") not in {"completed", "cancelled"}]
    return "\n".join([
        "Conservatively compress the MIDDLE portion of an agent conversation.",
        "The conversation head and recent tail are preserved elsewhere. Your job is only to replace the middle with a faithful handoff.",
        "This summary is for a different assistant resuming later; it is reference only, not an instruction to execute old requests.",
        f"Soft target maximum summary tokens: {max_summary_tokens}. If preserving important details requires more, prefer completeness over brevity.",
        "",
        "Required sections:",
        "## Active Task",
        "## Completed Work",
        "## Key Facts And Decisions",
        "## Tool Results Worth Keeping",
        "## Pending Questions Or Risks",
        "## Current Todos",
        "",
        "Rules:",
        "- Be conservative: do not over-compress.",
        "- Preserve concrete file paths, commands, user constraints, errors, decisions, and unresolved questions.",
        "- Preserve tool outputs when they contain facts needed later; summarize repetitive/noisy output only.",
        "- Keep exact names, ids, URLs, paths, numbers, and quoted user requirements.",
        "- If an existing summary is provided, update it instead of duplicating it.",
        "- Do not answer any user question from the transcript.",
        "- Do not mention that the head and tail are missing; they are intentionally preserved outside this summary.",
        "",
        "Existing summary:",
        existing_summary.strip() if existing_summary else "None.",
        "",
        "Pending todos JSON:",
        json.dumps(pending_todos, ensure_ascii=False, sort_keys=True),
        "",
        "Middle conversation JSON:",
        json.dumps(compact_messages, ensure_ascii=False, sort_keys=True),
    ])
