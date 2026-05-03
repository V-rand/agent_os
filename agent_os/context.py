from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import AgentOSConfig
from .compression import CompressionEngine, DeterministicCompressionEngine, SUMMARY_PREFIX
from .memory import MemoryStore
from .skills import SkillManager
from .storage import SQLiteStore
from .cache_stability import prompt_fingerprint


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_tokens(json.dumps(msg, ensure_ascii=False, default=str)) for msg in messages)


@dataclass(slots=True)
class CompiledContext:
    system_prompt: str
    messages: list[dict[str, Any]]
    estimated_tokens: int
    context_budget_ratio: float
    compressed: bool = False
    compression_metadata: dict[str, Any] | None = None


class PromptBuilder:
    def __init__(self, config: AgentOSConfig, memory: MemoryStore, skills: SkillManager):
        self.config = config
        self.memory = memory
        self.skills = skills
        self._snapshot_prompt: str | None = None
        self._snapshot_fingerprint: str | None = None

    def build(self) -> str:
        if self._snapshot_prompt is not None:
            return self._snapshot_prompt
        snapshot = self.memory.snapshot()
        parts = [
            self.config.identity,
            "You run a tool-using ReAct loop. Use tools when they materially improve correctness.",
            "Report errors directly; do not hide tool or API failures.",
            "Keep outputs business-agnostic. Upper-layer workflows own domain-specific policy.",
            f"Workspace root: {self.config.workspace_root}",
        ]
        if snapshot.get("memory"):
            parts.append(snapshot["memory"])
        if snapshot.get("user"):
            parts.append(snapshot["user"])
        skill_index = self.skills.index_prompt(self.config.enabled_toolsets - self.config.disabled_toolsets)
        if skill_index:
            parts.append(skill_index)
        self._snapshot_prompt = "\n\n".join(part for part in parts if part)
        self._snapshot_fingerprint = prompt_fingerprint(self._snapshot_prompt)
        return self._snapshot_prompt

    def fingerprint(self) -> str:
        if self._snapshot_fingerprint is None:
            self.build()
        return self._snapshot_fingerprint or ""


class ContextManager:
    def __init__(self, config: AgentOSConfig, store: SQLiteStore, prompt_builder: PromptBuilder,
                 compression_engine: CompressionEngine | None = None):
        self.config = config
        self.store = store
        self.prompt_builder = prompt_builder
        self.compression_engine = compression_engine or DeterministicCompressionEngine()

    def compile(
        self,
        session_id: str,
        live_messages: list[dict[str, Any]] | None = None,
        *,
        auto_compress: bool = True,
    ) -> CompiledContext:
        system_prompt = self.prompt_builder.build()
        messages = [dict(msg) for msg in (live_messages if live_messages is not None else self.load_history(session_id))]
        summary = self.store.latest_summary(session_id)
        if summary and not _already_has_summary(messages):
            messages.insert(0, {"role": "system", "content": f"{SUMMARY_PREFIX}\n\n{summary['content']}"})
        workspace_snapshot = self._workspace_snapshot_message(session_id)
        if workspace_snapshot:
            messages.insert(0, workspace_snapshot)
        compiled = [{"role": "system", "content": system_prompt}, *messages]
        tokens = estimate_messages_tokens(compiled)
        compressed = False
        compression_metadata: dict[str, Any] | None = None
        if auto_compress and self.config.compression_enabled and tokens >= int(self.config.context_budget_tokens * self.config.compression_trigger_ratio):
            before_tokens = tokens
            messages = self.compress(session_id, messages)
            compiled = [{"role": "system", "content": system_prompt}, *messages]
            tokens = estimate_messages_tokens(compiled)
            compressed = tokens < before_tokens
            latest = self.store.latest_summary(session_id)
            compression_metadata = latest.get("metadata") if latest else None
        return CompiledContext(
            system_prompt=system_prompt,
            messages=messages,
            estimated_tokens=tokens,
            context_budget_ratio=tokens / max(1, self.config.context_budget_tokens),
            compressed=compressed,
            compression_metadata=compression_metadata,
        )

    def compress(self, session_id: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        clean_messages = _without_existing_summary(messages)
        protect_head = max(0, self.config.protect_first_messages)
        protect_tail = max(1, self.config.protect_last_messages)
        if len(clean_messages) <= protect_head + protect_tail:
            return messages
        if not self._compression_allowed(session_id, clean_messages):
            return clean_messages
        head = clean_messages[:protect_head]
        remaining = clean_messages[protect_head:]
        middle, tail = _split_middle_tail(
            remaining,
            min_tail_messages=protect_tail,
            tail_token_budget=max(0, self.config.protect_last_tokens),
        )
        if not middle:
            return clean_messages
        before_tokens = estimate_messages_tokens(clean_messages)
        todos = self.store.list_todos(session_id)
        existing = self.store.latest_summary(session_id)
        result = self.compression_engine.compress(
            middle,
            todos,
            existing_summary=existing["content"] if existing else None,
        )
        summary_message = {"role": "system", "content": f"{SUMMARY_PREFIX}\n\n{result.summary}"}
        compressed_messages = [*head, summary_message, *tail]
        after_tokens = estimate_messages_tokens(compressed_messages)
        savings_ratio = 1 - (after_tokens / max(1, before_tokens))
        if savings_ratio < self.config.compression_min_savings_ratio:
            return clean_messages
        metadata = {
            "before_estimated_tokens": before_tokens,
            "after_estimated_tokens": after_tokens,
            "savings_ratio": savings_ratio,
            "protected_head_messages": len(head),
            "protected_tail_messages": len(tail),
            "compressed_middle_messages": len(middle),
            **(result.metadata or {}),
        }
        self.store.save_summary(
            session_id,
            result.summary,
            coverage_end_message_id=result.coverage_end_message_id,
            metadata=metadata,
        )
        return compressed_messages

    def _compression_allowed(self, session_id: str, messages: list[dict[str, Any]]) -> bool:
        latest = self.store.latest_summary(session_id)
        if not latest:
            return True
        coverage = latest.get("coverage_end_message_id")
        if not isinstance(coverage, int):
            return True
        new_messages = [msg for msg in messages if isinstance(msg.get("id"), int) and msg["id"] > coverage]
        return len(new_messages) >= max(0, self.config.compression_min_interval_messages)

    def load_history(self, session_id: str) -> list[dict[str, Any]]:
        """Restore persisted conversation into memory at session start.

        Hermes keeps the active context in process memory and uses durable
        storage as a ledger/recovery source. This method is the recovery
        boundary: runtime calls it once, then mutates its live list.
        """
        rows = self.store.list_messages(session_id)
        messages: list[dict[str, Any]] = []
        for row in rows:
            msg: dict[str, Any] = {"id": row["id"], "role": row["role"]}
            if row.get("content") is not None:
                msg["content"] = row["content"]
            if row.get("tool_call_id"):
                msg["tool_call_id"] = row["tool_call_id"]
            if row.get("tool_calls"):
                msg["tool_calls"] = row["tool_calls"]
            messages.append(msg)
        return messages

    def _workspace_snapshot_message(self, session_id: str) -> dict[str, str] | None:
        session = self.store.get_session(session_id)
        if not session:
            return None
        artifacts = self.store.list_artifacts(session_id)
        materials = self.store.list_materials(session_id)
        todos = self.store.list_todos(session_id)
        lines = [
            "[WORKSPACE SNAPSHOT - REFERENCE ONLY]",
            f"Session stage: {session.get('stage') or '(none)'}",
        ]
        if artifacts:
            lines.append("Artifacts:")
            for item in artifacts[:40]:
                source = ", ".join(item.get("source_paths") or [])
                suffix = f" sources={source}" if source else ""
                lines.append(f"- {item['path']} kind={item.get('kind', '')} stage={item.get('stage', '')}{suffix}")
        if materials:
            lines.append("Materials:")
            for item in materials[:40]:
                title = f" title={item.get('title')}" if item.get("title") else ""
                lines.append(f"- {item['path']} kind={item.get('kind', '')} stage={item.get('stage', '')}{title}")
        pending = [todo for todo in todos if todo.get("status") not in {"completed", "cancelled"}]
        if pending:
            lines.append("Open todos:")
            for todo in pending[:20]:
                metadata = todo.get("metadata") or {}
                stage = f" stage={metadata.get('stage')}" if metadata.get("stage") else ""
                lines.append(f"- [{todo.get('status')}] {todo.get('text')}{stage}")
        if len(lines) == 2:
            return None
        return {"role": "system", "content": "\n".join(lines)}


def _already_has_summary(messages: list[dict[str, Any]]) -> bool:
    return bool(messages and messages[0].get("role") == "system" and str(messages[0].get("content", "")).startswith(SUMMARY_PREFIX))


def _without_existing_summary(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [msg for msg in messages if not (msg.get("role") == "system" and str(msg.get("content", "")).startswith(SUMMARY_PREFIX))]


def _split_middle_tail(
    messages: list[dict[str, Any]],
    *,
    min_tail_messages: int,
    tail_token_budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not messages:
        return [], []
    tail_start = max(0, len(messages) - max(1, min_tail_messages))
    if tail_token_budget > 0:
        running = 0
        token_tail_start = len(messages)
        for idx in range(len(messages) - 1, -1, -1):
            item_tokens = estimate_messages_tokens([messages[idx]])
            if token_tail_start < len(messages) and running + item_tokens > tail_token_budget:
                break
            running += item_tokens
            token_tail_start = idx
        tail_start = min(tail_start, token_tail_start)
    return messages[:tail_start], messages[tail_start:]
