from __future__ import annotations

import json
import copy
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from typing import Any, Iterator

from .config import AgentOSConfig
from .compression import CompressionSubagentEngine
from .context import ContextManager, PromptBuilder, estimate_messages_tokens
from .cache_stability import model_tool_result_content, tool_schema_fingerprint
from .logging_utils import JSONLLogger
from .memory import MemoryStore
from .model_client import ModelClient, ModelClientError, OpenAIChatClient
from .skills import SkillManager
from .storage import SQLiteStore
from .tools.context import ToolContext
from .tools.registry import discover_builtin_tools, registry
from .types import ModelResponse, RunEvent, RunResult, Usage, json_safe


class AgentRuntime:
    def __init__(self, config: AgentOSConfig | None = None, *, model_client: ModelClient | None = None, depth: int = 0):
        self.config = config or AgentOSConfig.load()
        if self.config.web_tools_enabled:
            os.environ.setdefault("AGENT_OS_ENABLE_WEB_TOOLS", "1")
        self.depth = depth
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self.store = SQLiteStore(self.config.db_path)
        self.memory = MemoryStore(self.config.memory_dir)
        self.memory.load()
        self.skills = SkillManager(self.config.skills_dir)
        discover_builtin_tools()
        self.prompt_builder = PromptBuilder(self.config, self.memory, self.skills)
        self.model_client = model_client or OpenAIChatClient(self.config)
        compression_client = self.model_client
        if model_client is None and self.config.compression_model:
            compression_client = OpenAIChatClient(replace(self.config, model=self.config.compression_model))
        self.context_manager = ContextManager(
            self.config,
            self.store,
            self.prompt_builder,
            compression_engine=CompressionSubagentEngine(
                compression_client,
                max_summary_tokens=self.config.compression_max_summary_tokens,
            ),
        )
        self.logger = JSONLLogger(self.config.log_dir)

    def run(self, message: str, session_id: str | None = None) -> RunResult:
        events = list(self.stream(message, session_id=session_id))
        final = next((event for event in reversed(events) if event.type == "run.completed"), None)
        if final is None:
            error = next((event for event in reversed(events) if event.type == "run.failed"), None)
            raise RuntimeError(error.message if error else "run did not complete")
        payload = final.payload
        return RunResult(
            session_id=final.session_id,
            run_id=final.run_id,
            content=str(payload.get("content") or ""),
            messages=payload.get("messages") or [],
            usage=Usage(**payload.get("usage", {})),
            iterations=int(payload.get("iterations") or 0),
        )

    def stream(self, message: str, session_id: str | None = None) -> Iterator[RunEvent]:
        sid = self.store.ensure_session(
            session_id,
            title=message[:80] or "New session",
            workspace_root=str(self.config.workspace_root),
            metadata={"model": self.config.model, "base_url": self.config.base_url},
        )
        run_id = str(uuid.uuid4())
        yield from self._emit(sid, run_id, "run.started", "Run started", {"message": message})
        live_messages = self.context_manager.load_history(sid)
        user_message_id = self.store.add_message(sid, "user", message)
        live_messages.append({"id": user_message_id, "role": "user", "content": message})

        effective_toolsets = self.config.enabled_toolsets - self.config.disabled_toolsets
        tools = registry.schemas(effective_toolsets, self.config.disabled_toolsets)
        tools_fingerprint = tool_schema_fingerprint(tools)
        usage_total = Usage()

        try:
            compiled = self.context_manager.compile(sid, live_messages=live_messages, auto_compress=False)
            messages = _api_messages([{"role": "system", "content": compiled.system_prompt}, *compiled.messages])
            yield from self._emit(sid, run_id, "context.compiled", "Context compiled", {
                "estimated_tokens": compiled.estimated_tokens,
                "context_budget_ratio": compiled.context_budget_ratio,
                "compressed": compiled.compressed,
                "compression": compiled.compression_metadata,
                "tools": [tool["function"]["name"] for tool in tools],
                "tool_status": registry.status(effective_toolsets, self.config.disabled_toolsets),
                "system_prompt_fingerprint": self.prompt_builder.fingerprint(),
                "tool_schema_fingerprint": tools_fingerprint,
            })

            final_content = ""
            iteration = 0
            while iteration < self.config.max_iterations:
                iteration += 1
                sid, messages, live_messages = self._compress_loop_context_if_needed(sid, run_id, messages, live_messages)
                yield from self._emit(sid, run_id, "model.requested", f"Model request {iteration}", {})
                start = time.perf_counter()
                response = self.model_client.complete(messages=copy.deepcopy(messages), tools=copy.deepcopy(tools))
                latency = time.perf_counter() - start
                usage_total = _add_usage(usage_total, response.usage)
                yield from self._emit(sid, run_id, "model.responded", f"Model response {iteration}", {
                    "latency_seconds": latency,
                    "usage": asdict(response.usage),
                    "cache_hit_rate": response.usage.cache_hit_rate,
                    "cache_hit_rate_known": response.usage.cache_hit_rate_known,
                    "tool_call_count": len(response.tool_calls),
                })

                assistant_message = _assistant_message(response)
                messages.append(assistant_message)
                assistant_message_id = self.store.add_message(
                    sid,
                    "assistant",
                    response.content,
                    tool_calls=response.tool_calls or None,
                    metadata={"iteration": iteration, "latency_seconds": latency, "usage": asdict(response.usage)},
                )
                live_assistant = dict(assistant_message)
                live_assistant["id"] = assistant_message_id
                live_messages.append(live_assistant)

                if not response.tool_calls:
                    final_content = response.content or ""
                    payload = {
                        "content": final_content,
                        "messages": messages,
                        "usage": asdict(usage_total),
                        "iterations": iteration,
                    }
                    yield from self._emit(sid, run_id, "run.completed", "Run completed", payload)
                    return

                tool_messages = self._execute_tool_calls(sid, run_id, response.tool_calls)
                for tool_msg in tool_messages:
                    messages.append(tool_msg)
                    tool_message_id = self.store.add_message(
                        sid,
                        "tool",
                        tool_msg["content"],
                        tool_call_id=tool_msg["tool_call_id"],
                        metadata={"name": tool_msg.get("name")},
                    )
                    live_tool = dict(tool_msg)
                    live_tool["id"] = tool_message_id
                    live_messages.append(live_tool)
                    yield from self._emit(sid, run_id, "tool.completed", f"Tool {tool_msg.get('name')} completed", tool_msg)

            message_text = f"Max iterations exceeded: {self.config.max_iterations}"
            yield from self._emit(sid, run_id, "run.failed", message_text, {"iterations": iteration})
        except ModelClientError as exc:
            yield from self._emit(sid, run_id, "run.failed", exc.message, {"error": exc.to_payload()})
        except Exception as exc:
            yield from self._emit(sid, run_id, "run.failed", f"{type(exc).__name__}: {exc}", {
                "error": {"category": "unknown", "message": f"{type(exc).__name__}: {exc}"}
            })

    def _execute_tool_calls(self, session_id: str, run_id: str, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        context = ToolContext(
            config=self.config,
            store=self.store,
            memory=self.memory,
            skills=self.skills,
            session_id=session_id,
            run_id=run_id,
            model_client=self.model_client,
            depth=self.depth,
        )
        results: list[tuple[int, dict[str, Any]]] = []
        max_workers = min(8, max(1, len(tool_calls)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._execute_one_tool, context, run_id, session_id, call): idx
                for idx, call in enumerate(tool_calls)
            }
            for future in as_completed(futures):
                results.append((futures[future], future.result()))
        return [item for _, item in sorted(results, key=lambda pair: pair[0])]

    def _compress_loop_context_if_needed(
        self,
        session_id: str,
        run_id: str,
        messages: list[dict[str, Any]],
        live_messages: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        threshold = int(self.config.context_budget_tokens * self.config.compression_trigger_ratio)
        if not self.config.compression_enabled or estimate_messages_tokens(messages) < threshold:
            return session_id, messages, live_messages
        compressed_live = self.context_manager.compress(session_id, live_messages)
        if compressed_live == live_messages:
            return session_id, messages, live_messages
        active_session_id = session_id
        if self.config.compression_create_continuation_session:
            active_session_id = self._create_continuation_session(session_id, run_id, compressed_live)
        compiled = self.context_manager.compile(active_session_id, live_messages=compressed_live, auto_compress=False)
        api_messages = _api_messages([{"role": "system", "content": compiled.system_prompt}, *compiled.messages])
        return active_session_id, api_messages, compressed_live

    def _create_continuation_session(
        self,
        parent_session_id: str,
        run_id: str,
        compressed_live: list[dict[str, Any]],
    ) -> str:
        parent = self.store.get_session(parent_session_id) or {}
        child_session_id = self.store.create_session(
            title=f"Continuation: {parent.get('title') or parent_session_id}",
            workspace_root=str(self.config.workspace_root),
            parent_session_id=parent_session_id,
            stage=str(parent.get("stage") or "continuation"),
            metadata={
                "kind": "compression_continuation",
                "parent_session_id": parent_session_id,
                "parent_run_id": run_id,
            },
        )
        for msg in compressed_live:
            if msg.get("role") == "system" and str(msg.get("content", "")).startswith("[WORKSPACE SNAPSHOT"):
                continue
            self.store.add_message(
                child_session_id,
                str(msg.get("role") or "user"),
                msg.get("content"),
                tool_call_id=msg.get("tool_call_id"),
                tool_calls=msg.get("tool_calls"),
                metadata={
                    "origin": "compression_continuation",
                    "parent_session_id": parent_session_id,
                    "parent_message_id": msg.get("id"),
                },
            )
        self.store.add_run_event(
            run_id,
            parent_session_id,
            "context.continuation_created",
            "Compression created continuation session",
            {"child_session_id": child_session_id},
        )
        return child_session_id

    def _execute_one_tool(self, context: ToolContext, run_id: str, session_id: str, call: dict[str, Any]) -> dict[str, Any]:
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        arguments = _parse_arguments(function.get("arguments"))
        result = registry.execute(name, arguments, context)
        call_id = str(call.get("id") or f"call_{uuid.uuid4().hex}")
        self.store.add_tool_call(
            run_id=run_id,
            session_id=session_id,
            tool_call_id=call_id,
            name=name,
            arguments=arguments,
            success=result.success,
            result=result.content,
            error=result.error,
            latency_seconds=result.latency_seconds,
        )
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": model_tool_result_content(name, result),
        }

    def _emit(self, session_id: str, run_id: str, event_type: str, message: str,
              payload: dict[str, Any]) -> Iterator[RunEvent]:
        safe_payload = json_safe(payload)
        event = RunEvent(type=event_type, message=message, session_id=session_id, run_id=run_id, payload=safe_payload)
        self.store.add_run_event(run_id, session_id, event_type, message, safe_payload)
        self.logger.write(session_id=session_id, run_id=run_id, event_type=event_type, message=message, payload=safe_payload)
        yield event


def _assistant_message(response: ModelResponse) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": response.content or ""}
    if response.tool_calls:
        msg["tool_calls"] = response.tool_calls
    return msg


def _api_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for msg in messages:
        item = {key: value for key, value in msg.items() if key in {"role", "content", "tool_call_id", "tool_calls"}}
        cleaned.append(item)
    return cleaned


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        parsed = json.loads(raw or "{}")
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must decode to an object")
        return parsed
    raise ValueError(f"tool arguments must be JSON object or string, got {type(raw).__name__}")


def _add_usage(left: Usage, right: Usage) -> Usage:
    return Usage(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        cached_prompt_tokens=left.cached_prompt_tokens + right.cached_prompt_tokens,
        cache_creation_input_tokens=left.cache_creation_input_tokens + right.cache_creation_input_tokens,
        cache_hit_rate_known=left.cache_hit_rate_known or right.cache_hit_rate_known,
    )
