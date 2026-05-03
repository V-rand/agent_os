from __future__ import annotations

import json

from agent_os.types import ToolResult
from .context import ToolContext
from .registry import function_schema, registry


def memory_tool(*, context: ToolContext, action: str, target: str = "memory",
                content: str | None = None, match: str | None = None) -> ToolResult:
    try:
        if action == "read":
            entries = context.memory.read_live(target)
            payload = {
                "target": target,
                "entries": entries,
                "snapshot_changed": False,
                "note": "Live memory is shown. Current prompt snapshot is unchanged until the next session/runtime initialization.",
            }
            return ToolResult.ok(json.dumps(payload, ensure_ascii=False), data=payload)
        if action == "add":
            if content is None:
                raise ValueError("content is required for add")
            message = context.memory.add(target, content)
            payload = {"target": target, "action": action, "message": message, "snapshot_changed": False}
            return ToolResult.ok(json.dumps(payload, ensure_ascii=False), data=payload)
        if action == "replace":
            if content is None or match is None:
                raise ValueError("content and match are required for replace")
            message = context.memory.replace(target, match, content)
            payload = {"target": target, "action": action, "message": message, "snapshot_changed": False}
            return ToolResult.ok(json.dumps(payload, ensure_ascii=False), data=payload)
        if action == "remove":
            if match is None:
                raise ValueError("match is required for remove")
            message = context.memory.remove(target, match)
            payload = {"target": target, "action": action, "message": message, "snapshot_changed": False}
            return ToolResult.ok(json.dumps(payload, ensure_ascii=False), data=payload)
        raise ValueError("action must be read, add, replace, or remove")
    except Exception as exc:
        payload = {"target": target, "action": action, "error": f"{type(exc).__name__}: {exc}", "snapshot_changed": False}
        return ToolResult.fail(json.dumps(payload, ensure_ascii=False), data=payload)


registry.register(
    name="memory",
    toolset="memory",
    schema=function_schema("memory", "Read or update durable curated memory. Writes do not change the current prompt snapshot.", {
        "action": {"type": "string", "enum": ["read", "add", "replace", "remove"], "description": "Memory operation."},
        "target": {"type": "string", "enum": ["memory", "user"], "description": "Memory file target.", "default": "memory"},
        "content": {"type": "string", "description": "Entry content for add/replace."},
        "match": {"type": "string", "description": "Unique substring for replace/remove."},
    }, ["action"]),
    handler=memory_tool,
)
