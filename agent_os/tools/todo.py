from __future__ import annotations

import json
from typing import Any

from agent_os.types import ToolResult
from .context import ToolContext
from .registry import function_schema, registry


def todo_update(*, context: ToolContext, items: list[dict[str, Any]]) -> ToolResult:
    context.store.upsert_todos(context.session_id, items)
    todos = context.store.list_todos(context.session_id)
    return ToolResult.ok(json.dumps(todos, ensure_ascii=False), data=todos)


def todo_read(*, context: ToolContext) -> ToolResult:
    todos = context.store.list_todos(context.session_id)
    return ToolResult.ok(json.dumps(todos, ensure_ascii=False), data=todos)


def stage_update(*, context: ToolContext, stage: str) -> ToolResult:
    context.store.update_session_stage(context.session_id, stage)
    payload = {"session_id": context.session_id, "stage": stage}
    return ToolResult.ok(json.dumps(payload, ensure_ascii=False), data=payload)


registry.register(
    name="todo_update",
    toolset="todo",
    schema=function_schema("todo_update", "Create or update session todos. Preserve ids when updating existing todos.", {
        "items": {
            "type": "array",
            "description": "Todo items with optional id, text, status, metadata.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"]},
                    "stage": {"type": "string", "description": "Optional business-agnostic stage label stored in metadata."},
                    "metadata": {"type": "object"},
                },
                "required": ["text"],
            },
        },
    }, ["items"]),
    handler=todo_update,
)
registry.register(
    name="todo_read",
    toolset="todo",
    schema=function_schema("todo_read", "Read current session todos.", {}),
    handler=todo_read,
)
registry.register(
    name="stage_update",
    toolset="todo",
    schema=function_schema("stage_update", "Update the current session's business-agnostic stage label.", {
        "stage": {"type": "string", "description": "New session stage label."},
    }, ["stage"]),
    handler=stage_update,
)
