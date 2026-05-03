from __future__ import annotations

import json

from agent_os.types import ToolResult
from .context import ToolContext
from .registry import function_schema, registry


def delegate_task(*, context: ToolContext, task: str, expected_output: str | None = None) -> ToolResult:
    if not context.config.subagents_enabled:
        return ToolResult.fail("Subagents are disabled by configuration.")
    if context.depth >= context.config.max_subagent_depth:
        return ToolResult.fail(
            f"Subagent depth limit exceeded: depth={context.depth}, max={context.config.max_subagent_depth}"
        )

    from agent_os.runtime import AgentRuntime

    child_prompt = task
    if expected_output:
        child_prompt += f"\n\nExpected output:\n{expected_output}"
    child_session_id = context.store.create_session(
        title=f"Subagent: {task[:60]}",
        workspace_root=str(context.config.workspace_root),
        parent_session_id=context.session_id,
        stage="subagent",
        metadata={"parent_run_id": context.run_id, "depth": context.depth + 1},
    )
    child_runtime = AgentRuntime(context.config, model_client=context.model_client, depth=context.depth + 1)
    result = child_runtime.run(child_prompt, session_id=child_session_id)
    payload = {
        "child_session_id": child_session_id,
        "content": result.content,
        "iterations": result.iterations,
        "usage": {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "total_tokens": result.usage.total_tokens,
        },
    }
    return ToolResult.ok(json.dumps(payload, ensure_ascii=False), data=payload)


registry.register(
    name="delegate_task",
    toolset="delegate",
    schema=function_schema("delegate_task", "Run a focused task in an isolated child agent session and return only its final result.", {
        "task": {"type": "string", "description": "Self-contained task for the child agent."},
        "expected_output": {"type": "string", "description": "Optional expected output format or success criteria."},
    }, ["task"]),
    handler=delegate_task,
)
