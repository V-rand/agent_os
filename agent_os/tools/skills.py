from __future__ import annotations

from agent_os.types import ToolResult
from .context import ToolContext
from .registry import function_schema, registry


def skill_view(*, context: ToolContext, name: str) -> ToolResult:
    enabled = context.config.enabled_toolsets - context.config.disabled_toolsets
    payload = context.skills.view(name, enabled)
    return ToolResult.ok(str(payload["content"]), data=payload)


registry.register(
    name="skill_view",
    toolset="skills",
    schema=function_schema("skill_view", "Load full instructions for one available skill.", {
        "name": {"type": "string", "description": "Skill name from the available skills index."},
    }, ["name"]),
    handler=skill_view,
)
