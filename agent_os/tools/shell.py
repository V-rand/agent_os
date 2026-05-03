from __future__ import annotations

import subprocess

from agent_os.types import ToolResult
from .context import ToolContext
from .registry import function_schema, registry


def shell_exec(*, context: ToolContext, command: str, timeout_seconds: float | None = None) -> ToolResult:
    timeout = timeout_seconds or context.config.shell_timeout_seconds
    completed = subprocess.run(
        command,
        shell=True,
        cwd=str(context.config.workspace_root),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    output = "\n".join(
        part for part in [
            f"exit_code={completed.returncode}",
            "stdout:\n" + completed.stdout if completed.stdout else "",
            "stderr:\n" + completed.stderr if completed.stderr else "",
        ] if part
    )
    return ToolResult(success=completed.returncode == 0, content=output, error=None if completed.returncode == 0 else output)


registry.register(
    name="shell_exec",
    toolset="shell",
    schema=function_schema("shell_exec", "Run a shell command in the workspace with timeout and return exit code/stdout/stderr.", {
        "command": {"type": "string", "description": "Shell command to execute."},
        "timeout_seconds": {"type": "number", "description": "Optional timeout in seconds."},
    }, ["command"]),
    handler=shell_exec,
)
