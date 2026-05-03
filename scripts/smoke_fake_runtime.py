from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from agent_os import AgentOSConfig, AgentRuntime
from agent_os.types import ModelResponse


class FakeClient:
    def __init__(self):
        self.requests: list[dict[str, Any]] = []

    def complete(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        self.requests.append({"messages": messages, "tools": tools})
        if len(self.requests) == 1:
            return ModelResponse(
                content=None,
                tool_calls=[{
                    "id": "call_read",
                    "type": "function",
                    "function": {"name": "file_read", "arguments": json.dumps({"path": "hello.txt"})},
                }],
            )
        return ModelResponse(content="Fake smoke passed.")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "hello.txt").write_text("hello from fake smoke", encoding="utf-8")
        config = AgentOSConfig(
            model="fake",
            api_key="fake",
            base_url="https://example.test/v1",
            workspace_root=root,
            state_dir=root / ".agent_os",
        )
        result = AgentRuntime(config, model_client=FakeClient()).run("read hello.txt")
        print(result.content)
        return 0 if "passed" in result.content else 1


if __name__ == "__main__":
    raise SystemExit(main())
