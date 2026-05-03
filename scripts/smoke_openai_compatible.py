from __future__ import annotations

from agent_os import AgentOSConfig, AgentRuntime


def main() -> int:
    config = AgentOSConfig.load()
    runtime = AgentRuntime(config)
    result = runtime.run("Reply with exactly: Agent OS smoke ok")
    print(result.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
