# Agent OS

Business-agnostic bottom-layer agent runtime for upper-layer lawyer workflow or deep research systems.

V1 includes:

- OpenAI-compatible Chat Completions ReAct loop
- Tool registry with toolsets and availability filtering
- File, shell, memory, skill, and todo built-in tools
- Lightweight `delegate_task` subagent tool with child session isolation
- File-backed memory snapshots (`MEMORY.md`, `USER.md`)
- Skill index plus on-demand `skill_view` with metadata/dependency gating
- Pluggable context compression with todo preservation
- SQLite sessions, messages, tool calls, run events, summaries, todos, and child session links
- JSONL observability logs with basic secret redaction
- CLI plus terminal chat adapter over the shared runtime event stream

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export AGENT_OS_MODEL=qwen-plus
agent-os "hello"
agent-os chat
```

`agent-os chat` is the interactive terminal entry point. It shows concise runtime status for each turn, including estimated context tokens, context budget ratio, model latency, provider cache usage when reported, and tool execution time. In-chat commands include `/session`, `/history [n]`, `/tools`, `/events on|off`, and `/exit`.

Chat sessions are case-scoped. Use `/new <title>` to create an isolated session workspace under `.agent_os/sessions/<session_id>-<title>/workspace`. File, artifact, material, and shell tools run inside that session workspace when the session is resumed from SQLite, so files from different cases do not collide. Per-session process logs are mirrored to `.agent_os/sessions/<session>/logs/events.jsonl`; the global audit copy remains in `.agent_os/logs/<session_id>.jsonl`.

Inspection commands do not require a model API key:

```bash
agent-os sessions --json
agent-os show <session_id> --json
agent-os events <run_id> --json
agent-os tools
```

Optional `agent_os.json`:

```json
{
  "model": "qwen-plus",
  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "enabled_toolsets": ["files", "shell", "memory", "skills", "todo", "delegate"],
  "workspace_root": ".",
  "state_dir": ".agent_os",
  "max_iterations": 16,
  "compression_enabled": true,
  "subagents_enabled": true,
  "max_subagent_depth": 1
}
```

## Python API

```python
from agent_os import AgentOSConfig, AgentRuntime

config = AgentOSConfig.load()
runtime = AgentRuntime(config)
result = runtime.run("Summarize the workspace")
print(result.content)
```

## Smoke Checks

```bash
python scripts/smoke_fake_runtime.py
python scripts/smoke_openai_compatible.py
```

`smoke_fake_runtime.py` is offline and deterministic. `smoke_openai_compatible.py` requires a configured `.env` or environment variables for the target OpenAI-compatible provider.

## Boundaries

This repository implements only the reusable Agent OS layer. Domain workflows, legal case stages, and web workbench UX should live in upper-layer projects.
