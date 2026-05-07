from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO

from agent_os.config import AgentOSConfig
from agent_os.runtime import AgentRuntime
from agent_os.session_manager import SessionManager
from agent_os.storage import SQLiteStore
from agent_os.tools.registry import discover_builtin_tools, registry
from agent_os.types import RunEvent
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table


InputFunc = Callable[[str], str]


@dataclass(slots=True)
class TerminalChatOptions:
    config_path: str | None = None
    session_id: str | None = None
    raw_events: bool = False
    json_output: bool = False
    input_func: InputFunc | None = None
    output: TextIO | None = None
    error_output: TextIO | None = None


class TerminalChat:
    """Hermes-style terminal adapter over the runtime event stream.

    The runtime remains the only execution layer. This class only renders events
    and commands, so a future web UI can consume the same RunEvent stream.
    """

    def __init__(self, options: TerminalChatOptions):
        self.options = options
        if self.options.output is None:
            self.options.output = sys.stdout
        if self.options.error_output is None:
            self.options.error_output = sys.stderr
        self.config = AgentOSConfig.load(options.config_path)
        self.runtime = AgentRuntime(self.config)
        self.store = SQLiteStore(self.config.db_path)
        self.sessions = SessionManager(self.config, self.store)
        self.session_id = options.session_id
        self.input_func = options.input_func or input
        self.console = Console(file=self.options.output, force_terminal=False, color_system=None)
        self.error_console = Console(file=self.options.error_output, force_terminal=False, color_system=None)
        if self.session_id:
            self.sessions.resolve(self.session_id)

    def run(self) -> int:
        self._print_banner()
        while True:
            try:
                message = self.input_func("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(file=self.options.output)
                return 0
            if not message:
                continue
            try:
                if self._handle_command(message):
                    continue
            except SystemExit as exc:
                return int(exc.code or 0)
            final_payload = self._run_turn(message)
            if final_payload is None:
                continue
            if self.options.json_output:
                print(json.dumps(final_payload, ensure_ascii=False, indent=2), file=self.options.output)
            else:
                self.console.print(Panel(Markdown(str(final_payload.get("content", ""))), title="assistant", border_style="green"))

    def _print_banner(self) -> None:
        body = "\n".join([
            f"model: {self.config.model}",
            f"project: {self.config.workspace_root}",
            f"state: {self.config.state_dir}",
            f"session: {self.session_id or 'none'}",
            "commands: /new [title] /use <id> /sessions /session /files [path] /history [n] /tools /events on|off /exit",
        ])
        self.console.print(Panel(body, title="Agent OS chat", border_style="cyan"))

    def _handle_command(self, message: str) -> bool:
        parts = message.strip().split()
        if not parts:
            return True
        command = parts[0].lower()
        if command in {"/exit", "/quit"}:
            raise SystemExit(0)
        if command == "/new":
            title = " ".join(parts[1:]).strip() or "New case"
            if self.session_id:
                self.sessions.end(self.session_id, reason="new_session")
            session = self.sessions.create(title=title)
            self.session_id = str(session["id"])
            self._print_session()
            return True
        if command == "/use":
            if len(parts) < 2:
                print("usage: /use <session_id>", file=self.options.output)
                return True
            session = self.sessions.resolve(parts[1])
            self.session_id = str(session["id"])
            self._print_session()
            return True
        if command == "/sessions":
            limit = _parse_int(parts[1], 20) if len(parts) > 1 else 20
            self._print_sessions(limit)
            return True
        if command == "/rename":
            if not self.session_id:
                print("(no session yet)", file=self.options.output)
                return True
            title = " ".join(parts[1:]).strip()
            if not title:
                print("usage: /rename <title>", file=self.options.output)
                return True
            self.sessions.rename(self.session_id, title)
            self._print_session()
            return True
        if command == "/help":
            self._print_help()
            return True
        if command == "/session":
            self._print_session()
            return True
        if command == "/history":
            limit = _parse_int(parts[1], 12) if len(parts) > 1 else 12
            self._print_history(limit)
            return True
        if command == "/tools":
            self._print_tools()
            return True
        if command == "/files":
            path = parts[1] if len(parts) > 1 else "."
            self._print_files(path)
            return True
        if command == "/events":
            if len(parts) > 1:
                self.options.raw_events = parts[1].lower() in {"on", "true", "1", "yes"}
            print(f"raw_events={'on' if self.options.raw_events else 'off'}", file=self.options.output)
            return True
        return False

    def _print_help(self) -> None:
        lines = [
            "/new [title]          Create a case session with its own workspace.",
            "/use <session_id>     Switch to an existing session.",
            "/sessions [n]         List recent sessions.",
            "/session              Show the active session and paths.",
            "/rename <title>       Rename the active session.",
            "/files [path]         List files in the active session workspace.",
            "/history [n]          Show recent persisted messages for this session.",
            "/tools                Show visible tools and availability.",
            "/events on|off        Toggle raw runtime event JSON.",
            "/exit                 Leave the terminal chat.",
        ]
        self.console.print(Panel("\n".join(lines), title="commands", border_style="blue"))

    def _print_session(self) -> None:
        if not self.session_id:
            print("(no session yet)", file=self.options.output)
            return
        session = self.sessions.resolve(self.session_id)
        workspace = Path(str(session["workspace_root"]))
        body = "\n".join([
            f"id: {session['id']}",
            f"title: {session['title']}",
            f"stage: {session.get('stage', '')}",
            f"status: {session.get('status', '')}",
            f"workspace: {workspace}",
            f"log: {workspace.parent / 'logs' / 'events.jsonl'}",
            f"global_log: {self.config.log_dir / (str(session['id']) + '.jsonl')}",
            f"messages: {len(self.store.list_messages(self.session_id))}",
            f"todos: {len(self.store.list_todos(self.session_id))}",
        ])
        self.console.print(Panel(body, title="session", border_style="cyan"))

    def _print_sessions(self, limit: int) -> None:
        sessions = self.sessions.list(limit=max(1, limit))
        if not sessions:
            print("No sessions.", file=self.options.output)
            return
        table = Table(title="recent sessions")
        table.add_column("id")
        table.add_column("title")
        table.add_column("stage")
        table.add_column("updated")
        table.add_column("workspace")
        for session in sessions:
            table.add_row(
                str(session["id"]),
                str(session["title"]),
                str(session.get("stage") or ""),
                str(session.get("updated_at") or ""),
                _compact(str(session.get("workspace_root") or ""), 60),
            )
        self.console.print(table)

    def _print_history(self, limit: int) -> None:
        if not self.session_id:
            print("(no session yet)", file=self.options.output)
            return
        messages = self.store.list_messages(self.session_id, limit=max(1, limit))
        if not messages:
            print("(empty history)", file=self.options.output)
            return
        for msg in messages:
            content = _compact(str(msg.get("content") or ""), 160)
            print(f"{msg.get('id')} {msg.get('role')}> {content}", file=self.options.output)

    def _print_files(self, path: str) -> None:
        if not self.session_id:
            print("(no session yet)", file=self.options.output)
            return
        session = self.sessions.resolve(self.session_id)
        root = Path(str(session["workspace_root"])).resolve()
        target = (root / path).resolve()
        if target != root and root not in target.parents:
            print(f"path escapes session workspace: {path}", file=self.options.error_output)
            return
        if not target.exists():
            print(f"path not found: {path}", file=self.options.error_output)
            return
        table = Table(title=f"files: {target.relative_to(root) if target != root else '.'}")
        table.add_column("name")
        table.add_column("type")
        table.add_column("bytes", justify="right")
        items = sorted(target.iterdir()) if target.is_dir() else [target]
        for item in items[:200]:
            stat = item.stat()
            table.add_row(item.name + ("/" if item.is_dir() else ""), "dir" if item.is_dir() else "file", str(stat.st_size))
        self.console.print(table)

    def _print_tools(self) -> None:
        discover_builtin_tools()
        effective = self.config.enabled_toolsets - self.config.disabled_toolsets
        statuses = registry.status(effective, self.config.disabled_toolsets)
        visible = [item for item in statuses if item["visible"]]
        if not visible:
            print("No visible tools.", file=self.options.output)
            return
        for item in visible:
            print(f"{item['name']}  toolset={item['toolset']} available={item['available']}", file=self.options.output)

    def _run_turn(self, message: str) -> dict | None:
        final_payload = None
        try:
            if self.session_id is None:
                session = self.sessions.create(title=_compact(message, 80) or "New case")
                self.session_id = str(session["id"])
                if not self.options.json_output:
                    self._print_session()
            events = self.runtime.stream(message, session_id=self.session_id)
            for event in events:
                self.session_id = event.session_id
                rendered = render_event_status(event)
                if rendered and not self.options.json_output:
                    print(rendered, file=self.options.output)
                if self.options.raw_events:
                    print(json.dumps(_event_to_dict(event), ensure_ascii=False), file=self.options.output)
                if event.type == "run.completed":
                    final_payload = event.payload
                if event.type == "run.failed":
                    self.error_console.print(Panel(event.message, title="run failed", border_style="red"))
                    return None
        except SystemExit:
            raise
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=self.options.error_output)
            return None
        return final_payload


def run_terminal_chat(
    *,
    config_path: str | None,
    session_id: str | None,
    raw_events: bool,
    json_output: bool,
) -> int:
    try:
        return TerminalChat(TerminalChatOptions(
            config_path=config_path,
            session_id=session_id,
            raw_events=raw_events,
            json_output=json_output,
        )).run()
    except SystemExit as exc:
        return int(exc.code or 0)


def render_event_status(event: RunEvent) -> str | None:
    payload = event.payload
    if event.type == "context.compiled":
        tokens = int(payload.get("estimated_tokens") or 0)
        ratio = float(payload.get("context_budget_ratio") or 0.0)
        tools = payload.get("tools") or []
        compressed = " compressed" if payload.get("compressed") else ""
        workspace = payload.get("workspace_root")
        suffix = f", workspace={workspace}" if workspace else ""
        return f"context> {tokens} tokens ({ratio:.1%} budget), tools={len(tools)}{compressed}{suffix}"
    if event.type == "model.requested":
        return "model> requested"
    if event.type == "model.responded":
        usage = payload.get("usage") or {}
        latency = float(payload.get("latency_seconds") or 0.0)
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        total = int(usage.get("total_tokens") or 0)
        cache = _format_cache(payload)
        tool_calls = int(payload.get("tool_call_count") or 0)
        return f"model> {latency:.2f}s tokens={prompt}/{completion}/{total}{cache} tool_calls={tool_calls}"
    if event.type == "tool.completed":
        name = str(payload.get("name") or "tool")
        latency = float(payload.get("latency_seconds") or 0.0)
        status = "ok" if payload.get("success", True) else "error"
        error = payload.get("error")
        suffix = f" error={_compact(str(error), 120)}" if error else ""
        return f"tool> {name} {status} {latency:.2f}s{suffix}"
    if event.type == "context.continuation_created":
        child = payload.get("child_session_id")
        return f"context> continuation session={child}"
    return None


def _format_cache(payload: dict) -> str:
    usage = payload.get("usage") or {}
    cached = int(usage.get("cached_prompt_tokens") or 0)
    created = int(usage.get("cache_creation_input_tokens") or 0)
    if not payload.get("cache_hit_rate_known"):
        return " cache=unknown"
    rate = float(payload.get("cache_hit_rate") or 0.0)
    return f" cache={cached} ({rate:.1%}) created={created}"


def _compact(value: str, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)] + "..."


def _parse_int(value: str, default: int) -> int:
    try:
        return int(value)
    except ValueError:
        return default


def _event_to_dict(event: RunEvent) -> dict:
    if hasattr(event, "to_dict"):
        return event.to_dict()
    return {
        "type": event.type,
        "message": event.message,
        "session_id": event.session_id,
        "run_id": event.run_id,
        "payload": event.payload,
    }
