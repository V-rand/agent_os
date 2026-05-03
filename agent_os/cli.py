from __future__ import annotations

import argparse
import json
import sys

from .config import AgentOSConfig
from .runtime import AgentRuntime
from .storage import SQLiteStore
from .tools.registry import discover_builtin_tools, registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-os")
    parser.add_argument("--config", help="Path to agent_os.yaml/json.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a message through the agent.")
    run_parser.add_argument("message", nargs="*", help="Message to send. If omitted, stdin is read.")
    run_parser.add_argument("--session-id", help="Existing session id to resume.")
    run_parser.add_argument("--json", action="store_true", help="Print final result as JSON.")
    run_parser.add_argument("--events", action="store_true", help="Print runtime events.")

    sessions_parser = subparsers.add_parser("sessions", help="List recent sessions.")
    sessions_parser.add_argument("--limit", type=int, default=50)
    sessions_parser.add_argument("--json", action="store_true")

    show_parser = subparsers.add_parser("show", help="Show a session snapshot.")
    show_parser.add_argument("session_id")
    show_parser.add_argument("--json", action="store_true")

    events_parser = subparsers.add_parser("events", help="Show run events.")
    events_parser.add_argument("run_id")
    events_parser.add_argument("--json", action="store_true")

    tools_parser = subparsers.add_parser("tools", help="List currently available tools.")
    tools_parser.add_argument("--json", action="store_true")
    return parser


def build_legacy_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-os")
    parser.add_argument("message", nargs="*", help="Message to send. If omitted, stdin is read.")
    parser.add_argument("--config", help="Path to agent_os.yaml/json.")
    parser.add_argument("--session-id", help="Existing session id to resume.")
    parser.add_argument("--json", action="store_true", help="Print final result as JSON.")
    parser.add_argument("--events", action="store_true", help="Print runtime events.")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    commands = {"run", "sessions", "show", "events", "tools"}
    if not any(arg in commands for arg in raw_args):
        args = build_legacy_run_parser().parse_args(raw_args)
        return _run_message(args)
    args = build_parser().parse_args(raw_args)
    if args.command in {None, "run"}:
        return _run_message(args)
    config = AgentOSConfig.load(args.config)
    store = SQLiteStore(config.db_path)
    if args.command == "sessions":
        sessions = store.list_sessions(limit=args.limit)
        return _print_json_or_text(sessions, args.json, _render_sessions)
    if args.command == "show":
        try:
            snapshot = store.get_session_snapshot(args.session_id)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return _print_json_or_text(snapshot, args.json, _render_snapshot)
    if args.command == "events":
        events = store.get_run_events(args.run_id)
        return _print_json_or_text(events, args.json, _render_events)
    if args.command == "tools":
        discover_builtin_tools()
        effective = config.enabled_toolsets - config.disabled_toolsets
        tools = registry.schemas(effective, config.disabled_toolsets)
        payload = [{"name": item["function"]["name"], "description": item["function"].get("description", "")} for item in tools]
        return _print_json_or_text(payload, args.json, _render_tools)
    return 2


def _run_message(args: argparse.Namespace) -> int:
    message = " ".join(args.message).strip() or sys.stdin.read().strip()
    if not message:
        print("message is required", file=sys.stderr)
        return 2
    config = AgentOSConfig.load(args.config)
    runtime = AgentRuntime(config)
    final_payload = None
    for event in runtime.stream(message, session_id=args.session_id):
        if args.events:
            print(f"[{event.type}] {event.message}")
            if event.type in {"tool.completed", "model.responded"}:
                print(json.dumps(event.payload, ensure_ascii=False))
        if event.type == "run.completed":
            final_payload = event.payload
        if event.type == "run.failed":
            print(event.message, file=sys.stderr)
            return 1
    if final_payload is None:
        print("run did not complete", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(final_payload, ensure_ascii=False, indent=2))
    else:
        print(final_payload.get("content", ""))
    return 0


def _print_json_or_text(value, as_json: bool, renderer) -> int:
    if as_json:
        print(json.dumps(value, ensure_ascii=False, indent=2))
    else:
        print(renderer(value))
    return 0


def _render_sessions(sessions: list[dict]) -> str:
    if not sessions:
        return "No sessions."
    return "\n".join(f"{item['id']}  {item['updated_at']}  {item['title']}  stage={item.get('stage', '')}" for item in sessions)


def _render_snapshot(snapshot: dict) -> str:
    session = snapshot["session"]
    return "\n".join([
        f"Session: {session['id']}",
        f"Title: {session['title']}",
        f"Stage: {session.get('stage', '')}",
        f"Messages: {len(snapshot['messages'])}",
        f"Todos: {len(snapshot['todos'])}",
        f"Children: {len(snapshot['children'])}",
        f"Summaries: {len(snapshot['summaries'])}",
    ])


def _render_events(events: list[dict]) -> str:
    if not events:
        return "No events."
    return "\n".join(f"{item['created_at']}  {item['type']}  {item['message']}" for item in events)


def _render_tools(tools: list[dict]) -> str:
    if not tools:
        return "No tools available."
    return "\n".join(f"{item['name']}: {item['description']}" for item in tools)


if __name__ == "__main__":
    raise SystemExit(main())
