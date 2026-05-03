from __future__ import annotations

import ast
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent_os.types import ToolResult

ToolHandler = Callable[..., ToolResult | str | dict[str, Any]]


@dataclass(slots=True)
class ToolEntry:
    name: str
    toolset: str
    schema: dict[str, Any]
    handler: ToolHandler
    check_fn: Callable[[], bool] | None = None
    description: str = ""


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolEntry] = {}

    def register(self, *, name: str, toolset: str, schema: dict[str, Any], handler: ToolHandler,
                 check_fn: Callable[[], bool] | None = None, description: str = "") -> None:
        if name in self._tools:
            raise ValueError(f"duplicate tool name: {name}")
        _validate_tool_schema(name, schema)
        self._tools[name] = ToolEntry(name=name, toolset=toolset, schema=schema, handler=handler,
                                      check_fn=check_fn, description=description)

    def get(self, name: str) -> ToolEntry | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def status(self, enabled_toolsets: set[str], disabled_toolsets: set[str] | None = None) -> list[dict[str, Any]]:
        disabled = disabled_toolsets or set()
        out: list[dict[str, Any]] = []
        for entry in sorted(self._tools.values(), key=lambda item: item.name):
            enabled = entry.toolset in enabled_toolsets and entry.toolset not in disabled
            available = _check_available(entry.check_fn) if entry.check_fn else True
            out.append({
                "name": entry.name,
                "toolset": entry.toolset,
                "enabled": enabled,
                "available": available,
                "visible": enabled and available,
            })
        return out

    def schemas(self, enabled_toolsets: set[str], disabled_toolsets: set[str] | None = None) -> list[dict[str, Any]]:
        disabled = disabled_toolsets or set()
        out: list[dict[str, Any]] = []
        for entry in sorted(self._tools.values(), key=lambda item: item.name):
            if entry.toolset not in enabled_toolsets or entry.toolset in disabled:
                continue
            if entry.check_fn and not _check_available(entry.check_fn):
                continue
            out.append(entry.schema)
        return out

    def execute(self, name: str, arguments: dict[str, Any], context: Any) -> ToolResult:
        entry = self.get(name)
        if not entry:
            return ToolResult.fail(f"Tool not found: {name}")
        if entry.check_fn and not _check_available(entry.check_fn):
            return ToolResult.fail(f"Tool unavailable: {name}")
        start = time.perf_counter()
        try:
            result = entry.handler(context=context, **arguments)
            latency = time.perf_counter() - start
            if isinstance(result, ToolResult):
                result.latency_seconds = latency
                return result
            if isinstance(result, dict):
                return ToolResult.ok(json.dumps(result, ensure_ascii=False), data=result, latency_seconds=latency)
            return ToolResult.ok(str(result), latency_seconds=latency)
        except Exception as exc:
            return ToolResult.fail(f"{type(exc).__name__}: {exc}", latency_seconds=time.perf_counter() - start)


_CHECK_CACHE: dict[Callable[[], bool], tuple[float, bool]] = {}
_CHECK_TTL = 30.0


def _check_available(fn: Callable[[], bool]) -> bool:
    now = time.monotonic()
    cached = _CHECK_CACHE.get(fn)
    if cached and now - cached[0] < _CHECK_TTL:
        return cached[1]
    try:
        value = bool(fn())
    except Exception:
        value = False
    _CHECK_CACHE[fn] = (now, value)
    return value


def _validate_tool_schema(name: str, schema: dict[str, Any]) -> None:
    if schema.get("type") != "function":
        raise ValueError(f"{name} schema type must be function")
    function = schema.get("function")
    if not isinstance(function, dict):
        raise ValueError(f"{name} schema missing function object")
    if function.get("name") != name:
        raise ValueError(f"{name} schema function.name mismatch")
    parameters = function.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        raise ValueError(f"{name} schema parameters must be an object schema")
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        raise ValueError(f"{name} schema parameters.properties must be an object")
    required = parameters.get("required") or []
    if not isinstance(required, list):
        raise ValueError(f"{name} schema parameters.required must be a list")
    missing = [item for item in required if item not in properties]
    if missing:
        raise ValueError(f"{name} schema required keys missing properties: {missing}")


registry = ToolRegistry()


def discover_builtin_tools(tools_dir: Path | None = None) -> list[str]:
    path = tools_dir or Path(__file__).resolve().parent
    modules = [
        f"agent_os.tools.{file.stem}"
        for file in sorted(path.glob("*.py"))
        if file.name not in {"__init__.py", "registry.py"} and _has_register_call(file)
    ]
    imported: list[str] = []
    for module in modules:
        importlib.import_module(module)
        imported.append(module)
    return imported


def _has_register_call(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return False
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if isinstance(func, ast.Attribute) and func.attr == "register":
            owner = func.value
            if isinstance(owner, ast.Name) and owner.id == "registry":
                return True
    return False


def function_schema(name: str, description: str, properties: dict[str, Any],
                    required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        },
    }
