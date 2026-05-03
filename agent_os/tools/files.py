from __future__ import annotations

import os
import tempfile
import hashlib
from pathlib import Path

from agent_os.types import ToolResult
from .context import ToolContext
from .registry import function_schema, registry


def _resolve(root: Path, path: str) -> Path:
    candidate = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    root = root.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path escapes workspace root: {path}")
    return candidate


def file_read(*, context: ToolContext, path: str, max_chars: int | None = None) -> ToolResult:
    target = _resolve(context.config.workspace_root, path)
    text = target.read_text(encoding="utf-8", errors="replace")
    truncated = False
    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
        truncated = True
    result = ToolResult.ok(text)
    result.metadata = {
        "path": str(target.relative_to(context.config.workspace_root)),
        "truncated": truncated,
        "explicit_limit": max_chars is not None,
    }
    return result


def file_info(*, context: ToolContext, path: str) -> ToolResult:
    target = _resolve(context.config.workspace_root, path)
    stat = target.stat()
    data = {
        "path": str(target.relative_to(context.config.workspace_root)),
        "is_file": target.is_file(),
        "is_dir": target.is_dir(),
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
    }
    if target.is_file():
        digest = hashlib.sha256()
        with target.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        data["sha256"] = digest.hexdigest()
        try:
            sample = target.read_text(encoding="utf-8", errors="strict")[:4096]
            data["text"] = True
            data["estimated_tokens"] = max(1, len(sample) // 4) if stat.st_size <= 4096 else max(1, stat.st_size // 4)
        except UnicodeError:
            data["text"] = False
    return ToolResult.ok(str(data), data=data)


def file_read_chunk(*, context: ToolContext, path: str, offset: int = 0, max_chars: int = 8000) -> ToolResult:
    target = _resolve(context.config.workspace_root, path)
    text = target.read_text(encoding="utf-8", errors="replace")
    start = max(0, int(offset))
    end = min(len(text), start + max(1, int(max_chars)))
    chunk = text[start:end]
    result = ToolResult.ok(chunk)
    result.metadata = {
        "path": str(target.relative_to(context.config.workspace_root)),
        "offset": start,
        "end": end,
        "total_chars": len(text),
        "truncated": end < len(text),
        "explicit_chunk": True,
    }
    return result


def file_write(*, context: ToolContext, path: str, content: str) -> ToolResult:
    target = _resolve(context.config.workspace_root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name, dir=str(target.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return ToolResult.ok(f"Wrote {target.relative_to(context.config.workspace_root)}")


def file_list(*, context: ToolContext, path: str = ".", max_entries: int = 200) -> ToolResult:
    target = _resolve(context.config.workspace_root, path)
    entries = []
    for item in sorted(target.iterdir())[:max_entries]:
        suffix = "/" if item.is_dir() else ""
        entries.append(f"{item.name}{suffix}")
    return ToolResult.ok("\n".join(entries))


def file_search(*, context: ToolContext, query: str, path: str = ".", max_matches: int = 100) -> ToolResult:
    target = _resolve(context.config.workspace_root, path)
    matches: list[str] = []
    for file in sorted(target.rglob("*")):
        if len(matches) >= max_matches:
            break
        if not file.is_file():
            continue
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if query in line:
                rel = file.relative_to(context.config.workspace_root)
                matches.append(f"{rel}:{line_no}: {line[:300]}")
                if len(matches) >= max_matches:
                    break
    return ToolResult.ok("\n".join(matches) if matches else "No matches.")


registry.register(
    name="file_read",
    toolset="files",
    schema=function_schema("file_read", "Read a UTF-8 text file under the workspace root.", {
        "path": {"type": "string", "description": "Workspace-relative file path."},
        "max_chars": {"type": "integer", "description": "Optional maximum characters to return. Omit to return the full file."},
    }, ["path"]),
    handler=file_read,
)
registry.register(
    name="file_write",
    toolset="files",
    schema=function_schema("file_write", "Write a UTF-8 text file under the workspace root.", {
        "path": {"type": "string", "description": "Workspace-relative file path."},
        "content": {"type": "string", "description": "Complete file content."},
    }, ["path", "content"]),
    handler=file_write,
)
registry.register(
    name="file_info",
    toolset="files",
    schema=function_schema("file_info", "Return file metadata, hash, and estimated text tokens under the workspace root.", {
        "path": {"type": "string", "description": "Workspace-relative file path."},
    }, ["path"]),
    handler=file_info,
)
registry.register(
    name="file_read_chunk",
    toolset="files",
    schema=function_schema("file_read_chunk", "Read an explicit character chunk from a UTF-8 text file under the workspace root.", {
        "path": {"type": "string", "description": "Workspace-relative file path."},
        "offset": {"type": "integer", "description": "Character offset to start reading.", "default": 0},
        "max_chars": {"type": "integer", "description": "Maximum characters to return.", "default": 8000},
    }, ["path"]),
    handler=file_read_chunk,
)
registry.register(
    name="file_list",
    toolset="files",
    schema=function_schema("file_list", "List direct children of a workspace directory.", {
        "path": {"type": "string", "description": "Workspace-relative directory path.", "default": "."},
        "max_entries": {"type": "integer", "description": "Maximum entries to return.", "default": 200},
    }),
    handler=file_list,
)
registry.register(
    name="file_search",
    toolset="files",
    schema=function_schema("file_search", "Search text files under a workspace path for a literal query.", {
        "query": {"type": "string", "description": "Literal text to search for."},
        "path": {"type": "string", "description": "Workspace-relative directory path.", "default": "."},
        "max_matches": {"type": "integer", "description": "Maximum matching lines.", "default": 100},
    }, ["query"]),
    handler=file_search,
)
