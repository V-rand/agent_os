from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agent_os.types import ToolResult
from .context import ToolContext
from .files import _resolve
from .registry import function_schema, registry


def artifact_upsert(*, context: ToolContext, path: str, content: str, kind: str = "",
                    stage: str = "", source_paths: list[str] | None = None,
                    metadata: dict[str, Any] | None = None) -> ToolResult:
    target = _resolve(context.config.workspace_root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    meta = {**(metadata or {}), **_file_metadata(context.config.workspace_root, target)}
    artifact = context.store.upsert_artifact(
        context.session_id,
        str(target.relative_to(context.config.workspace_root)),
        kind=kind,
        stage=stage,
        source_paths=source_paths or [],
        metadata=meta,
    )
    return ToolResult.ok(json.dumps(artifact, ensure_ascii=False), data=artifact)


def artifact_read(*, context: ToolContext, path: str) -> ToolResult:
    artifact = context.store.read_artifact(context.session_id, path)
    target = _resolve(context.config.workspace_root, path)
    content = target.read_text(encoding="utf-8", errors="replace")
    payload = {"artifact": artifact, "content": content}
    return ToolResult.ok(json.dumps(payload, ensure_ascii=False), data=payload)


def artifact_list(*, context: ToolContext, stage: str | None = None, kind: str | None = None) -> ToolResult:
    artifacts = context.store.list_artifacts(context.session_id, stage=stage, kind=kind)
    return ToolResult.ok(json.dumps(artifacts, ensure_ascii=False), data=artifacts)


def material_register(*, context: ToolContext, path: str, kind: str = "", stage: str = "",
                      title: str = "", metadata: dict[str, Any] | None = None) -> ToolResult:
    target = _resolve(context.config.workspace_root, path)
    meta = {**(metadata or {}), **_file_metadata(context.config.workspace_root, target)}
    material = context.store.upsert_material(
        context.session_id,
        str(target.relative_to(context.config.workspace_root)),
        kind=kind,
        stage=stage,
        title=title or target.name,
        metadata=meta,
    )
    return ToolResult.ok(json.dumps(material, ensure_ascii=False), data=material)


def material_list(*, context: ToolContext, stage: str | None = None, kind: str | None = None) -> ToolResult:
    materials = context.store.list_materials(context.session_id, stage=stage, kind=kind)
    return ToolResult.ok(json.dumps(materials, ensure_ascii=False), data=materials)


def _file_metadata(root: Path, target: Path) -> dict[str, Any]:
    stat = target.stat()
    meta: dict[str, Any] = {
        "path": str(target.relative_to(root)),
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "is_file": target.is_file(),
        "is_dir": target.is_dir(),
    }
    if target.is_file():
        digest = hashlib.sha256()
        with target.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        meta["sha256"] = digest.hexdigest()
        if _is_probably_text(target):
            text = target.read_text(encoding="utf-8", errors="replace")
            meta["estimated_tokens"] = max(1, len(text) // 4)
            meta["summary"] = text[:500]
    return meta


def _is_probably_text(path: Path) -> bool:
    try:
        path.read_text(encoding="utf-8", errors="strict")
        return True
    except UnicodeError:
        return False


registry.register(
    name="artifact_upsert",
    toolset="artifacts",
    schema=function_schema("artifact_upsert", "Write a workspace artifact file and index its provenance metadata in SQLite.", {
        "path": {"type": "string"},
        "content": {"type": "string"},
        "kind": {"type": "string", "default": ""},
        "stage": {"type": "string", "default": ""},
        "source_paths": {"type": "array", "items": {"type": "string"}, "default": []},
        "metadata": {"type": "object", "default": {}},
    }, ["path", "content"]),
    handler=artifact_upsert,
)
registry.register(
    name="artifact_read",
    toolset="artifacts",
    schema=function_schema("artifact_read", "Read a workspace artifact file plus its SQLite index metadata.", {
        "path": {"type": "string"},
    }, ["path"]),
    handler=artifact_read,
)
registry.register(
    name="artifact_list",
    toolset="artifacts",
    schema=function_schema("artifact_list", "List indexed artifacts for the current session.", {
        "stage": {"type": "string"},
        "kind": {"type": "string"},
    }),
    handler=artifact_list,
)
registry.register(
    name="material_register",
    toolset="materials",
    schema=function_schema("material_register", "Index an existing workspace material file without copying its content into SQLite.", {
        "path": {"type": "string"},
        "kind": {"type": "string", "default": ""},
        "stage": {"type": "string", "default": ""},
        "title": {"type": "string", "default": ""},
        "metadata": {"type": "object", "default": {}},
    }, ["path"]),
    handler=material_register,
)
registry.register(
    name="material_list",
    toolset="materials",
    schema=function_schema("material_list", "List indexed source materials for the current session.", {
        "stage": {"type": "string"},
        "kind": {"type": "string"},
    }),
    handler=material_list,
)
