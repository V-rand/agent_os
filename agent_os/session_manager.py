from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import AgentOSConfig
from .storage import SQLiteStore


_SAFE_TITLE_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(slots=True)
class SessionPaths:
    root: Path
    workspace: Path
    logs: Path
    exports: Path


class SessionManager:
    """Creates and resolves case-scoped sessions and workspaces."""

    def __init__(self, config: AgentOSConfig, store: SQLiteStore | None = None):
        self.config = config
        self.store = store or SQLiteStore(config.db_path)

    def create(self, *, title: str = "New session", stage: str = "ready") -> dict:
        session_id = self.store.create_session(
            title=title or "New session",
            workspace_root=str(self.config.workspace_root),
            stage=stage,
            metadata={
                "kind": "case_session",
                "base_workspace_root": str(self.config.workspace_root),
            },
        )
        paths = self.paths_for(session_id, title=title)
        self._ensure_paths(paths)
        self.store.update_session_workspace(session_id, str(paths.workspace))
        return self.store.get_session(session_id) or {}

    def resolve(self, session_id: str) -> dict:
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError(f"session not found: {session_id}")
        workspace = Path(str(session.get("workspace_root") or "")).expanduser()
        if not workspace.is_absolute():
            workspace = (self.config.workspace_root / workspace).resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        self._ensure_paths(self.paths_from_workspace(workspace))
        return session

    def list(self, *, limit: int = 50) -> list[dict]:
        return self.store.list_sessions(limit=limit)

    def rename(self, session_id: str, title: str) -> dict:
        self.store.update_session_title(session_id, title)
        return self.resolve(session_id)

    def end(self, session_id: str, *, reason: str = "closed") -> None:
        self.store.end_session(session_id, reason=reason)

    def paths_for(self, session_id: str, *, title: str = "") -> SessionPaths:
        suffix = _safe_suffix(title)
        dirname = session_id if not suffix else f"{session_id}-{suffix}"
        root = self.config.state_dir / "sessions" / dirname
        return SessionPaths(root=root, workspace=root / "workspace", logs=root / "logs", exports=root / "exports")

    def paths_from_workspace(self, workspace: Path) -> SessionPaths:
        root = workspace.parent if workspace.name == "workspace" else workspace
        return SessionPaths(root=root, workspace=workspace, logs=root / "logs", exports=root / "exports")

    def _ensure_paths(self, paths: SessionPaths) -> None:
        paths.workspace.mkdir(parents=True, exist_ok=True)
        paths.logs.mkdir(parents=True, exist_ok=True)
        paths.exports.mkdir(parents=True, exist_ok=True)


def _safe_suffix(title: str) -> str:
    cleaned = _SAFE_TITLE_RE.sub("-", title.strip())[:48].strip("-._")
    return cleaned.lower()

