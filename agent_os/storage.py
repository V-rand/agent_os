from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class SQLiteStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._initialize()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _transaction(self):
        with self._lock:
            try:
                yield
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def _initialize(self) -> None:
        with self._transaction():
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    parent_session_id TEXT,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL,
                    workspace_root TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    ended_at TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    tool_call_id TEXT,
                    tool_calls_json TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    tool_call_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    result TEXT NOT NULL,
                    error TEXT,
                    latency_seconds REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    parent_session_id TEXT,
                    content TEXT NOT NULL,
                    coverage_end_message_id INTEGER,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS todos (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT '',
                    stage TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    source_paths_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(session_id, path),
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS materials (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT '',
                    stage TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(session_id, path),
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
                CREATE INDEX IF NOT EXISTS idx_tool_calls_run ON tool_calls(run_id, id);
                CREATE INDEX IF NOT EXISTS idx_run_events_run ON run_events(run_id, id);
                CREATE INDEX IF NOT EXISTS idx_todos_session ON todos(session_id, status, updated_at);
                CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id, stage, kind, path);
                CREATE INDEX IF NOT EXISTS idx_materials_session ON materials(session_id, stage, kind, path);
                """
            )
            self._migrate_optional_columns()

    def _migrate_optional_columns(self) -> None:
        session_columns = self._column_names("sessions")
        if "stage" not in session_columns:
            self._conn.execute("ALTER TABLE sessions ADD COLUMN stage TEXT NOT NULL DEFAULT ''")

    def create_session(self, *, session_id: str | None = None, title: str = "New session",
                       workspace_root: str, parent_session_id: str | None = None,
                       stage: str = "", metadata: dict[str, Any] | None = None) -> str:
        sid = session_id or str(uuid.uuid4())
        now = utcnow_iso()
        with self._transaction():
            self._conn.execute(
                """
                INSERT INTO sessions (id, parent_session_id, status, stage, title, workspace_root, created_at, updated_at, metadata_json)
                VALUES (?, ?, 'active', ?, ?, ?, ?, ?, ?)
                """,
                (sid, parent_session_id, stage, title, workspace_root, now, now, json.dumps(metadata or {}, ensure_ascii=False)),
            )
        return sid

    def ensure_session(self, session_id: str | None, *, title: str, workspace_root: str,
                       stage: str = "", metadata: dict[str, Any] | None = None) -> str:
        if session_id and self.get_session(session_id):
            return session_id
        return self.create_session(session_id=session_id, title=title, workspace_root=workspace_root, stage=stage, metadata=metadata)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return self._session_from_row(row) if row else None

    def list_sessions(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._session_from_row(row) for row in rows]

    def update_session_stage(self, session_id: str, stage: str) -> None:
        now = utcnow_iso()
        with self._transaction():
            self._conn.execute("UPDATE sessions SET stage = ?, updated_at = ? WHERE id = ?", (stage, now, session_id))

    def list_child_sessions(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE parent_session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
        return [self._session_from_row(row) for row in rows]

    def add_message(self, session_id: str, role: str, content: str | None,
                    *, tool_call_id: str | None = None, tool_calls: list[dict[str, Any]] | None = None,
                    metadata: dict[str, Any] | None = None) -> int:
        now = utcnow_iso()
        with self._transaction():
            cur = self._conn.execute(
                """
                INSERT INTO messages (session_id, role, content, tool_call_id, tool_calls_json, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    tool_call_id,
                    json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                ),
            )
            self._conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
            return int(cur.lastrowid)

    def list_messages(self, session_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC"
        params: tuple[Any, ...] = (session_id,)
        if limit is not None:
            sql = "SELECT * FROM (SELECT * FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?) ORDER BY id ASC"
            params = (session_id, limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._message_from_row(row) for row in rows]

    def latest_summary(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM conversation_summaries WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return self._summary_from_row(row) if row else None

    def list_summaries(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM conversation_summaries WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
        return [self._summary_from_row(row) for row in rows]

    def save_summary(self, session_id: str, content: str, *, coverage_end_message_id: int | None = None,
                     parent_session_id: str | None = None, metadata: dict[str, Any] | None = None) -> str:
        summary_id = str(uuid.uuid4())
        with self._transaction():
            self._conn.execute(
                """
                INSERT INTO conversation_summaries
                    (id, session_id, parent_session_id, content, coverage_end_message_id, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary_id,
                    session_id,
                    parent_session_id,
                    content,
                    coverage_end_message_id,
                    utcnow_iso(),
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
        return summary_id

    def add_run_event(self, run_id: str, session_id: str, event_type: str, message: str,
                      payload: dict[str, Any] | None = None) -> None:
        with self._transaction():
            self._conn.execute(
                """
                INSERT INTO run_events (run_id, session_id, event_type, message, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, session_id, event_type, message, json.dumps(payload or {}, ensure_ascii=False), utcnow_iso()),
            )

    def get_run_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM run_events WHERE run_id = ? ORDER BY id ASC",
                (run_id,),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def get_session_events(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM run_events WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def add_tool_call(self, *, run_id: str, session_id: str, tool_call_id: str, name: str,
                      arguments: dict[str, Any], success: bool, result: str,
                      error: str | None, latency_seconds: float) -> None:
        with self._transaction():
            self._conn.execute(
                """
                INSERT INTO tool_calls
                    (run_id, session_id, tool_call_id, name, arguments_json, success, result, error, latency_seconds, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    session_id,
                    tool_call_id,
                    name,
                    json.dumps(arguments, ensure_ascii=False),
                    1 if success else 0,
                    result,
                    error,
                    latency_seconds,
                    utcnow_iso(),
                ),
            )

    def get_tool_calls(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tool_calls WHERE run_id = ? ORDER BY id ASC",
                (run_id,),
            ).fetchall()
        return [self._tool_call_from_row(row) for row in rows]

    def get_session_snapshot(self, session_id: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(f"session not found: {session_id}")
        return {
            "session": session,
            "messages": self.list_messages(session_id),
            "todos": self.list_todos(session_id),
            "children": self.list_child_sessions(session_id),
            "summaries": self.list_summaries(session_id),
            "artifacts": self.list_artifacts(session_id),
            "materials": self.list_materials(session_id),
        }

    def list_todos(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM todos WHERE session_id = ? ORDER BY created_at ASC", (session_id,)).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            out.append(item)
        return out

    def upsert_todos(self, session_id: str, items: Iterable[dict[str, Any]]) -> None:
        now = utcnow_iso()
        with self._transaction():
            for item in items:
                todo_id = str(item.get("id") or uuid.uuid4())
                metadata = dict(item.get("metadata") or {})
                if item.get("stage") is not None:
                    metadata["stage"] = str(item["stage"])
                self._conn.execute(
                    """
                    INSERT INTO todos (id, session_id, text, status, created_at, updated_at, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        text = excluded.text,
                        status = excluded.status,
                        updated_at = excluded.updated_at,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        todo_id,
                        session_id,
                        str(item["text"]),
                        str(item.get("status") or "pending"),
                        now,
                        now,
                        json.dumps(metadata, ensure_ascii=False),
                    ),
                )

    def upsert_artifact(self, session_id: str, path: str, *, kind: str = "",
                        stage: str = "", source_paths: list[str] | None = None,
                        metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        now = utcnow_iso()
        artifact_id = str(uuid.uuid4())
        source_paths_json = json.dumps(source_paths or [], ensure_ascii=False)
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self._transaction():
            cur = self._conn.execute(
                """
                INSERT INTO artifacts
                    (id, session_id, path, kind, stage, metadata_json, source_paths_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, path) DO UPDATE SET
                    kind = excluded.kind,
                    stage = excluded.stage,
                    metadata_json = excluded.metadata_json,
                    source_paths_json = excluded.source_paths_json,
                    updated_at = excluded.updated_at
                RETURNING *
                """,
                (artifact_id, session_id, path, kind, stage, metadata_json, source_paths_json, now, now),
            )
            row = cur.fetchone()
            self._conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
        return self._artifact_from_row(row)

    def read_artifact(self, session_id: str, path: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM artifacts WHERE session_id = ? AND path = ?",
                (session_id, path),
            ).fetchone()
        return self._artifact_from_row(row) if row else None

    def list_artifacts(self, session_id: str, *, stage: str | None = None,
                       kind: str | None = None) -> list[dict[str, Any]]:
        clauses = ["session_id = ?"]
        params: list[Any] = [session_id]
        if stage is not None:
            clauses.append("stage = ?")
            params.append(stage)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        sql = f"SELECT * FROM artifacts WHERE {' AND '.join(clauses)} ORDER BY path ASC"
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [self._artifact_from_row(row) for row in rows]

    def search_artifacts(self, session_id: str, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        pattern = f"%{query}%"
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM artifacts
                WHERE session_id = ? AND path LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (session_id, pattern, limit),
            ).fetchall()
        return [self._artifact_from_row(row) for row in rows]

    def upsert_material(self, session_id: str, path: str, *, kind: str = "", stage: str = "",
                        title: str = "", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        now = utcnow_iso()
        material_id = str(uuid.uuid4())
        with self._transaction():
            cur = self._conn.execute(
                """
                INSERT INTO materials
                    (id, session_id, path, kind, stage, title, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, path) DO UPDATE SET
                    kind = excluded.kind,
                    stage = excluded.stage,
                    title = excluded.title,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                RETURNING *
                """,
                (material_id, session_id, path, kind, stage, title, json.dumps(metadata or {}, ensure_ascii=False), now, now),
            )
            row = cur.fetchone()
            self._conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
        return self._material_from_row(row)

    def list_materials(self, session_id: str, *, stage: str | None = None,
                       kind: str | None = None) -> list[dict[str, Any]]:
        clauses = ["session_id = ?"]
        params: list[Any] = [session_id]
        if stage is not None:
            clauses.append("stage = ?")
            params.append(stage)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        sql = f"SELECT * FROM materials WHERE {' AND '.join(clauses)} ORDER BY stage ASC, path ASC"
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [self._material_from_row(row) for row in rows]

    @staticmethod
    def _message_from_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["metadata"] = json.loads(data.pop("metadata_json") or "{}")
        tool_calls_json = data.pop("tool_calls_json")
        data["tool_calls"] = json.loads(tool_calls_json) if tool_calls_json else None
        return data

    @staticmethod
    def _session_from_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["metadata"] = json.loads(data.pop("metadata_json") or "{}")
        return data

    @staticmethod
    def _summary_from_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["metadata"] = json.loads(data.pop("metadata_json") or "{}")
        return data

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["type"] = data.pop("event_type")
        data["payload"] = json.loads(data.pop("payload_json") or "{}")
        return data

    @staticmethod
    def _tool_call_from_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["arguments"] = json.loads(data.pop("arguments_json") or "{}")
        data["success"] = bool(data["success"])
        return data

    @staticmethod
    def _artifact_from_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data.pop("content", None)
        data["metadata"] = json.loads(data.pop("metadata_json") or "{}")
        data["source_paths"] = json.loads(data.pop("source_paths_json") or "[]")
        return data

    @staticmethod
    def _material_from_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["metadata"] = json.loads(data.pop("metadata_json") or "{}")
        return data

    def _column_names(self, table_name: str) -> set[str]:
        rows = self._conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {str(row["name"]) for row in rows}
