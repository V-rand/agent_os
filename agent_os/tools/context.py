from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_os.config import AgentOSConfig
from agent_os.memory import MemoryStore
from agent_os.skills import SkillManager
from agent_os.storage import SQLiteStore


@dataclass(slots=True)
class ToolContext:
    config: AgentOSConfig
    store: SQLiteStore
    memory: MemoryStore
    skills: SkillManager
    session_id: str
    run_id: str
    model_client: Any = None
    depth: int = 0
