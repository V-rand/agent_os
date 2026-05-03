from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"{path} requires PyYAML, or use agent_os.json instead") from exc
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


@dataclass(slots=True)
class AgentOSConfig:
    model: str = "qwen-plus"
    api_key: str | None = None
    base_url: str | None = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    workspace_root: Path = field(default_factory=lambda: Path.cwd())
    state_dir: Path = field(default_factory=lambda: Path.cwd() / ".agent_os")
    enabled_toolsets: set[str] = field(default_factory=lambda: {"files", "shell", "memory", "skills", "todo", "delegate", "context", "artifacts", "materials"})
    disabled_toolsets: set[str] = field(default_factory=set)
    max_iterations: int = 16
    request_timeout_seconds: float = 120.0
    model_max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    shell_timeout_seconds: float = 30.0
    context_budget_tokens: int = 120_000
    compression_enabled: bool = True
    compression_model: str | None = None
    compression_trigger_ratio: float = 0.88
    protect_first_messages: int = 3
    protect_last_messages: int = 20
    protect_last_tokens: int = 24_000
    compression_max_summary_tokens: int = 4000
    compression_min_interval_messages: int = 8
    compression_min_savings_ratio: float = 0.10
    compression_create_continuation_session: bool = True
    subagents_enabled: bool = True
    max_subagent_depth: int = 1
    web_tools_enabled: bool = False
    web_request_timeout_seconds: float = 20.0
    identity: str = "You are Agent OS, a business-agnostic agent runtime for upper-layer workflows."
    extra_body: dict[str, Any] = field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        return self.state_dir / "state.db"

    @property
    def log_dir(self) -> Path:
        return self.state_dir / "logs"

    @property
    def memory_dir(self) -> Path:
        return self.state_dir / "memories"

    @property
    def skills_dir(self) -> Path:
        return self.state_dir / "skills"

    @classmethod
    def load(cls, config_path: str | Path | None = None, cwd: Path | None = None) -> "AgentOSConfig":
        root = cwd or Path.cwd()
        _load_dotenv(root / ".env")
        path = Path(config_path) if config_path else _first_existing(root)
        data = _load_config_file(path) if path else {}
        cfg = cls()
        cfg.model = str(data.get("model") or os.getenv("AGENT_OS_MODEL") or cfg.model)
        cfg.api_key = data.get("api_key") or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        cfg.base_url = data.get("base_url") or os.getenv("OPENAI_BASE_URL") or cfg.base_url
        cfg.workspace_root = Path(data.get("workspace_root") or os.getenv("AGENT_OS_WORKSPACE") or root).resolve()
        cfg.state_dir = Path(data.get("state_dir") or os.getenv("AGENT_OS_STATE_DIR") or root / ".agent_os").resolve()
        cfg.enabled_toolsets = set(data.get("enabled_toolsets") or cfg.enabled_toolsets)
        cfg.disabled_toolsets = set(data.get("disabled_toolsets") or [])
        cfg.max_iterations = int(data.get("max_iterations") or cfg.max_iterations)
        cfg.request_timeout_seconds = float(data.get("request_timeout_seconds") or cfg.request_timeout_seconds)
        cfg.model_max_retries = int(data.get("model_max_retries") or cfg.model_max_retries)
        cfg.retry_backoff_seconds = float(data.get("retry_backoff_seconds") or cfg.retry_backoff_seconds)
        cfg.shell_timeout_seconds = float(data.get("shell_timeout_seconds") or cfg.shell_timeout_seconds)
        cfg.context_budget_tokens = int(data.get("context_budget_tokens") or cfg.context_budget_tokens)
        cfg.compression_enabled = _bool(data.get("compression_enabled"), cfg.compression_enabled)
        cfg.compression_model = data.get("compression_model") or os.getenv("AGENT_OS_COMPRESSION_MODEL") or cfg.compression_model
        cfg.compression_trigger_ratio = float(data.get("compression_trigger_ratio") or cfg.compression_trigger_ratio)
        cfg.protect_first_messages = int(data.get("protect_first_messages") or cfg.protect_first_messages)
        cfg.protect_last_messages = int(data.get("protect_last_messages") or cfg.protect_last_messages)
        cfg.protect_last_tokens = int(data.get("protect_last_tokens") or cfg.protect_last_tokens)
        cfg.compression_max_summary_tokens = int(data.get("compression_max_summary_tokens") or cfg.compression_max_summary_tokens)
        cfg.compression_min_interval_messages = int(data.get("compression_min_interval_messages") or cfg.compression_min_interval_messages)
        cfg.compression_min_savings_ratio = float(data.get("compression_min_savings_ratio") or cfg.compression_min_savings_ratio)
        cfg.compression_create_continuation_session = _bool(
            data.get("compression_create_continuation_session"),
            cfg.compression_create_continuation_session,
        )
        cfg.subagents_enabled = _bool(data.get("subagents_enabled"), cfg.subagents_enabled)
        cfg.max_subagent_depth = int(data.get("max_subagent_depth") or cfg.max_subagent_depth)
        cfg.web_tools_enabled = _bool(
            data.get("web_tools_enabled") if "web_tools_enabled" in data else os.getenv("AGENT_OS_ENABLE_WEB_TOOLS"),
            cfg.web_tools_enabled,
        )
        cfg.web_request_timeout_seconds = float(data.get("web_request_timeout_seconds") or cfg.web_request_timeout_seconds)
        cfg.identity = str(data.get("identity") or cfg.identity)
        extra_body = data.get("extra_body") or {}
        if not isinstance(extra_body, dict):
            raise ValueError("extra_body must be a mapping")
        cfg.extra_body = extra_body
        return cfg


def _first_existing(root: Path) -> Path | None:
    for name in ("agent_os.yaml", "agent_os.yml", "agent_os.json"):
        path = root / name
        if path.exists():
            return path
    return None


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
