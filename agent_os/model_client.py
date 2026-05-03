from __future__ import annotations

import time
from typing import Any, Protocol

from .config import AgentOSConfig
from .types import ModelResponse, Usage


class ModelClient(Protocol):
    def complete(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        ...


class ModelClientError(RuntimeError):
    def __init__(self, category: str, message: str, raw: Any = None):
        super().__init__(message)
        self.category = category
        self.raw = raw
        self.message = message

    def to_payload(self) -> dict[str, Any]:
        return {"category": self.category, "message": self.message}


class OpenAIChatClient:
    def __init__(self, config: AgentOSConfig):
        if not config.api_key:
            raise ValueError("Missing API key. Set OPENAI_API_KEY, DASHSCOPE_API_KEY, or api_key in config.")
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=config.request_timeout_seconds)

    def complete(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        last_error: ModelClientError | None = None
        for attempt in range(self.config.model_max_retries + 1):
            try:
                return self._complete_once(messages=messages, tools=tools)
            except Exception as exc:
                error = exc if isinstance(exc, ModelClientError) else _classify_exception(exc)
                if not isinstance(error, ModelClientError):
                    error = _classify_exception(error)
                last_error = error
                if attempt >= self.config.model_max_retries or error.category in {"auth", "bad_request"}:
                    raise error
                time.sleep(self.config.retry_backoff_seconds * (attempt + 1))
        raise last_error or ModelClientError("unknown", "model request failed")

    def _complete_once(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            kwargs["parallel_tool_calls"] = True
        if self.config.extra_body:
            kwargs["extra_body"] = self.config.extra_body
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise _classify_exception(exc) from exc
        choice = response.choices[0]
        message = choice.message
        tool_calls = []
        for call in getattr(message, "tool_calls", None) or []:
            tool_calls.append({
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            })
        usage = Usage()
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            cached_tokens, cache_known = _extract_cached_prompt_tokens(raw_usage)
            cache_creation_tokens = _extract_cache_creation_input_tokens(raw_usage)
            usage = Usage(
                prompt_tokens=int(_get_usage_value(raw_usage, "prompt_tokens") or 0),
                completion_tokens=int(_get_usage_value(raw_usage, "completion_tokens") or 0),
                total_tokens=int(_get_usage_value(raw_usage, "total_tokens") or 0),
                cached_prompt_tokens=cached_tokens,
                cache_creation_input_tokens=cache_creation_tokens,
                cache_hit_rate_known=cache_known,
            )
        return ModelResponse(content=getattr(message, "content", None), tool_calls=tool_calls, usage=usage, raw=response)


def _classify_exception(exc: Exception) -> ModelClientError:
    name = type(exc).__name__.lower()
    text = str(exc)
    if "auth" in name or "authentication" in name or "unauthorized" in text.lower() or "401" in text:
        category = "auth"
    elif "rate" in name or "429" in text:
        category = "rate_limit"
    elif "timeout" in name or "timed out" in text.lower():
        category = "timeout"
    elif "badrequest" in name or "bad_request" in name or "400" in text:
        category = "bad_request"
    elif "connection" in name or "connect" in text.lower() or "network" in text.lower():
        category = "connection"
    else:
        category = "unknown"
    return ModelClientError(category, f"{type(exc).__name__}: {exc}", raw=exc)


def _get_usage_value(raw_usage: Any, key: str) -> Any:
    if isinstance(raw_usage, dict):
        return raw_usage.get(key)
    return getattr(raw_usage, key, None)


def _get_nested_usage_value(raw_usage: Any, key: str) -> Any:
    current: Any = raw_usage
    for part in key.split("."):
        if current is None:
            return None
        current = current.get(part) if isinstance(current, dict) else getattr(current, part, None)
    return current


def _extract_cached_prompt_tokens(raw_usage: Any) -> tuple[int, bool]:
    candidate_keys = [
        "prompt_tokens_details.cached_tokens",
        "prompt_tokens_details.cached_prompt_tokens",
        "input_tokens_details.cached_tokens",
        "input_token_details.cache_read",
        "input_token_details.cached_tokens",
        "cache_read_input_tokens",
        "cached_prompt_tokens",
    ]
    for key in candidate_keys:
        value = _get_nested_usage_value(raw_usage, key) if "." in key else _get_usage_value(raw_usage, key)
        if value is not None:
            try:
                return int(value), True
            except (TypeError, ValueError):
                continue
    return 0, False


def _extract_cache_creation_input_tokens(raw_usage: Any) -> int:
    candidate_keys = [
        "prompt_tokens_details.cache_creation_input_tokens",
        "input_tokens_details.cache_creation_input_tokens",
        "cache_creation_input_tokens",
    ]
    for key in candidate_keys:
        value = _get_nested_usage_value(raw_usage, key) if "." in key else _get_usage_value(raw_usage, key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return 0
