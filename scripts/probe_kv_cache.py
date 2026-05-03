from __future__ import annotations

import json

from agent_os.config import AgentOSConfig
from agent_os.model_client import OpenAIChatClient


def main() -> int:
    config = AgentOSConfig.load()
    client = OpenAIChatClient(config)
    # Bailian implicit cache needs at least 256 prompt tokens. Keep a long,
    # identical prefix and vary only the final user request if you edit this.
    stable_prefix = "Agent OS stable cache probe prefix. " * 500
    messages = [
        {"role": "system", "content": f"You are a cache probe. Return exactly: ok\n\n{stable_prefix}"},
        {"role": "user", "content": "Probe stable prefix caching. Return exactly: ok"},
    ]
    first = client.complete(messages=messages, tools=[])
    second = client.complete(messages=messages, tools=[])
    print(json.dumps({
        "model": config.model,
        "first": _usage_payload(first.usage),
        "second": _usage_payload(second.usage),
        "cache_hit_rate_known": second.usage.cache_hit_rate_known,
        "second_cache_hit_rate": second.usage.cache_hit_rate,
    }, ensure_ascii=False, indent=2))
    return 0


def _usage_payload(usage) -> dict:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "cached_prompt_tokens": usage.cached_prompt_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
        "cache_hit_rate_known": usage.cache_hit_rate_known,
        "cache_hit_rate": usage.cache_hit_rate,
    }


if __name__ == "__main__":
    raise SystemExit(main())
