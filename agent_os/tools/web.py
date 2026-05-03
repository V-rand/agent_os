from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any

from agent_os.types import ToolResult
from .context import ToolContext
from .registry import function_schema, registry


def _web_available() -> bool:
    return os.getenv("AGENT_OS_ENABLE_WEB_TOOLS", "").strip().lower() in {"1", "true", "yes", "on"}


def web_read(*, context: ToolContext, url: str, mode: str = "distilled") -> ToolResult:
    if not context.config.web_tools_enabled and not _web_available():
        return ToolResult.fail("web tools are disabled; set web_tools_enabled or AGENT_OS_ENABLE_WEB_TOOLS=1")
    raw = _fetch_url(url, timeout=context.config.web_request_timeout_seconds)
    text = _html_to_text(raw)
    if mode == "raw":
        result = ToolResult.ok(text)
        result.summary = _distill_text(context, text, source=f"URL: {url}")
    elif mode == "distilled":
        distilled = _distill_text(context, text, source=f"URL: {url}")
        result = ToolResult.ok(distilled)
        result.summary = distilled
        result.artifacts = [{"type": "raw_web_content", "url": url, "bytes": len(raw.encode("utf-8"))}]
    else:
        return ToolResult.fail("mode must be 'distilled' or 'raw'")
    result.metadata = {"url": url, "mode": mode, "raw_chars": len(text), "truncated": False}
    return result


def web_search(*, context: ToolContext, query: str, limit: int = 5, mode: str = "summary") -> ToolResult:
    if not context.config.web_tools_enabled and not _web_available():
        return ToolResult.fail("web tools are disabled; set web_tools_enabled or AGENT_OS_ENABLE_WEB_TOOLS=1")
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return ToolResult.fail("web_search requires TAVILY_API_KEY for now; web_read can fetch a known URL")
    payload = json.dumps({
        "query": query,
        "max_results": max(1, min(int(limit), 10)),
        "include_answer": mode == "summary",
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {tavily_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=context.config.web_request_timeout_seconds) as response:
        data = json.loads(response.read().decode("utf-8", errors="replace"))
    results = []
    for item in data.get("results", [])[:max(1, min(int(limit), 10))]:
        results.append({
            "title": item.get("title"),
            "url": item.get("url"),
            "content": item.get("content"),
            "score": item.get("score"),
        })
    distilled = _distill_search_results(context, query, data.get("answer"), results)
    content = json.dumps({"answer": data.get("answer"), "results": results, "distilled": distilled}, ensure_ascii=False, sort_keys=True)
    result = ToolResult.ok(content, data={"answer": data.get("answer"), "results": results})
    result.summary = distilled
    result.metadata = {"query": query, "mode": mode, "provider": "tavily", "truncated": False}
    return result


def _fetch_url(url: str, *, timeout: float) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("url must use http or https")
    request = urllib.request.Request(url, headers={"User-Agent": "agent-os/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("content-type", "")
        charset = "utf-8"
        match = re.search(r"charset=([^;]+)", content_type, flags=re.I)
        if match:
            charset = match.group(1).strip()
        return response.read().decode(charset, errors="replace")


def _html_to_text(raw: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _distill_text(context: ToolContext, text: str, *, source: str) -> str:
    if context.model_client is not None and len(text) > 1200:
        try:
            response = context.model_client.complete(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a web content distillation subagent. Extract only facts useful to the caller. "
                            "Do not answer unrelated questions. Preserve source attribution, dates, names, numbers, and uncertainty. "
                            "Return concise markdown with sections: Key Facts, Caveats, Source."
                        ),
                    },
                    {"role": "user", "content": f"{source}\n\nRaw extracted text:\n{text}"},
                ],
                tools=[],
            )
            distilled = (response.content or "").strip()
            if distilled:
                return distilled
        except Exception:
            pass
    sentences = re.split(r"(?<=[。.!?])\s+", text)
    selected = [item.strip() for item in sentences if item.strip()][:12]
    return "\n".join(f"- {item}" for item in selected) if selected else text[:2000]


def _distill_search_results(context: ToolContext, query: str, answer: Any, results: list[dict[str, Any]]) -> str:
    payload = json.dumps({"query": query, "answer": answer, "results": results}, ensure_ascii=False, sort_keys=True)
    if context.model_client is not None and len(payload) > 1200:
        try:
            response = context.model_client.complete(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a search-result distillation subagent. Summarize search results for an agent. "
                            "Preserve URLs, distinguish snippets from verified facts, and highlight which result to read next."
                        ),
                    },
                    {"role": "user", "content": payload},
                ],
                tools=[],
            )
            distilled = (response.content or "").strip()
            if distilled:
                return distilled
        except Exception:
            pass
    if answer:
        return str(answer)
    lines = [f"{idx}. {item.get('title') or '(untitled)'} - {item.get('url')}" for idx, item in enumerate(results, start=1)]
    return "\n".join(lines) if lines else f"No search results for {query!r}."


registry.register(
    name="web_read",
    toolset="web",
    schema=function_schema("web_read", "Read a URL. Default mode returns a distilled summary; mode='raw' returns full extracted text.", {
        "url": {"type": "string", "description": "HTTP or HTTPS URL to read."},
        "mode": {"type": "string", "enum": ["distilled", "raw"], "description": "Use distilled for context economy, raw for full text.", "default": "distilled"},
    }, ["url"]),
    handler=web_read,
    check_fn=_web_available,
)
registry.register(
    name="web_search",
    toolset="web",
    schema=function_schema("web_search", "Search the web and return summarized results. Requires TAVILY_API_KEY.", {
        "query": {"type": "string", "description": "Search query."},
        "limit": {"type": "integer", "description": "Maximum results, 1-10.", "default": 5},
        "mode": {"type": "string", "enum": ["summary"], "description": "Search result mode.", "default": "summary"},
    }, ["query"]),
    handler=web_search,
    check_fn=_web_available,
)
