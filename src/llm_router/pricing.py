"""Turn provider usage into `Usage`, and `Usage` into dollars.

Prices are configuration, never built in: providers change them, and a price table in code goes stale silently.
"""
from __future__ import annotations

import math

from .base import ModelSpec, Response, Usage

_PER = 1_000_000


def price(spec: ModelSpec, usage: Usage) -> float:
    """Cost in USD of `usage` at the prices of `spec`."""
    cache_read = spec.input_per_m if spec.cache_read_per_m is None else spec.cache_read_per_m
    cache_write = spec.input_per_m if spec.cache_write_per_m is None else spec.cache_write_per_m
    reasoning = spec.output_per_m if spec.reasoning_per_m is None else spec.reasoning_per_m
    return (
        usage.input * spec.input_per_m
        + usage.cache_read * cache_read
        + usage.cache_write * cache_write
        + usage.output * spec.output_per_m
        + usage.reasoning * reasoning
    ) / _PER


def cost_of(spec: ModelSpec, response: Response) -> float:
    """The provider's own reported cost when it gave a valid one, otherwise table pricing."""
    reported = response.reported_cost
    if (isinstance(reported, (int, float)) and not isinstance(reported, bool) and math.isfinite(reported)
            and reported >= 0):
        return float(reported)
    return price(spec, response.usage)


def worst_case(spec: ModelSpec, messages: list[dict], room: int) -> float:
    """The most one call can cost: the input, estimated generously (three characters a token) at the highest
    input price, plus the whole output room at the highest output price."""
    chars = sum(len(str(m.get("content", ""))) for m in messages)
    input_tokens = chars // 3 + 8 * len(messages) + 8
    input_rate = max(spec.input_per_m, spec.cache_write_per_m or 0.0)
    output_rate = max(spec.output_per_m, spec.reasoning_per_m or 0.0)
    return (input_tokens * input_rate + room * output_rate) / _PER


def _int(value) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def usage_from_openai(u: dict | None) -> Usage:
    """Usage of an OpenAI-compatible chat completion (OpenAI, OpenRouter, Gemini's compatible endpoint, ...).

    `completion_tokens` includes the reasoning tokens when the provider reports them in
    `completion_tokens_details`. Some compatible endpoints report thinking only in `total_tokens`: then the excess
    over prompt + completion is counted as reasoning, and the visible output is left as it is.
    """
    if not u:
        return Usage()
    prompt = _int(u.get("prompt_tokens"))
    completion = _int(u.get("completion_tokens"))
    total = _int(u.get("total_tokens"))
    cached = min(_int((u.get("prompt_tokens_details") or {}).get("cached_tokens")), prompt)
    reasoning = _int((u.get("completion_tokens_details") or {}).get("reasoning_tokens"))
    if reasoning:
        visible = max(completion - reasoning, 0)
    else:
        visible = completion
        if total > prompt + completion:
            reasoning = total - prompt - completion
    return Usage(input=prompt - cached, cache_read=cached, output=visible, reasoning=reasoning)


def usage_from_anthropic(u: dict | None) -> Usage:
    """Usage of an Anthropic message: input is already net of the cache."""
    if not u:
        return Usage()
    return Usage(
        input=_int(u.get("input_tokens")),
        cache_read=_int(u.get("cache_read_input_tokens")),
        cache_write=_int(u.get("cache_creation_input_tokens")),
        output=_int(u.get("output_tokens")),
    )


def reported_cost(u: dict | None) -> float | None:
    """A cost the provider put in the usage block (OpenRouter does), or None when absent or not a valid number."""
    if not u or "cost" not in u:
        return None
    try:
        value = float(u["cost"])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None
