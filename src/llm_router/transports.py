"""Transports: the only code that talks to a provider.

A transport is any async callable `(spec, api_key, messages, max_output_tokens) -> Response`. It raises
`TransportError` for an HTTP error, and lets anything else propagate: the router counts both as a failure of
that model and moves on.

The SDKs are imported only when a real transport is used, so the router, the tests and the demo need neither.
Their own automatic retries are switched off: retrying is the router's job, and every try shows in the ledger.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from .base import ModelSpec, Response, TransportError
from .pricing import reported_cost, usage_from_anthropic, usage_from_openai


class Transport(Protocol):
    def __call__(self, spec: ModelSpec, api_key: str, messages: list[dict],
                 max_output_tokens: int) -> Awaitable[Response]: ...


def _as_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    return dump() if callable(dump) else {}


def _retry_after(error: Exception) -> float | None:
    try:
        value = error.response.headers.get("retry-after")  # type: ignore[attr-defined]
        return float(value) if value is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def _status_error(error: Exception) -> TransportError | None:
    """An SDK error that carries an HTTP status, as a TransportError; None for anything else."""
    status = getattr(error, "status_code", None)
    if not isinstance(status, int):
        return None
    return TransportError(f"HTTP {status}: {type(error).__name__}", status=status, retry_after=_retry_after(error))


class OpenAICompatTransport:
    """OpenAI's chat completions API, and every provider that copies it (OpenRouter, Gemini, DeepSeek, ...).

    `max_tokens_param`: OpenAI's newer models want `max_completion_tokens`; most compatible providers take
    `max_tokens`. `use_reported_cost`: trust a `cost` field in the usage block (OpenRouter sends one).
    """

    def __init__(self, base_url: str | None = None, *, max_tokens_param: str = "max_tokens",
                 use_reported_cost: bool = False, client_factory: Callable[[str, str | None], Any] | None = None):
        self.base_url = base_url or None
        self.max_tokens_param = max_tokens_param
        self.use_reported_cost = use_reported_cost
        self._factory = client_factory
        self._clients: dict[str, Any] = {}

    def _client(self, api_key: str):
        if api_key not in self._clients:
            if self._factory is not None:
                self._clients[api_key] = self._factory(api_key, self.base_url)
            else:
                from openai import AsyncOpenAI
                self._clients[api_key] = AsyncOpenAI(api_key=api_key, base_url=self.base_url, max_retries=0)
        return self._clients[api_key]

    async def __call__(self, spec: ModelSpec, api_key: str, messages: list[dict], max_output_tokens: int) -> Response:
        client = self._client(api_key)
        try:
            reply = await client.chat.completions.create(
                model=spec.model, messages=messages, **{self.max_tokens_param: max_output_tokens})
        except Exception as e:
            mapped = _status_error(e)
            if mapped is None:
                raise
            raise mapped from e
        choices = getattr(reply, "choices", None) or []
        message = getattr(choices[0], "message", None) if choices else None
        text = getattr(message, "content", None) or ""
        if getattr(message, "refusal", None):
            finish = "refusal"
        elif choices and getattr(choices[0], "finish_reason", None) == "length":
            finish = "length"
        else:
            finish = "stop"
        usage = _as_dict(getattr(reply, "usage", None))
        cost = reported_cost(usage) if self.use_reported_cost else None
        return Response(text=text, usage=usage_from_openai(usage), finish=finish, reported_cost=cost)


class AnthropicTransport:
    """Anthropic's messages API. System messages are moved to the `system` parameter, as the API wants."""

    def __init__(self, *, client_factory: Callable[[str], Any] | None = None):
        self._factory = client_factory
        self._clients: dict[str, Any] = {}

    def _client(self, api_key: str):
        if api_key not in self._clients:
            if self._factory is not None:
                self._clients[api_key] = self._factory(api_key)
            else:
                from anthropic import AsyncAnthropic
                self._clients[api_key] = AsyncAnthropic(api_key=api_key, max_retries=0)
        return self._clients[api_key]

    async def __call__(self, spec: ModelSpec, api_key: str, messages: list[dict], max_output_tokens: int) -> Response:
        system = "\n\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
        rest = [m for m in messages if m.get("role") != "system"]
        kwargs = {"system": system} if system else {}
        try:
            reply = await self._client(api_key).messages.create(
                model=spec.model, messages=rest, max_tokens=max_output_tokens, **kwargs)
        except Exception as e:
            mapped = _status_error(e)
            if mapped is None:
                raise
            raise mapped from e
        blocks = getattr(reply, "content", None) or []
        text = "".join(getattr(b, "text", "") or "" for b in blocks if getattr(b, "type", None) == "text")
        stop = getattr(reply, "stop_reason", None)
        finish = {"max_tokens": "length", "refusal": "refusal"}.get(stop, "stop")
        return Response(text=text, usage=usage_from_anthropic(_as_dict(getattr(reply, "usage", None))), finish=finish)


@dataclass
class Delay:
    """A scripted step that waits `seconds`, then plays `then` (a Response or an exception)."""

    seconds: float
    then: Any = None


class FakeTransport:
    """A scripted provider for tests and demos: no network, no keys.

    `script` maps a model key to the list of things its calls return, in order: a `Response`, an exception to
    raise, or a `Delay`. The last item repeats once the others are used up.
    """

    def __init__(self, script: dict[str, list]):
        self.script = {key: list(steps) for key, steps in script.items()}
        self.calls: list[tuple[str, int]] = []

    async def __call__(self, spec: ModelSpec, api_key: str, messages: list[dict], max_output_tokens: int) -> Response:
        self.calls.append((spec.key, max_output_tokens))
        steps = self.script.get(spec.key)
        if not steps:
            raise AssertionError(f"no scripted answer for {spec.key!r}")
        step = steps.pop(0) if len(steps) > 1 else steps[0]
        if isinstance(step, Delay):
            await asyncio.sleep(step.seconds)
            step = step.then
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, Response):
            return step
        raise AssertionError(f"bad scripted step for {spec.key!r}: {step!r}")
