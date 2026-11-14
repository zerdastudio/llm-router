"""Shared test helpers: a clock that moves only when told, and a small router factory."""
from __future__ import annotations

import asyncio

from llm_router import CircuitBreaker, FakeTransport, KeyPool, ModelSpec, Response, RoleBudget, Router, Usage


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds

    async def sleep(self, seconds: float) -> None:
        self.t += seconds


def run(coro):
    return asyncio.run(coro)


OK = Usage(input=1000, output=100)


def ok(text: str = "fine") -> Response:
    return Response(text, OK)


def make_router(script: dict, *, chain=("a", "b"), fallback=(), cap=1.0, window=3600.0, clock=None,
                threshold=3, max_chain=5, keys=1, rpm=None, on_attempt=None, **spec_overrides):
    clock = clock or FakeClock()
    models = {}
    for key in set(chain) | set(fallback):
        models[key] = ModelSpec(key, f"p-{key}", f"model-{key}", input_per_m=1.0, output_per_m=10.0,
                                **spec_overrides.get(key, {}))
    transport = FakeTransport(script)
    pools = {
        m.provider: KeyPool(m.provider, [(f"{m.provider}-{i}", f"secret-{i}") for i in range(keys)], rpm=rpm,
                            clock=clock, sleep=clock.sleep)
        for m in models.values()
    }
    router = Router(
        models=models, chains={"r": list(chain)}, budgets={"r": RoleBudget(cap, window, clock=clock)},
        pools=pools, transports={m.provider: transport for m in models.values()}, fallback=fallback,
        max_chain=max_chain, breaker=CircuitBreaker(threshold, 60, clock=clock), on_attempt=on_attempt,
        clock=clock,
    )
    return router, transport, clock
