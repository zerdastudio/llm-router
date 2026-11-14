"""Local limits: per-key rate buckets, key cooldowns, and a circuit breaker per model.

All clocks are injectable, so the behaviour over time is tested without waiting.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]


class TokenBucket:
    """`capacity` units per `per_seconds`, refilled continuously (not reset at a fixed hour)."""

    def __init__(self, capacity: float, per_seconds: float, clock: Clock):
        if not (capacity > 0 and per_seconds > 0):
            raise ValueError("a bucket needs a positive capacity and period")
        self.capacity = float(capacity)
        self.rate = capacity / per_seconds
        self.level = float(capacity)
        self._clock = clock
        self._last = clock()

    def _refill(self) -> None:
        now = self._clock()
        self.level = min(self.capacity, self.level + (now - self._last) * self.rate)
        self._last = now

    def wait_for(self, amount: float) -> float:
        """Seconds until `amount` is available: 0 now, inf if it never fits."""
        self._refill()
        if amount > self.capacity:
            return math.inf
        if self.level >= amount:
            return 0.0
        return (amount - self.level) / self.rate

    def take(self, amount: float) -> None:
        self._refill()
        self.level -= amount


@dataclass
class Credential:
    name: str
    secret: str = field(repr=False)
    requests_minute: TokenBucket | None = None
    tokens_minute: TokenBucket | None = None
    requests_day: TokenBucket | None = None
    cooldown_until: float = 0.0

    def _needs(self, tokens: int):
        return [(b, n) for b, n in ((self.requests_minute, 1), (self.tokens_minute, tokens), (self.requests_day, 1))
                if b is not None]

    def wait_for(self, tokens: int) -> float:
        return max((b.wait_for(n) for b, n in self._needs(tokens)), default=0.0)

    def take(self, tokens: int) -> None:
        for b, n in self._needs(tokens):
            b.take(n)


class KeyPool:
    """The API keys of one provider, used in turn, each with its own limits.

    `acquire` reserves the estimated tokens before the call. When no key is free now, it waits for the earliest
    refill if that comes within `max_wait` seconds; it never waits out a cooldown after an error, and it gives up
    (returns None) instead of waiting longer.
    """

    def __init__(self, provider: str, keys: list[tuple[str, str]], *, rpm: int | None = None,
                 tpm: int | None = None, rpd: int | None = None, max_wait: float = 10.0,
                 clock: Clock = time.monotonic, sleep: Sleep = asyncio.sleep):
        if not keys:
            raise ValueError(f"provider {provider!r} has no keys")
        self.provider = provider
        self.max_wait = max_wait
        self._clock = clock
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._next = 0
        self.credentials = [
            Credential(
                name=name,
                secret=secret,
                requests_minute=TokenBucket(rpm, 60, clock) if rpm else None,
                tokens_minute=TokenBucket(tpm, 60, clock) if tpm else None,
                requests_day=TokenBucket(rpd, 86400, clock) if rpd else None,
            )
            for name, secret in keys
        ]

    async def acquire(self, tokens: int) -> Credential | None:
        waited = 0.0
        while True:
            async with self._lock:
                now = self._clock()
                soonest = math.inf
                count = len(self.credentials)
                for i in range(count):
                    index = (self._next + i) % count
                    cred = self.credentials[index]
                    if cred.cooldown_until > now:
                        continue
                    wait = cred.wait_for(tokens)
                    if wait == 0:
                        cred.take(tokens)
                        self._next = (index + 1) % count
                        return cred
                    soonest = min(soonest, wait)
            if soonest == math.inf or waited + soonest > self.max_wait:
                return None
            # Sleep outside the lock, so other requests can use keys that free up meanwhile.
            await self._sleep(soonest)
            waited += soonest

    def cooldown(self, cred: Credential, seconds: float) -> None:
        cred.cooldown_until = max(cred.cooldown_until, self._clock() + max(seconds, 0.0))


CLOSED, PROBE = "closed", "probe"


class CircuitBreaker:
    """Skips a model after `threshold` failures in a row, for `cooldown` seconds.

    When the cooldown is over the model gets one more chance, and only one: a single request may probe it while
    the others keep skipping it. A success closes the breaker, a failure opens it again straight away. A probe
    that ends without a verdict (the request was answered earlier in the chain, or cancelled) is handed back
    with `done`; one that is never handed back expires after another cooldown.
    """

    def __init__(self, threshold: int = 3, cooldown: float = 60.0, clock: Clock = time.monotonic):
        if threshold < 1:
            raise ValueError("threshold must be at least 1")
        self.threshold = threshold
        self.cooldown = cooldown
        self._clock = clock
        self._failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}
        self._probing: dict[str, float] = {}

    def admit(self, key: str) -> str | None:
        """CLOSED: use it. PROBE: use it, as the one trial after a cooldown. None: skip it."""
        if self._failures.get(key, 0) < self.threshold:
            return CLOSED
        now = self._clock()
        if now - self._opened_at[key] < self.cooldown:
            return None
        started = self._probing.get(key)
        if started is not None and now - started < self.cooldown:
            return None
        self._probing[key] = now
        return PROBE

    def allows(self, key: str) -> bool:
        return self.admit(key) is not None

    def done(self, key: str) -> None:
        self._probing.pop(key, None)

    def failure(self, key: str) -> None:
        self._probing.pop(key, None)
        self._failures[key] = self._failures.get(key, 0) + 1
        if self._failures[key] >= self.threshold:
            self._opened_at[key] = self._clock()

    def success(self, key: str) -> None:
        self._probing.pop(key, None)
        self._failures.pop(key, None)
        self._opened_at.pop(key, None)

    def snapshot(self) -> dict[str, dict]:
        """Current state only, not a history: a model that recovered disappears from it."""
        return {key: {"failures": n, "open": key in self._opened_at} for key, n in self._failures.items()}
