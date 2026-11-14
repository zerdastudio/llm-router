"""The router: one request, one role, a chain of models tried in order."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Iterable

from .base import (Attempt, BudgetExhausted, ChainExhausted, ModelSpec, Response, Result, TransportError,
                   UnknownRole, Usage)
from .budget import RoleBudget
from .limits import PROBE, CircuitBreaker, Clock, KeyPool
from .pricing import cost_of, worst_case
from .transports import Transport

log = logging.getLogger("llm_router")


def cooldown_for(status: int | None) -> float | None:
    """Seconds a key rests after an HTTP error: rate limits and server errors (but not 501, which will not get
    better by waiting). None: the key does not rest."""
    if status == 429:
        return 30.0
    if status is not None and 500 <= status <= 599 and status != 501:
        return 10.0
    return None


def estimate_tokens(messages: list[dict], room: int) -> int:
    """A rough count for the rate limiter: about four characters a token, plus the output room."""
    chars = sum(len(str(m.get("content", ""))) for m in messages)
    return chars // 4 + 4 * len(messages) + room


def _describe(error: BaseException) -> str:
    """A short text for an error, even one whose own str() raises."""
    try:
        text = str(error)
    except Exception:
        text = "<unprintable>"
    return f"{type(error).__name__}: {text}"[:300]


class _RaisedTimeout(Exception):
    """A TimeoutError raised by the transport itself, kept apart from the router's own timeout."""

    def __init__(self, original: BaseException):
        super().__init__(_describe(original))


async def _call(transport: Transport, spec: ModelSpec, secret: str, messages: list[dict], room: int):
    try:
        return await transport(spec, secret, messages, room)
    except TimeoutError as e:
        raise _RaisedTimeout(e) from e


class Router:
    """Sends a request to the first model of a role's chain that gives a usable answer.

    A model is skipped when it fails in transit (error or timeout), when it answers with no text, or when it
    refuses; a text cut short moves on too, while another model remains. Every call reserves its worst-case cost
    against the role's cap before it is sent; every priced response is charged, the discarded ones included.
    """

    def __init__(
        self,
        *,
        models: dict[str, ModelSpec],
        chains: dict[str, list[str]],
        budgets: dict[str, RoleBudget],
        pools: dict[str, KeyPool],
        transports: dict[str, Transport],
        fallback: Iterable[str] = (),
        max_chain: int = 5,
        breaker: CircuitBreaker | None = None,
        on_attempt: Callable[[Attempt], None] | None = None,
        clock: Clock = time.monotonic,
    ):
        fallback = list(fallback)
        for role, chain in chains.items():
            if role not in budgets:
                raise ValueError(f"role {role!r} has no budget")
            for key in list(chain) + fallback:
                if key not in models:
                    raise ValueError(f"role {role!r} names an unknown model {key!r}")
        for spec in models.values():
            if spec.provider not in pools or spec.provider not in transports:
                raise ValueError(f"model {spec.key!r}: provider {spec.provider!r} has no keys or no transport")
        if max_chain < 1:
            raise ValueError("max_chain must be at least 1")
        self.models = models
        self.budgets = budgets
        self.pools = pools
        self.transports = transports
        self.breaker = breaker or CircuitBreaker(clock=clock)
        self.on_attempt = on_attempt
        self.max_chain = max_chain
        self._chains = {role: list(chain) for role, chain in chains.items()}
        self._fallback = fallback
        self._clock = clock
        self.warnings: list[str] = []  # filled by the config loader: what was dropped, and why

    @property
    def roles(self) -> list[str]:
        return list(self._chains)

    def chain_for(self, role: str) -> list[str]:
        """The role's own models, then the shared fallback ones, without repeats. No filtering, no limit."""
        if role not in self._chains:
            raise UnknownRole(role)
        return list(dict.fromkeys(self._chains[role] + self._fallback))

    def _record(self, attempts: list[Attempt], attempt: Attempt) -> None:
        attempts.append(attempt)
        if self.on_attempt is not None:
            try:
                self.on_attempt(attempt)
            except Exception as e:  # a broken ledger must not break the request
                log.warning("on_attempt callback failed: %s", _describe(e))

    def _ms(self, started: float) -> int:
        return max(int((self._clock() - started) * 1000), 0)

    async def complete(self, role: str, messages: list[dict], *, max_output_tokens: int | None = None) -> Result:
        budget = self.budgets.get(role)
        if budget is None or role not in self._chains:
            raise UnknownRole(role)

        full = self.chain_for(role)
        verdicts = {key: self.breaker.admit(key) for key in full}
        live = [key for key in full if verdicts[key] is not None]
        selected = (live or full)[: self.max_chain]   # if every model is cooling down, try them anyway
        probes = [key for key in full if verdicts[key] == PROBE]

        attempts: list[Attempt] = []
        request_cost = 0.0
        saved: tuple[str, Response] | None = None   # the first truncated text, kept in case nothing better comes

        def result(key: str, response: Response, truncated: bool) -> Result:
            return Result(text=response.text, key=key, truncated=truncated, usage=response.usage,
                          request_cost=request_cost, role_spent=budget.spent, attempts=attempts)

        try:
            for position, key in enumerate(selected):
                spec = self.models[key]
                last = position == len(selected) - 1
                room = max_output_tokens or spec.max_output_tokens

                while True:   # twice at most: the retry sets room to the ceiling, so it cannot happen again
                    def attempt(outcome: str, **kw) -> Attempt:
                        return Attempt(role=role, key=key, provider=spec.provider, model=spec.model,
                                       outcome=outcome, **kw)

                    try:
                        reservation = budget.reserve(role, worst_case(spec, messages, room), attempts)
                    except BudgetExhausted:
                        if saved is not None:
                            return result(*saved, truncated=True)
                        raise
                    try:
                        pool = self.pools[spec.provider]
                        cred = await pool.acquire(estimate_tokens(messages, room))
                        if cred is None:
                            self._record(attempts, attempt("no_capacity", error="no key free within the wait limit"))
                            break

                        started = self._clock()
                        failed = dict(cost=None)
                        try:
                            response = await asyncio.wait_for(
                                _call(self.transports[spec.provider], spec, cred.secret, messages, room),
                                spec.timeout)
                        except asyncio.CancelledError:
                            self._record(attempts, attempt("cancelled", error="cancelled while the call was in flight",
                                                           duration_ms=self._ms(started), **failed))
                            raise
                        except TimeoutError:
                            self.breaker.failure(key)
                            self._record(attempts, attempt("timeout", error=f"no answer in {spec.timeout}s",
                                                           duration_ms=self._ms(started), **failed))
                            break
                        except TransportError as e:
                            rest = cooldown_for(e.status)
                            if rest is not None:
                                pool.cooldown(cred, e.retry_after if e.retry_after is not None else rest)
                            self.breaker.failure(key)
                            self._record(attempts, attempt("http_error", error=_describe(e),
                                                           duration_ms=self._ms(started), **failed))
                            break
                        except Exception as e:
                            self.breaker.failure(key)
                            self._record(attempts, attempt("error", error=_describe(e),
                                                           duration_ms=self._ms(started), **failed))
                            break

                        if not isinstance(response, Response):
                            self.breaker.failure(key)
                            self._record(attempts, attempt("bad_response", duration_ms=self._ms(started),
                                                           error=f"transport returned {type(response).__name__}",
                                                           **failed))
                            break
                        if not isinstance(response.usage, Usage):
                            log.warning("%s: response without usage; priced at zero", key)
                            response.usage = Usage()
                        cost = cost_of(spec, response)
                        over = cost - reservation.amount if cost > reservation.amount else None
                        if over is not None:
                            log.warning("%s cost $%.6f, $%.6f more than the worst case reserved for it: the cap can "
                                        "be passed (are the prices in the config out of date?)",
                                        key, cost, over)
                        budget.settle(reservation, cost)
                        request_cost += cost
                        done = dict(usage=response.usage, cost=cost, duration_ms=self._ms(started),
                                    over_reserved=over)
                        if not isinstance(response.text, str):
                            response.text = ""

                        if response.finish == "refusal":
                            self._record(attempts, attempt("refused", **done))
                            break
                        if not response.text.strip():
                            thought = response.usage.output + response.usage.reasoning > 0
                            ceiling = spec.output_ceiling
                            if response.finish == "length" and thought and ceiling and room < ceiling:
                                self._record(attempts, attempt("retry_more_room", **done))
                                room = ceiling
                                continue
                            self.breaker.failure(key)
                            self._record(attempts, attempt("empty", **done))
                            break
                        if response.finish == "length" and not last:
                            self._record(attempts, attempt("truncated", **done))
                            saved = saved or (key, response)
                            break

                        self.breaker.success(key)
                        truncated = response.finish == "length"
                        self._record(attempts, attempt("truncated" if truncated else "ok", **done))
                        return result(key, response, truncated)
                    finally:
                        budget.release(reservation)
        finally:
            for key in probes:   # a probe that got no verdict goes back, so another request can try
                self.breaker.done(key)

        if saved is not None:
            return result(*saved, truncated=True)
        raise ChainExhausted(role, attempts)


__all__ = ["Router", "cooldown_for", "estimate_tokens"]
