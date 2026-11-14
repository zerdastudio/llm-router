"""Data shapes shared by the router, the transports and the ledger."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    """One model, reached through one provider, with its prices in USD per million tokens.

    `key` is the name the router uses (in chains, the breaker and the ledger); `model` is the identifier sent to
    the provider. They can differ, so the same model can be registered twice, through two providers.
    """

    key: str
    provider: str
    model: str
    input_per_m: float = 0.0
    output_per_m: float = 0.0
    cache_read_per_m: float | None = None   # default: the input price
    cache_write_per_m: float | None = None  # default: the input price
    reasoning_per_m: float | None = None    # default: the output price
    timeout: float = 60.0
    max_output_tokens: int = 1024
    # Room for one retry when a model spends all its output tokens thinking and writes no text. None: no retry.
    output_ceiling: int | None = None

    def __post_init__(self):
        for name in ("key", "provider", "model"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"model spec: {name} must be a non-empty string")
        where = f"model {self.key!r}"
        for name in ("input_per_m", "output_per_m", "cache_read_per_m", "cache_write_per_m", "reasoning_per_m"):
            value = getattr(self, name)
            if value is None and name not in ("input_per_m", "output_per_m"):
                continue
            if not _number(value) or value < 0:
                raise ValueError(f"{where}: {name} must be a number >= 0, not {value!r}")
        if not _number(self.timeout) or self.timeout <= 0:
            raise ValueError(f"{where}: timeout must be a number of seconds > 0, not {self.timeout!r}")
        if not _integer(self.max_output_tokens) or self.max_output_tokens < 1:
            raise ValueError(f"{where}: max_output_tokens must be an integer >= 1, not {self.max_output_tokens!r}")
        if self.output_ceiling is not None and (not _integer(self.output_ceiling)
                                                or self.output_ceiling < self.max_output_tokens):
            raise ValueError(f"{where}: output_ceiling must be an integer >= max_output_tokens, "
                             f"not {self.output_ceiling!r}")


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass
class Usage:
    """Tokens of one response, split the way providers bill them."""

    input: int = 0        # input not served from the cache
    cache_read: int = 0
    cache_write: int = 0
    output: int = 0       # visible output
    reasoning: int = 0    # thinking tokens, billed as output by most providers

    @property
    def total(self) -> int:
        return self.input + self.cache_read + self.cache_write + self.output + self.reasoning


@dataclass
class Response:
    """What a transport returns for one call."""

    text: str
    usage: Usage = field(default_factory=Usage)
    finish: str = "stop"                   # "stop" | "length" (cut short) | "refusal"
    reported_cost: float | None = None     # a cost the provider reported itself; wins over table pricing


@dataclass
class Attempt:
    """One try of one model, as the ledger records it."""

    role: str
    key: str
    provider: str
    model: str
    # ok | truncated | empty | refused | retry_more_room | http_error | timeout | error | no_capacity
    # | bad_response | cancelled
    outcome: str
    error: str | None = None
    usage: Usage = field(default_factory=Usage)
    # None means unknown: the call failed in transit, and the provider may still have billed it.
    cost: float | None = 0.0
    duration_ms: int = 0
    # How much the call cost beyond what was reserved for it (stale prices in the config); None when it fit.
    over_reserved: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Result:
    text: str
    key: str                 # the model that answered
    truncated: bool          # True when the text was cut short and no later model did better
    usage: Usage             # the usage of the returned response only
    request_cost: float      # every priced response of this request, discarded ones included
    role_spent: float        # the role's spending in its current window, after this request
    attempts: list[Attempt]


class RouterError(Exception):
    """Base class of the errors the router raises on purpose."""


class UnknownRole(RouterError, LookupError):
    def __init__(self, role: str):
        super().__init__(f"unknown role {role!r}")
        self.role = role


class BudgetExhausted(RouterError, PermissionError):
    """The worst case of the next call does not fit in what the role has left. That call was not sent.

    `attempts` lists what this request did before it stopped (empty when it stopped before the first call).
    """

    def __init__(self, role: str, spent: float, cap: float, resets_in: float | None, *, needed: float = 0.0,
                 reserved: float = 0.0, attempts: list | None = None):
        when = "never" if resets_in is None else f"in {resets_in:.0f}s"
        held = f", ${reserved:.4f} held by requests in flight" if reserved else ""
        super().__init__(f"role {role!r} spent ${spent:.4f} of ${cap:.4f}{held}; the next call may cost up to "
                         f"${needed:.4f}; the window resets {when}")
        self.role, self.spent, self.cap, self.resets_in = role, spent, cap, resets_in
        self.needed, self.reserved = needed, reserved
        self.attempts = attempts or []


class ChainExhausted(RouterError, RuntimeError):
    """Every model in the role's chain failed, and none left a usable (even truncated) text."""

    def __init__(self, role: str, attempts: list[Attempt]):
        tried = ", ".join(f"{a.key}:{a.outcome}" for a in attempts) or "nothing could be tried"
        super().__init__(f"role {role!r}: every model failed ({tried})")
        self.role, self.attempts = role, attempts


class TransportError(Exception):
    """A provider call that failed with an HTTP status (or none, for network errors)."""

    def __init__(self, message: str, status: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
