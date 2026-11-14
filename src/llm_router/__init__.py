"""Route LLM calls across providers: fallback chains per role, spending caps, rate limits, a cost ledger."""
from .base import (Attempt, BudgetExhausted, ChainExhausted, ModelSpec, Response, Result, RouterError,
                   TransportError, UnknownRole, Usage)
from .budget import RoleBudget
from .config import build_router, load_router
from .ledger import JsonlLedger, summarize
from .limits import CircuitBreaker, KeyPool, TokenBucket
from .pricing import price
from .router import Router
from .transports import AnthropicTransport, Delay, FakeTransport, OpenAICompatTransport, Transport

__version__ = "0.1.0"

__all__ = [
    "Attempt", "BudgetExhausted", "ChainExhausted", "ModelSpec", "Response", "Result", "RouterError",
    "TransportError", "UnknownRole", "Usage", "RoleBudget", "build_router", "load_router", "JsonlLedger",
    "summarize", "CircuitBreaker", "KeyPool", "TokenBucket", "price", "Router", "AnthropicTransport", "Delay",
    "FakeTransport", "OpenAICompatTransport", "Transport",
]
