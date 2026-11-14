"""A scripted run: no keys, no network, about a second.

Three made-up providers. The cheap model times out, then gets rate limited, and its circuit breaker opens; the
strong model cuts one answer short and refuses another; the backup model catches what falls through. Every call
reserves its worst case against the role's cap first; in the end the next call no longer fits, and it is refused
before it is sent. Every attempt goes to a JSON Lines ledger, summarized at the bottom.

    python examples/demo.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from llm_router import (BudgetExhausted, CircuitBreaker, Delay, FakeTransport, JsonlLedger, KeyPool, ModelSpec,
                        Response, RoleBudget, Router, TransportError, Usage)
from llm_router.ledger import main as ledger_summary

ANSWER = Usage(input=1200, output=250)

# One-line summaries: 300 tokens of output room is plenty, and it is what each call reserves against the cap.
MODELS = {
    "fast": ModelSpec("fast", "alpha", "alpha-mini", input_per_m=0.15, output_per_m=0.60, timeout=0.1,
                      max_output_tokens=300),
    "smart": ModelSpec("smart", "beta", "beta-large", input_per_m=3.0, output_per_m=15.0, max_output_tokens=300),
    "backup": ModelSpec("backup", "gamma", "gamma-flash", input_per_m=0.50, output_per_m=1.50,
                        max_output_tokens=300),
}

# A made-up ticket of about 3,600 characters: roughly the 1,200 input tokens the scripted providers bill.
TICKET = "Summarize this support ticket in one line.\n\n" + " ".join(
    f"Message {i}: the quarterly export still fails on our largest files and the team needs it before Friday."
    for i in range(1, 36))

SCRIPT = {
    "fast": [
        Delay(0.5, Response("too late")),                                   # request 1: times out
        TransportError("HTTP 429: rate limited", status=429, retry_after=30),  # request 2: key cools down
    ],
    "smart": [
        Response("The customer reports that the export", ANSWER, finish="length"),  # request 1: cut short
        Response("Export fails on files over 2 GB; the customer needs it by Friday.", ANSWER),
        Response("Refund requested for a double charge on the March invoice.", ANSWER),
        Response("", Usage(input=1200, output=12), finish="refusal"),                  # request 4: refuses
    ],
    "backup": [
        Response("Export fails on large files; urgent for the customer.", ANSWER),
        Response("The customer wants the account closed after a data export.", ANSWER),
    ],
}


def show(n: int, result=None, error=None, attempts=()) -> None:
    print(f"request {n}")
    for a in attempts:
        cost = "?" if a.cost is None else f"${a.cost:.6f}"
        print(f"   {a.key:7} {a.outcome:12} {cost:>10}  {a.error or ''}".rstrip())
    if result is not None:
        cut = " (cut short)" if result.truncated else ""
        print(f"   -> {result.key}{cut}: {result.text!r}")
        print(f"   request ${result.request_cost:.6f}, role spent ${result.role_spent:.6f}\n")
    else:
        print(f"   -> refused before any call: {error}\n")


async def main() -> None:
    ledger_path = os.path.join(tempfile.mkdtemp(prefix="llm-router-demo-"), "attempts.jsonl")
    router = Router(
        models=MODELS,
        chains={"summarize": ["fast", "smart"]},
        fallback=["backup"],
        budgets={"summarize": RoleBudget(cap_usd=0.032, window_s=3600)},
        pools={p: KeyPool(p, [(f"{p}-key", "demo")], rpm=60) for p in ("alpha", "beta", "gamma")},
        transports={p: FakeTransport(SCRIPT) for p in ("alpha", "beta", "gamma")},
        breaker=CircuitBreaker(threshold=2, cooldown=300),
        on_attempt=JsonlLedger(ledger_path),
    )
    ticket = [{"role": "user", "content": TICKET}]
    for n in range(1, 6):
        try:
            result = await router.complete("summarize", ticket)
            show(n, result, attempts=result.attempts)
        except BudgetExhausted as e:
            show(n, error=e)
        if n == 2:
            print(f"   breaker: {router.breaker.snapshot()}\n")
    print("ledger summary")
    ledger_summary([ledger_path])


if __name__ == "__main__":
    asyncio.run(main())
