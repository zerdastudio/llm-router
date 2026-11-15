# llm-router

[![tests](https://github.com/zerdastudio/llm-router/actions/workflows/tests.yml/badge.svg)](https://github.com/zerdastudio/llm-router/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Route LLM calls across providers from inside your own Python process: **a fallback chain per role, a spending
cap per role (in this process, enforced by reserving each call's worst case before it is sent), rate limits per
API key, a circuit breaker per model, and a ledger of every attempt with its cost**, the failed and discarded ones
included.

Pure Python, asyncio, no required dependencies. OpenAI, Anthropic, and OpenAI-compatible APIs (OpenRouter,
Gemini, DeepSeek, local servers) through optional SDKs.

**Where it comes from.** This is the routing layer of a multi-agent system I built and have run in production
for months, against live providers, rewritten here as a standalone library with its own test suite. Earlier
versions lived inside that system and were never published; this standalone rewrite starts at 2.0.

## When to use it

When the routing should live in your code, small enough to read in an afternoon: a service or an agent that
calls a few providers, needs a fallback when one fails, and must not overspend. If you need a hundred providers,
streaming and tool calls, or one budget shared by many processes, a full gateway or a large multi-provider
SDK is the better choice: this library stays small on purpose, so every decision it makes shows in the ledger.

## In your code

```python
import asyncio
from llm_router import BudgetExhausted, ChainExhausted, JsonlLedger, load_router

router = load_router("examples/router.example.toml", on_attempt=JsonlLedger("attempts.jsonl"))

async def summarize(text: str) -> str | None:
    try:
        result = await router.complete("summarize", [{"role": "user", "content": f"Summarize: {text}"}])
    except BudgetExhausted as e:       # the next call would not fit under the role's cap; it was not sent
        print(e)                       # what was spent, what the next call needed, when the window resets
        return None
    except ChainExhausted as e:        # every model failed; e.attempts says how
        print([(a.key, a.outcome) for a in e.attempts])
        return None
    if result.truncated:               # the best answer was cut short: decide whether that is good enough
        print("truncated answer from", result.key)
    print(f"{result.key}: ${result.request_cost:.6f} this request, ${result.role_spent:.6f} spent by the role")
    return result.text

asyncio.run(summarize("..."))
```

`Result` also carries `usage` and every `attempts` row; `router.warnings` lists what the configuration dropped.

## What it looks like

`python examples/demo.py` runs a scripted day with three made-up providers. No keys, no network:

```text
request 1
   fast    timeout               ?  no answer in 0.1s
   smart   truncated     $0.007350
   backup  ok            $0.000975
   -> backup: 'Export fails on large files; urgent for the customer.'
   request $0.008325, role spent $0.008325

request 2
   fast    http_error            ?  TransportError: HTTP 429: rate limited
   smart   ok            $0.007350
   -> smart: 'Export fails on files over 2 GB; the customer needs it by Friday.'
   request $0.007350, role spent $0.015675

   breaker: {'fast': {'failures': 2, 'open': True}}

request 3
   smart   ok            $0.007350
   -> smart: 'Refund requested for a double charge on the March invoice.'
   request $0.007350, role spent $0.023025

request 4
   smart   refused       $0.003780
   backup  ok            $0.000975
   -> backup: 'The customer wants the account closed after a data export.'
   request $0.004755, role spent $0.027780

request 5
   -> refused before any call: role 'summarize' spent $0.0278 of $0.0320; the next call may cost up to $0.0082; the window resets in 3600s

ledger summary
provider/model                     tries   ok    known $ unknown avg ok ms  failures
alpha/alpha-mini                       2    0   0.000000       2         -  http_error 1, timeout 1
beta/beta-large                        4    2   0.025830       0         0  refused 1, truncated 1
gamma/gamma-flash                      2    2   0.001950       0         0  -
```

The cheap model times out, then hits a rate limit, and after two failures its breaker opens: request 3 does not
even try it. The strong model cuts one answer short (kept, in case nothing better comes) and refuses another;
the backup model catches both. Every call first reserves its worst case against the role's cap; every paid
response counts, the discarded ones too; and when the next call's worst case no longer fits, it is refused before
anything is sent. A `?` cost means the call failed in transit: the provider may still have billed it, and the
ledger says so instead of writing zero.

## Quick start

```bash
pip install zerda-llm-router                       # the library, no dependencies
pip install "zerda-llm-router[openai,anthropic]"    # with the provider SDKs
```

The import name is `llm_router`. Not `pip install llm-router`: that is another project, with the same
import name. To run the tests and the demo, work from a clone:

```bash
git clone https://github.com/zerdastudio/llm-router
cd llm-router
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                    # 99 tests, offline, under a second
python examples/demo.py   # the run above
```

For real calls, install the SDKs you need and put your keys in environment variables, then run the code above:

```bash
pip install -e ".[openai,anthropic]"
export OPENAI_API_KEY=...   ANTHROPIC_API_KEY=...
```

Then `python -m llm_router.ledger attempts.jsonl` prints the per-model table: `known $` is what the priced
attempts cost, `unknown` counts the attempts that failed in transit, whose cost nobody knows.

## How a request goes

1. **The chain.** The role's models, then the shared fallback models, without repeats. Models whose breaker is
   open are left out, unless that would leave nothing: then all are tried. After its cooldown, a model with an
   open breaker gets one trial request, and only one, while the others keep skipping it. At most `max_chain`
   models.
2. **A reservation.** Before each call, its worst-case cost is held against the role's cap: the input
   (estimated generously from the characters) plus the whole output room, at the model's highest prices. The call
   is sent only if spending + what other calls in flight hold + this worst case fits under the cap. Otherwise:
   `BudgetExhausted`, with the attempts made so far (or the truncated text kept so far, if there is one).
3. **A key.** For each model, a key of its provider that is not cooling down and has room in its
   requests-per-minute, tokens-per-minute and requests-per-day buckets. Keys are used in turn. If none is free,
   the router waits for the next refill when it is close (`max_wait`), and otherwise skips the model.
4. **The call**, with a timeout. An error or a timeout moves on to the next model, and frees the reservation.
   A 429, or any 5xx except 501, also puts that key on a cooldown (the provider's `retry-after`, when it sends
   one). A cancelled request is written to the ledger before the cancellation goes on.
5. **The answer.** It replaces the reservation with its real cost. An answer that is not a `Response` counts as
   a failure of that model; one without usage is priced at zero, with a warning. Then:
   - a refusal moves on (a refusal is about the request, so it does not count against the model);
   - no visible text moves on; if the model spent all its tokens thinking, it first gets one retry with more
     room, up to `output_ceiling`;
   - a text cut short moves on while another model remains, and is kept: if nothing better comes, it is returned
     with `truncated=True`;
   - anything else is the answer.
6. If the chain runs out with nothing usable, a `ChainExhausted` error lists every attempt.

## Configuration

See [`examples/router.example.toml`](https://github.com/zerdastudio/llm-router/blob/main/examples/router.example.toml). In short:

```toml
[providers.openrouter]
transport = "openai"                       # or "anthropic"
base_url = "https://openrouter.ai/api/v1"
api_key_env = ["OPENROUTER_API_KEY"]       # names of env variables, never the keys
use_reported_cost = true
rpm = 60

[models.cheap]
provider = "openrouter"
model = "some/model-id"
input_per_m = 0.10                         # USD per million tokens
output_per_m = 0.30

[roles.summarize]
chain = ["cheap", "strong"]
budget_usd = 1.0
window_hours = 24
```

An empty or missing key variable is skipped; a provider left with no keys drops out with its models, and a role
left with no models is not registered. All of it is reported in `router.warnings`, so a half-configured router
says what it is missing instead of failing on the first call.

What would break a request later is refused when the router is built: a model without `input_per_m` and
`output_per_m` (the reservation needs them; a model that really costs nothing says `free = true`, and gets a
warning that the cap does not limit it), a timeout that is not positive, an output room or ceiling that is not
a whole number of tokens, a rate limit below 1.

## Design choices

- **Prices are configuration, not code.** Providers change them; a table in code goes stale without telling
  anyone. When a provider reports the real cost (OpenRouter does), that wins.
- **Usage is split the way providers bill it**: uncached input, cache reads, cache writes, visible output and
  reasoning tokens. Some OpenAI-compatible endpoints report thinking only in the total; the excess is counted
  as reasoning.
- **The SDKs' own retries are off.** Retrying is the router's job, and every try shows in the ledger.
- **Everything with a clock takes the clock as a parameter**, so rate limits, cooldowns, breakers and budget
  windows are tested to the second without waiting.
- **Transports are plain async callables** `(spec, api_key, messages, max_output_tokens) -> Response`. A new
  provider is one small function; the tests use a scripted fake one.

## Limits

- State lives in memory, per process: spending, reservations, buckets and breakers start fresh at every restart,
  and two processes do not share them. The cap is a guard, not an accounting system: use the ledger for that.
- The cap holds as long as each worst case is not below the real cost. The output side is exact; the input side
  is estimated at three characters a token, so a provider that bills far more input than that can take the
  spending a little past the cap. A provider-reported cost above the table price (stale prices in the config)
  goes past the reservation too: the router logs a warning, and the ledger row says by how much
  (`over_reserved`). A small cap with a large output room refuses everything: the error says how
  much the next call would need.
- Calls that fail in transit are not charged (their cost is unknown, and the ledger says so). So the cap does not
  stop a provider that keeps failing; the breaker slows it down, and once every model is open they are tried
  anyway, because a request with no model to try can only fail.
- Token counts for the rate limiter are estimated (about four characters a token) before the call.
- Text only: no streaming, no tool calls, no structured-output validation.
- The test suite runs offline: in CI the transports run against stand-in SDK clients. The live check is
  `tests/test_live.py`, skipped without keys: on 14 November 2026 it passed against the OpenAI (`gpt-4o-mini`)
  and Anthropic (`claude-sonnet-4-6`) APIs, a real call through the router and a wrong key for each.

## Part of a larger system

This library is one piece of a multi-agent system I built, rewritten to stand on its own. The other piece
published so far:

- [vector-memory](https://github.com/zerdastudio/vector-memory): long-term memory for an agent in one SQLite file, built so it does not quietly rot.

Also by me, not part of the system: [ai-csv-batch](https://github.com/zerdastudio/ai-csv-batch), every row of a CSV
through Claude, GPT or Gemini, with retries, resume and a cost log.

More about the system on my [GitHub profile](https://github.com/zerdastudio).

## License

MIT
