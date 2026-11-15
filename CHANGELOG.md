# Changelog

## 2.0.0 (2026-11-14)

First public release: the routing layer of a multi-agent system I built and have run in production for months,
rewritten as a standalone library with its own test suite. Fallback chains per role, a spending cap per role that
reserves each call's worst case before it is sent, rate limits per API key, a circuit breaker per model, and a ledger
of every attempt with its cost. Live check against the OpenAI and Anthropic APIs on 14 November 2026.

On PyPI as `zerda-llm-router` (the name `llm-router` belongs to another project); the import name is `llm_router`.
