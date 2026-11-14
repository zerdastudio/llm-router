"""Live checks of the real transports, against the providers' APIs. They cost a fraction of a cent.

Skipped unless LIVE_TESTS=1, so CI and a plain `pytest` never touch the network. To run them:

    pip install -e ".[dev,openai,anthropic]"
    LIVE_TESTS=1 OPENAI_API_KEY=... LIVE_OPENAI_MODEL=<a chat model> \
    ANTHROPIC_API_KEY=... LIVE_ANTHROPIC_MODEL=<a model> pytest -v tests/test_live.py

LIVE_OPENAI_BASE_URL points the OpenAI-compatible transport at another provider (with that provider's key and
model). A provider whose key or model is missing is skipped, and the reason names the missing variable.
"""
import asyncio
import os

import pytest

from llm_router import AnthropicTransport, ChainExhausted, OpenAICompatTransport, TransportError, build_router
from llm_router.base import ModelSpec

pytestmark = pytest.mark.skipif(os.environ.get("LIVE_TESTS") != "1", reason="live tests: set LIVE_TESTS=1")

QUESTION = [{"role": "system", "content": "Answer with one word."},
            {"role": "user", "content": "What is the capital of France?"}]


def need(*names):
    missing = [n for n in names if not os.environ.get(n, "").strip()]
    if missing:
        pytest.skip("missing " + ", ".join(missing))
    return [os.environ[n] for n in names]


def openai_transport():
    base = os.environ.get("LIVE_OPENAI_BASE_URL") or None
    # OpenAI's own API wants max_completion_tokens; the compatible providers mostly take max_tokens
    return OpenAICompatTransport(base, max_tokens_param="max_tokens" if base else "max_completion_tokens")


PROVIDERS = {
    "openai": ("OPENAI_API_KEY", "LIVE_OPENAI_MODEL", "openai", openai_transport),
    "anthropic": ("ANTHROPIC_API_KEY", "LIVE_ANTHROPIC_MODEL", "anthropic", AnthropicTransport),
}


@pytest.mark.parametrize("name", PROVIDERS)
def test_a_real_call_through_the_router_is_answered_counted_and_priced(name):
    key_var, model_var, transport_kind, make = PROVIDERS[name]
    _, model = need(key_var, model_var)
    rows = []
    conf = {
        "providers": {name: {"transport": transport_kind, "api_key_env": [key_var]}},
        # nominal prices: the test checks that real usage is turned into a cost, not what the provider bills
        "models": {"m": {"provider": name, "model": model, "input_per_m": 1.0, "output_per_m": 1.0,
                         "max_output_tokens": 200, "output_ceiling": 2000, "timeout": 60}},
        "roles": {"ask": {"chain": ["m"], "budget_usd": 1.0}},
    }
    router = build_router(conf, env=os.environ, transports={name: make()}, on_attempt=rows.append)
    result = None
    try:
        result = asyncio.run(router.complete("ask", QUESTION))
    except ChainExhausted:
        pass
    if result is None:
        # the ledger keeps the status only; one direct call shows the provider's own error message. No traceback:
        # pytest would print the transport's arguments, the API key among them
        said = "nothing (the direct call worked)"
        try:
            spec = router.models["m"]
            asyncio.run(make()(spec, os.environ[key_var], QUESTION, spec.max_output_tokens))
        except Exception as e:
            said = str(e.__cause__ or e)
        pytest.fail(f"every attempt failed: {[(a.outcome, a.error) for a in rows]}; the provider said: {said}",
                    pytrace=False)
    assert "paris" in result.text.lower()
    assert result.usage.input > 0 and result.usage.output + result.usage.reasoning > 0
    assert rows[-1].outcome == "ok"                # a thinking model may first need a retry with more room
    for a in rows:                                 # every token at $1 per million
        assert a.cost == pytest.approx(a.usage.total / 1e6)
    assert result.request_cost == pytest.approx(sum(a.cost for a in rows)) and result.request_cost > 0


@pytest.mark.parametrize("name", PROVIDERS)
def test_a_wrong_key_comes_back_as_an_http_error(name):
    _, model_var, _, make = PROVIDERS[name]
    (model,) = need(model_var)
    spec = ModelSpec(key="m", provider=name, model=model, input_per_m=1.0, output_per_m=1.0)
    with pytest.raises(TransportError) as e:
        asyncio.run(make()(spec, "not-a-real-key", QUESTION, 20))
    assert e.value.status in (400, 401, 403)     # 401 at OpenAI and Anthropic; some compatible APIs say 400
