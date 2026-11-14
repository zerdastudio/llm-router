import os
import tomllib

import pytest

from helpers import ok, run

from llm_router import FakeTransport, JsonlLedger, load_router, summarize
from llm_router.config import build_router
from llm_router.ledger import main as ledger_main

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.join(HERE, "..", "examples", "router.example.toml")

CONF = """
[router]
fallback = ["backup"]
budget_usd = 2.0
max_chain = 3

[providers.one]
transport = "openai"
api_key_env = ["ONE_KEY", "ONE_KEY_2"]
rpm = 100

[providers.two]
transport = "anthropic"
api_key_env = "TWO_KEY"

[models.a]
provider = "one"
model = "one-small"
input_per_m = 0.1
output_per_m = 0.4

[models.b]
provider = "two"
model = "two-big"
input_per_m = 3
output_per_m = 15
output_ceiling = 8000

[models.backup]
provider = "one"
model = "one-tiny"
input_per_m = 0.05
output_per_m = 0.2

[roles.summarize]
chain = ["a", "b"]
budget_usd = 0.5
window_hours = 1

[roles.review]
chain = ["b"]
"""


def conf():
    return tomllib.loads(CONF)


def test_build_with_all_keys():
    env = {"ONE_KEY": "k1", "ONE_KEY_2": "k2", "TWO_KEY": "k3"}
    router = build_router(conf(), env=env)
    assert router.roles == ["summarize", "review"]
    assert router.chain_for("summarize") == ["a", "b", "backup"]
    assert router.budgets["summarize"].cap == 0.5 and router.budgets["summarize"].window_s == 3600
    assert router.budgets["review"].cap == 2.0 and router.budgets["review"].window_s == 24 * 3600
    assert [c.name for c in router.pools["one"].credentials] == ["ONE_KEY", "ONE_KEY_2"]
    assert router.models["b"].output_ceiling == 8000
    assert router.warnings == []


def test_empty_keys_drop_the_provider_its_models_and_empty_roles():
    router = build_router(conf(), env={"ONE_KEY": "k1", "TWO_KEY": "  "})
    assert set(router.pools) == {"one"}
    assert router.chain_for("summarize") == ["a", "backup"]
    # `review` had only `b`, but the shared fallback still gives it a model.
    assert router.chain_for("review") == ["backup"]
    assert any("provider 'two'" in w for w in router.warnings)
    assert any("model 'b'" in w for w in router.warnings)


def test_role_with_nothing_left_is_not_registered():
    c = conf()
    c["router"]["fallback"] = []
    router = build_router(c, env={"ONE_KEY": "k1"})
    assert router.roles == ["summarize"]
    assert any("role 'review': no model left" in w for w in router.warnings)


def test_no_usable_role_at_all_is_an_error():
    with pytest.raises(ValueError, match="no role"):
        build_router(conf(), env={})


def test_bad_price_and_missing_budget_are_errors():
    c = conf()
    c["models"]["a"]["input_per_m"] = -1
    with pytest.raises(ValueError, match="input_per_m"):
        build_router(c, env={"ONE_KEY": "k"})
    c = conf()
    del c["router"]["budget_usd"]
    with pytest.raises(ValueError, match="budget_usd"):
        build_router(c, env={"ONE_KEY": "k", "TWO_KEY": "k"})


def test_unknown_transport_is_an_error():
    c = conf()
    c["providers"]["one"]["transport"] = "carrier-pigeon"
    with pytest.raises(ValueError, match="unknown transport"):
        build_router(c, env={"ONE_KEY": "k"})


def test_example_file_loads_and_runs_with_injected_transports(tmp_path):
    env = {"OPENAI_API_KEY": "x", "ANTHROPIC_API_KEY": "y", "OPENROUTER_API_KEY": "z"}
    with open(EXAMPLE, "rb") as f:
        example = tomllib.load(f)
    script = {key: [ok(f"from {key}")] for key in example["models"]}
    fake = FakeTransport(script)
    ledger = tmp_path / "attempts.jsonl"
    router = load_router(EXAMPLE, env=env, transports={p: fake for p in example["providers"]},
                         on_attempt=JsonlLedger(str(ledger)))
    assert router.warnings == []
    first = router.chain_for("summarize")[0]
    result = run(router.complete("summarize", [{"role": "user", "content": "hi"}]))
    assert result.text == f"from {first}"
    rows = summarize(str(ledger))
    assert len(rows) == 1 and rows[0]["ok"] == 1


def test_ledger_summary_counts_and_unknown_costs(tmp_path, capsys):
    from llm_router import Attempt
    path = str(tmp_path / "l.jsonl")
    ledger = JsonlLedger(path)
    ledger(Attempt("r", "a", "p", "m", "ok", cost=0.01, duration_ms=100))
    ledger(Attempt("r", "a", "p", "m", "ok", cost=0.02, duration_ms=300))
    ledger(Attempt("r", "a", "p", "m", "timeout", cost=None, duration_ms=5000))
    ledger(Attempt("r", "a", "p", "m", "retry_more_room", cost=0.001))
    [row] = summarize(path)
    assert row["attempts"] == 4 and row["ok"] == 2
    assert row["failed"] == {"timeout": 1}
    assert row["cost_usd"] == pytest.approx(0.031) and row["unknown_cost"] == 1
    assert row["avg_ok_ms"] == 200   # the slow timeout does not count
    assert ledger_main([path]) == 0
    assert "p/m" in capsys.readouterr().out
    assert ledger_main([]) == 2


def test_missing_price_is_an_error_and_free_must_be_said():
    c = conf()
    del c["models"]["backup"]["output_per_m"]
    with pytest.raises(ValueError, match="free = true"):
        build_router(c, env={"ONE_KEY": "k", "TWO_KEY": "k"})
    c = conf()
    del c["models"]["backup"]["input_per_m"], c["models"]["backup"]["output_per_m"]
    c["models"]["backup"]["free"] = True
    router = build_router(c, env={"ONE_KEY": "k", "TWO_KEY": "k"})
    assert router.models["backup"].output_per_m == 0
    assert any("declared free" in w for w in router.warnings)


@pytest.mark.parametrize("field,value", [("output_ceiling", "8192"), ("output_ceiling", 100), ("timeout", 0),
                                         ("timeout", "60"), ("max_output_tokens", -5), ("max_output_tokens", 1.5)])
def test_bad_model_settings_are_refused_at_build(field, value):
    c = conf()
    c["models"]["a"][field] = value
    with pytest.raises(ValueError, match=field):
        build_router(c, env={"ONE_KEY": "k", "TWO_KEY": "k"})


@pytest.mark.parametrize("field,value", [("rpm", 0), ("tpm", "lots"), ("max_wait", -1)])
def test_bad_provider_limits_are_refused(field, value):
    c = conf()
    c["providers"]["one"][field] = value
    with pytest.raises(ValueError, match=field):
        build_router(c, env={"ONE_KEY": "k", "TWO_KEY": "k"})


def test_every_setting_reaches_the_router():
    c = conf()
    c["router"].update(window_hours=2, breaker_threshold=7, breaker_cooldown=45)
    c["providers"]["one"].update(tpm=5000, rpd=900, max_wait=3, max_tokens_param="max_completion_tokens",
                                 use_reported_cost=True, base_url="https://example.invalid/v1")
    c["models"]["a"].update(timeout=12, max_output_tokens=300, output_ceiling=900, cache_read_per_m=0.01)
    router = build_router(c, env={"ONE_KEY": "k", "TWO_KEY": "k"})
    assert router.budgets["review"].window_s == 2 * 3600          # [router] window_hours as default
    assert router.budgets["summarize"].window_s == 3600           # the role's own wins
    assert (router.breaker.threshold, router.breaker.cooldown) == (7, 45)
    assert router.max_chain == 3
    pool = router.pools["one"]
    assert pool.max_wait == 3
    cred = pool.credentials[0]
    assert (cred.requests_minute.capacity, cred.tokens_minute.capacity, cred.requests_day.capacity) == (100, 5000, 900)
    assert cred.requests_day.rate == pytest.approx(900 / 86400)
    t = router.transports["one"]
    assert (t.max_tokens_param, t.use_reported_cost, t.base_url) == \
        ("max_completion_tokens", True, "https://example.invalid/v1")
    spec = router.models["a"]
    assert (spec.timeout, spec.max_output_tokens, spec.output_ceiling, spec.cache_read_per_m) == (12, 300, 900, 0.01)
    assert type(router.transports["two"]).__name__ == "AnthropicTransport"
