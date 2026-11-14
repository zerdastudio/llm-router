import pytest

from helpers import OK, ok, make_router, run

from llm_router import (BudgetExhausted, ChainExhausted, Delay, Response, TransportError, UnknownRole, Usage)

MSG = [{"role": "user", "content": "hello"}]


def outcomes(result_or_error):
    return [(a.key, a.outcome) for a in result_or_error.attempts]


def test_first_model_answers():
    router, transport, _ = make_router({"a": [ok("A")], "b": [ok("B")]})
    result = run(router.complete("r", MSG))
    assert (result.text, result.key, result.truncated) == ("A", "a", False)
    assert outcomes(result) == [("a", "ok")]
    assert transport.calls == [("a", 1024)]


def test_http_error_falls_back_and_cools_the_key():
    router, _, clock = make_router({"a": [TransportError("busy", status=429, retry_after=12), ok("A")],
                                    "b": [ok("B")]})
    result = run(router.complete("r", MSG))
    assert result.key == "b"
    assert outcomes(result) == [("a", "http_error"), ("b", "ok")]
    assert result.attempts[0].cost is None  # failed in transit: cost unknown, not zero
    cred = router.pools["p-a"].credentials[0]
    assert cred.cooldown_until == clock() + 12
    # While the only key of `a` cools down, `a` cannot be tried at all.
    again = run(router.complete("r", MSG))
    assert outcomes(again) == [("a", "no_capacity"), ("b", "ok")]
    clock.advance(12)
    assert run(router.complete("r", MSG)).key == "a"


def test_status_without_cooldown_does_not_cool_the_key():
    router, _, _ = make_router({"a": [TransportError("bad request", status=400)], "b": [ok()]})
    run(router.complete("r", MSG))
    assert router.pools["p-a"].credentials[0].cooldown_until == 0.0


def test_timeout_moves_on():
    router, _, _ = make_router({"a": [Delay(1.0, ok("late"))], "b": [ok("B")]}, a={"timeout": 0.01})
    result = run(router.complete("r", MSG))
    assert outcomes(result) == [("a", "timeout"), ("b", "ok")]


def test_unexpected_exception_moves_on_even_if_unprintable():
    class Nasty(Exception):
        def __str__(self):
            raise RuntimeError("cannot print me")

    router, _, _ = make_router({"a": [Nasty()], "b": [ok("B")]})
    result = run(router.complete("r", MSG))
    assert outcomes(result) == [("a", "error"), ("b", "ok")]
    assert "<unprintable>" in result.attempts[0].error


def test_empty_answer_moves_on_and_is_paid():
    router, _, _ = make_router({"a": [Response("   ", OK)], "b": [ok("B")]})
    result = run(router.complete("r", MSG))
    assert outcomes(result) == [("a", "empty"), ("b", "ok")]
    assert result.request_cost == pytest.approx(2 * (1000 * 1 + 100 * 10) / 1e6)
    assert router.breaker.snapshot()["a"]["failures"] == 1


def test_refusal_moves_on_without_tripping_the_breaker():
    router, _, _ = make_router({"a": [Response("", OK, finish="refusal")], "b": [ok("B")]})
    result = run(router.complete("r", MSG))
    assert outcomes(result) == [("a", "refused"), ("b", "ok")]
    assert "a" not in router.breaker.snapshot()


def test_truncated_moves_on_and_is_kept_if_nothing_better():
    cut = Response("half an ans", OK, finish="length")
    router, _, _ = make_router({"a": [cut], "b": [TransportError("down", status=503)]})
    result = run(router.complete("r", MSG))
    assert outcomes(result) == [("a", "truncated"), ("b", "http_error")]
    assert (result.text, result.key, result.truncated) == ("half an ans", "a", True)


def test_truncated_from_the_last_model_is_returned():
    router, _, _ = make_router({"a": [TransportError("down", status=503)],
                                "b": [Response("cut", OK, finish="length")]})
    result = run(router.complete("r", MSG))
    assert (result.key, result.truncated) == ("b", True)
    assert outcomes(result) == [("a", "http_error"), ("b", "truncated")]


def test_thinking_without_text_gets_one_retry_with_more_room():
    thought = Response("", Usage(input=100, reasoning=1024), finish="length")
    router, transport, _ = make_router({"a": [thought, ok("A")], "b": [ok("B")]},
                                       a={"output_ceiling": 4096})
    result = run(router.complete("r", MSG))
    assert outcomes(result) == [("a", "retry_more_room"), ("a", "ok")]
    assert transport.calls == [("a", 1024), ("a", 4096)]


def test_no_second_retry_and_no_retry_without_ceiling():
    thought = Response("", Usage(input=100, reasoning=1024), finish="length")
    router, transport, _ = make_router({"a": [thought], "b": [ok("B")]}, a={"output_ceiling": 4096})
    result = run(router.complete("r", MSG))
    assert outcomes(result) == [("a", "retry_more_room"), ("a", "empty"), ("b", "ok")]
    router2, _, _ = make_router({"a": [thought], "b": [ok("B")]})
    assert outcomes(run(router2.complete("r", MSG))) == [("a", "empty"), ("b", "ok")]


def test_every_model_fails():
    router, _, _ = make_router({"a": [TransportError("x", status=500)], "b": [Response("", OK)]})
    with pytest.raises(ChainExhausted) as info:
        run(router.complete("r", MSG))
    assert outcomes(info.value) == [("a", "http_error"), ("b", "empty")]
    assert isinstance(info.value, RuntimeError)


def test_unknown_role():
    router, _, _ = make_router({"a": [ok()]})
    with pytest.raises(UnknownRole):
        run(router.complete("nobody", MSG))


def test_budget_refuses_before_any_call_and_reopens():
    # worst case of a call: ~13 input tokens at $1/M + 1024 output tokens at $10/M = ~$0.0103; a real call costs $0.002
    router, transport, clock = make_router({"a": [ok("A")]}, chain=("a",), cap=0.012, window=60)
    assert run(router.complete("r", MSG)).role_spent == pytest.approx(0.002)
    with pytest.raises(BudgetExhausted) as info:   # 0.002 spent + 0.0103 worst case > 0.012
        run(router.complete("r", MSG))
    assert info.value.resets_in == pytest.approx(60) and info.value.needed == pytest.approx(0.0103, abs=1e-4)
    assert isinstance(info.value, PermissionError)
    assert len(transport.calls) == 1
    clock.advance(60)
    assert run(router.complete("r", MSG)).role_spent == pytest.approx(0.002)


def test_discarded_answers_count_against_the_budget():
    router, _, _ = make_router({"a": [Response("", OK)], "b": [ok()]}, cap=1.0)
    result = run(router.complete("r", MSG))
    assert result.role_spent == pytest.approx(result.request_cost) == pytest.approx(0.004)


def test_breaker_skips_a_failing_model_then_gives_it_another_chance():
    router, transport, clock = make_router({"a": [TransportError("x", status=400)] * 2 + [ok("A")],
                                            "b": [ok("B")]}, threshold=2)
    run(router.complete("r", MSG))
    run(router.complete("r", MSG))
    transport.calls.clear()
    assert run(router.complete("r", MSG)).key == "b"   # `a` is skipped
    assert transport.calls == [("b", 1024)]
    clock.advance(60)
    assert run(router.complete("r", MSG)).key == "a"   # half open: tried again, and it works
    assert "a" not in router.breaker.snapshot()


def test_if_every_model_is_open_they_are_tried_anyway():
    router, _, _ = make_router({"a": [TransportError("x", status=400)], "b": [TransportError("x", status=400)]},
                               threshold=1)
    with pytest.raises(ChainExhausted):
        run(router.complete("r", MSG))
    with pytest.raises(ChainExhausted) as info:
        run(router.complete("r", MSG))
    assert [a.key for a in info.value.attempts] == ["a", "b"]


def test_fallback_is_appended_without_repeats_and_chain_is_limited():
    router, transport, _ = make_router({"a": [TransportError("x", status=500)], "b": [TransportError("x", status=500)],
                                        "c": [ok("C")]}, chain=("a", "b"), fallback=("b", "c"), max_chain=2)
    assert router.chain_for("r") == ["a", "b", "c"]
    with pytest.raises(ChainExhausted):
        run(router.complete("r", MSG))
    assert [k for k, _ in transport.calls] == ["a", "b"]


def test_on_attempt_sees_every_attempt_and_a_broken_callback_is_harmless():
    seen = []
    router, _, _ = make_router({"a": [TransportError("x", status=500)], "b": [ok()]}, on_attempt=seen.append)
    run(router.complete("r", MSG))
    assert [a.outcome for a in seen] == ["http_error", "ok"]

    def broken(attempt):
        raise ValueError("disk full")

    router2, _, _ = make_router({"a": [ok()]}, on_attempt=broken)
    assert run(router2.complete("r", MSG)).text == "fine"


def test_keys_are_used_in_turn():
    router, transport, _ = make_router({"a": [ok()]}, chain=("a",), keys=2)
    used = []
    original = transport.__call__

    async def spy(spec, api_key, messages, room):
        used.append(api_key)
        return await original(spec, api_key, messages, room)

    router.transports["p-a"] = spy
    for _ in range(3):
        run(router.complete("r", MSG))
    assert used == ["secret-0", "secret-1", "secret-0"]


def test_constructor_rejects_inconsistent_setups():
    router, _, _ = make_router({"a": [ok()]})
    with pytest.raises(ValueError):
        type(router)(models=router.models, chains={"r": ["a", "zzz"]}, budgets=router.budgets, pools=router.pools,
                     transports=router.transports)
    with pytest.raises(ValueError):
        type(router)(models=router.models, chains={"r": ["a"]}, budgets={}, pools=router.pools,
                     transports=router.transports)
