"""The cases of the first external review: concurrency, cancellation, bad answers, the half-open breaker."""
import asyncio

import pytest

from helpers import OK, FakeClock, make_router, ok, run

from llm_router import BudgetExhausted, ChainExhausted, Delay, KeyPool, Response, TransportError, Usage
from llm_router.router import estimate_tokens

MSG = [{"role": "user", "content": "hello"}]


def test_cap_holds_under_concurrency():
    # each call: worst case ~$0.0103 reserved, real cost $0.002; cap $0.025 leaves room for two in flight
    async def go():
        router, transport, _ = make_router({"a": [Delay(0.02, ok("A"))]}, chain=("a",), cap=0.025)
        results = await asyncio.gather(*(router.complete("r", MSG) for _ in range(50)), return_exceptions=True)
        answered = [r for r in results if not isinstance(r, BaseException)]
        refused = [r for r in results if isinstance(r, BudgetExhausted)]
        assert len(answered) == 2 and len(refused) == 48
        assert router.budgets["r"].spent == pytest.approx(0.004) and router.budgets["r"].reserved == 0
        # nothing is left reserved: the next ones go through until the cap really is reached
        while True:
            try:
                await router.complete("r", MSG)
            except BudgetExhausted:
                break
        assert router.budgets["r"].spent <= 0.025
    run(go())


def test_one_request_cannot_run_far_past_the_cap():
    think = Response("", Usage(input=1000, reasoning=1024), finish="length")
    router, transport, _ = make_router({"a": [think], "b": [think], "c": [ok()]}, chain=("a", "b", "c"), cap=0.025,
                                       a={"output_ceiling": 32768}, b={"output_ceiling": 32768})
    with pytest.raises(BudgetExhausted) as info:
        run(router.complete("r", MSG))
    # a: 0.0112 spent; the retry with 32768 tokens of room would cost up to $0.33: refused before it is sent
    assert router.budgets["r"].spent <= 0.025
    assert [a.outcome for a in info.value.attempts] == ["retry_more_room"]
    assert transport.calls == [("a", 1024)]


def test_a_truncated_answer_is_returned_when_the_budget_stops_the_chain():
    # a: worst case 0.0103 fits in 0.012, costs 0.002; b: 0.002 + 0.0103 does not fit
    router, _, _ = make_router({"a": [Response("half", OK, finish="length")], "b": [ok()]}, cap=0.012)
    result = run(router.complete("r", MSG))
    assert (result.key, result.truncated, result.text) == ("a", True, "half")


def test_cancelled_call_is_in_the_ledger_and_leaves_nothing_held():
    seen = []

    async def go():
        router, _, _ = make_router({"a": [Delay(10, ok("late")), ok("A")]}, chain=("a",), on_attempt=seen.append)
        task = asyncio.create_task(router.complete("r", MSG))
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert [(a.outcome, a.cost) for a in seen] == [("cancelled", None)]
        assert router.budgets["r"].reserved == 0 and router.budgets["r"].spent == 0
        assert (await router.complete("r", MSG)).text == "A"
    run(go())


@pytest.mark.parametrize("status,rests", [(429, 30), (500, 10), (503, 10), (520, 10), (529, 10), (501, 0),
                                          (400, 0), (None, 0)])
def test_which_errors_rest_the_key(status, rests):
    router, _, clock = make_router({"a": [TransportError("x", status=status)], "b": [ok()]})
    run(router.complete("r", MSG))
    assert max(router.pools["p-a"].credentials[0].cooldown_until - clock(), 0) == rests


@pytest.mark.parametrize("bad", [Response("x", usage=None), Response("x", OK, reported_cost="0.1"),
                                 {"text": "x"}, Response(None, OK)])
def test_malformed_answers_are_handled(bad):
    seen = []
    router, _, _ = make_router({"a": [ok()], "b": [ok("B")]}, on_attempt=seen.append)
    honest = router.transports["p-b"]

    async def sloppy(spec, api_key, messages, room):   # a hand-written transport that gets its answer wrong
        return bad

    router.transports["p-a"] = sloppy
    router.transports["p-b"] = honest
    result = run(router.complete("r", MSG))
    assert seen and seen[-1].outcome == "ok"
    if isinstance(bad, dict):
        assert seen[0].outcome == "bad_response" and result.key == "b"
    elif bad.text is None or bad.text == "":
        assert seen[0].outcome == "empty"
    else:
        assert result.key == "a" and result.text == "x"
        assert result.request_cost >= 0


def test_only_one_probe_when_the_breaker_half_opens():
    async def go():
        clock = FakeClock()
        router, transport, _ = make_router({"a": [Delay(0.02, ok("A"))], "b": [ok("B")]}, clock=clock, threshold=1)
        router.breaker.failure("a")
        clock.advance(60)
        results = await asyncio.gather(*(router.complete("r", MSG) for _ in range(10)))
        assert sum(1 for k, _ in transport.calls if k == "a") == 1
        assert sorted(r.key for r in results) == ["a"] + ["b"] * 9
        assert "a" not in router.breaker.snapshot()
    run(go())


def test_an_unused_probe_is_handed_back():
    clock = FakeClock()
    router, transport, _ = make_router({"a": [ok("A")], "b": [ok("B")]}, chain=("b", "a"), clock=clock, threshold=1)
    router.breaker.failure("a")
    clock.advance(60)
    assert run(router.complete("r", MSG)).key == "b"      # `a` was admitted as a probe but never called
    router._chains["r"] = ["a", "b"]
    assert run(router.complete("r", MSG)).key == "a"      # so the next request may still probe it


def test_a_timeout_raised_by_the_transport_is_an_error_not_our_timeout():
    router, _, _ = make_router({"a": [TimeoutError("socket read timed out")], "b": [ok()]})
    result = run(router.complete("r", MSG))
    first = result.attempts[0]
    assert first.outcome == "error" and "socket read timed out" in first.error


def test_timeouts_and_errors_count_against_the_breaker_but_no_capacity_does_not():
    router, _, _ = make_router({"a": [Delay(1, ok())], "b": [ValueError("boom")], "c": [ok()]},
                               chain=("a", "b", "c"), a={"timeout": 0.01})
    result = run(router.complete("r", MSG))
    assert [a.outcome for a in result.attempts] == ["timeout", "error", "ok"]
    assert result.attempts[0].cost is None and result.attempts[1].cost is None
    assert router.breaker.snapshot() == {"a": {"failures": 1, "open": False}, "b": {"failures": 1, "open": False}}

    router2, _, _ = make_router({"a": [ok()], "b": [ok("B")]}, rpm=1)
    run(router2.complete("r", MSG))
    router2.pools["p-a"].max_wait = 0
    assert [a.outcome for a in run(router2.complete("r", MSG)).attempts] == ["no_capacity", "ok"]
    assert "a" not in router2.breaker.snapshot()


def test_failed_calls_are_not_charged():
    router, _, _ = make_router({"a": [TransportError("x", status=500)], "b": [TransportError("x", status=500)]},
                               cap=0.02)
    for _ in range(20):
        with pytest.raises(ChainExhausted):
            run(router.complete("r", MSG))
    assert router.budgets["r"].spent == 0 and router.budgets["r"].reserved == 0


def test_a_reported_cost_past_the_reservation_is_logged_and_recorded(caplog):
    seen = []
    pricey = Response("x", OK, reported_cost=0.5)   # the table says ~$0.0103 at most
    router, _, _ = make_router({"a": [pricey]}, chain=("a",), on_attempt=seen.append)
    with caplog.at_level("WARNING", logger="llm_router"):
        run(router.complete("r", MSG))
    assert seen[0].over_reserved == pytest.approx(0.5 - 0.010257)
    assert any("reserved" in r.getMessage() for r in caplog.records)
    router2, _, _ = make_router({"a": [ok()]}, chain=("a",), on_attempt=seen.append)
    run(router2.complete("r", MSG))
    assert seen[-1].over_reserved is None


def test_the_callers_output_room_is_used_and_reserved():
    router, transport, _ = make_router({"a": [ok()]}, chain=("a",), cap=0.0012)
    run(router.complete("r", MSG, max_output_tokens=50))   # worst case ~$0.00051: fits; 1024 tokens would not
    assert transport.calls == [("a", 50)]


def test_the_token_estimate_includes_the_output_room():
    assert estimate_tokens([{"role": "user", "content": "x" * 40}], 100) == 40 // 4 + 4 + 100


def test_the_first_truncated_text_is_the_one_kept():
    router, _, _ = make_router({"a": [Response("first", OK, finish="length")],
                                "b": [Response("second", OK, finish="length")],
                                "c": [TransportError("down", status=503)]}, chain=("a", "b", "c"))
    result = run(router.complete("r", MSG))
    assert (result.key, result.text, result.truncated) == ("a", "first", True)


def test_waits_add_up_and_never_pass_max_wait():
    clock = FakeClock()

    async def sleep(seconds):   # moves the clock and lets the other task run
        clock.advance(seconds)
        await asyncio.sleep(0)

    pool = KeyPool("p", [("k", "s")], rpm=1, max_wait=90, clock=clock, sleep=sleep)

    async def go():
        assert await pool.acquire(1) is not None
        # A waits 60 s for the refill; B arrives just then and takes it; A would need another 60 s: 120 > 90.
        return await asyncio.gather(pool.acquire(1), pool.acquire(1))
    a, b = run(go())
    assert a is None and b is not None
