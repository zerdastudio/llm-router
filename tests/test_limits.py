import math

import pytest

from helpers import FakeClock, run

from llm_router import CircuitBreaker, KeyPool, RoleBudget, TokenBucket


def test_bucket_refills_continuously():
    clock = FakeClock()
    bucket = TokenBucket(60, 60, clock)   # one per second
    bucket.take(60)
    assert bucket.wait_for(1) == pytest.approx(1.0)
    clock.advance(30)
    assert bucket.wait_for(30) == 0
    assert bucket.wait_for(31) == pytest.approx(1.0)
    assert bucket.wait_for(61) == math.inf


def test_bucket_rejects_nonsense():
    with pytest.raises(ValueError):
        TokenBucket(0, 60, FakeClock())


def test_pool_waits_for_a_refill_within_the_limit():
    clock = FakeClock()
    pool = KeyPool("p", [("k", "s")], rpm=1, max_wait=60, clock=clock, sleep=clock.sleep)
    assert run(pool.acquire(10)) is not None
    start = clock()
    assert run(pool.acquire(10)) is not None
    assert clock() - start == pytest.approx(60)


def test_pool_gives_up_instead_of_waiting_too_long():
    clock = FakeClock()
    pool = KeyPool("p", [("k", "s")], rpd=1, max_wait=10, clock=clock, sleep=clock.sleep)
    assert run(pool.acquire(10)) is not None
    start = clock()
    assert run(pool.acquire(10)) is None    # the daily bucket refills far beyond the wait limit
    assert clock() == start


def test_pool_token_bucket_counts_tokens():
    clock = FakeClock()
    pool = KeyPool("p", [("k", "s")], tpm=1000, max_wait=0, clock=clock, sleep=clock.sleep)
    assert run(pool.acquire(800)) is not None
    assert run(pool.acquire(800)) is None
    assert run(pool.acquire(5000)) is None  # never fits


def test_pool_skips_cooling_keys_and_does_not_wait_for_them():
    clock = FakeClock()
    pool = KeyPool("p", [("k1", "s1"), ("k2", "s2")], clock=clock, sleep=clock.sleep)
    first = run(pool.acquire(1))
    pool.cooldown(first, 30)
    assert run(pool.acquire(1)).name == "k2"
    assert run(pool.acquire(1)).name == "k2"
    pool.cooldown(pool.credentials[1], 30)
    assert run(pool.acquire(1)) is None
    clock.advance(30)
    assert run(pool.acquire(1)) is not None


def test_pool_needs_keys():
    with pytest.raises(ValueError):
        KeyPool("p", [])


def test_secrets_stay_out_of_repr():
    pool = KeyPool("p", [("k", "very-secret")])
    assert "very-secret" not in repr(pool.credentials[0])


def test_breaker_opens_and_half_opens():
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=2, cooldown=10, clock=clock)
    breaker.failure("m")
    assert breaker.allows("m")
    breaker.failure("m")
    assert not breaker.allows("m")
    clock.advance(10)
    assert breaker.allows("m")          # one more chance
    breaker.failure("m")
    assert not breaker.allows("m")      # and it reopens at once
    clock.advance(10)
    assert breaker.allows("m")
    breaker.success("m")
    assert breaker.snapshot() == {}


def test_budget_window_and_validation():
    clock = FakeClock()
    budget = RoleBudget(1.0, 100, clock=clock)
    assert budget.resets_in() is None
    budget.settle(budget.reserve("r", 0.5), 0.2)
    clock.advance(40)
    budget.settle(budget.reserve("r", 0.5), 1.3)   # a second call inside the window does not move its start
    assert not budget.is_open()
    assert budget.resets_in() == pytest.approx(60)
    clock.advance(60)
    assert budget.is_open() and budget.spent == 0
    for bad in (0, -1, float("nan"), float("inf"), "1"):
        with pytest.raises(ValueError):
            RoleBudget(bad)
    with pytest.raises(ValueError):
        RoleBudget(1.0, 0)


def test_budget_without_window_never_reopens():
    clock = FakeClock()
    budget = RoleBudget(1.0, None, clock=clock)
    budget.settle(budget.reserve("r", 1.0), 2)
    clock.advance(10**9)
    assert not budget.is_open() and budget.resets_in() is None


def test_budget_ignores_bad_costs():
    budget = RoleBudget(1.0)
    for bad in (None, float("nan"), -5, 0):
        budget.add(bad)
    assert budget.spent == 0


def test_bucket_never_fills_past_capacity():
    clock = FakeClock()
    bucket = TokenBucket(10, 60, clock)
    clock.advance(10_000)
    bucket.take(10)
    assert bucket.wait_for(1) == pytest.approx(6.0)


def test_daily_bucket_refills_over_a_day_not_an_hour():
    clock = FakeClock()
    pool = KeyPool("p", [("k", "s")], rpd=24, max_wait=0, clock=clock, sleep=clock.sleep)
    for _ in range(24):
        assert run(pool.acquire(1)) is not None
    clock.advance(1800)
    assert run(pool.acquire(1)) is None
    clock.advance(1800)
    assert run(pool.acquire(1)) is not None


def test_pool_never_waits_past_max_wait():
    clock = FakeClock()
    pool = KeyPool("p", [("k", "s")], rpm=1, max_wait=59, clock=clock, sleep=clock.sleep)
    assert run(pool.acquire(1)) is not None
    assert run(pool.acquire(1)) is None and clock() == 1000.0


def test_reservations_are_held_and_released_once():
    budget = RoleBudget(1.0)
    r1 = budget.reserve("r", 0.6)
    with pytest.raises(Exception) as info:
        budget.reserve("r", 0.5)
    assert info.value.reserved == pytest.approx(0.6)
    budget.release(r1)
    budget.release(r1)
    assert budget.reserved == 0
    budget.settle(budget.reserve("r", 0.9), 0.3)
    assert (budget.spent, budget.reserved) == (pytest.approx(0.3), 0)
