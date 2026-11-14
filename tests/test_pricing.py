import pytest

from llm_router import ModelSpec, Response, Usage, price
from llm_router.pricing import cost_of, reported_cost, usage_from_anthropic, usage_from_openai, worst_case

SPEC = ModelSpec("m", "p", "m", input_per_m=2.0, output_per_m=8.0, cache_read_per_m=0.5)


def test_price_by_category_with_defaults():
    usage = Usage(input=1_000_000, cache_read=1_000_000, cache_write=1_000_000, output=1_000_000, reasoning=1_000_000)
    # input 2 + cache read 0.5 + cache write (defaults to input) 2 + output 8 + reasoning (defaults to output) 8
    assert price(SPEC, usage) == pytest.approx(20.5)


def test_openai_usage_with_cache_and_reasoning_details():
    u = {"prompt_tokens": 1000, "completion_tokens": 300, "total_tokens": 1300,
         "prompt_tokens_details": {"cached_tokens": 400}, "completion_tokens_details": {"reasoning_tokens": 200}}
    assert usage_from_openai(u) == Usage(input=600, cache_read=400, output=100, reasoning=200)


def test_openai_usage_with_thinking_only_in_the_total():
    u = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 400}
    assert usage_from_openai(u) == Usage(input=100, output=50, reasoning=250)


def test_openai_usage_tolerates_missing_and_odd_fields():
    assert usage_from_openai(None) == Usage()
    assert usage_from_openai({"prompt_tokens": None, "completion_tokens": "x"}) == Usage()
    # a cache count larger than the prompt is capped, never a negative input
    assert usage_from_openai({"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 50}}) == \
        Usage(input=0, cache_read=10)


def test_anthropic_usage():
    u = {"input_tokens": 50, "cache_read_input_tokens": 900, "cache_creation_input_tokens": 100, "output_tokens": 70}
    assert usage_from_anthropic(u) == Usage(input=50, cache_read=900, cache_write=100, output=70)


def test_reported_cost_wins_when_valid_including_zero():
    usage = Usage(input=1000)
    assert cost_of(SPEC, Response("x", usage, reported_cost=0.0)) == 0.0
    assert cost_of(SPEC, Response("x", usage, reported_cost=0.5)) == 0.5
    for bad in (float("nan"), -1.0):
        assert cost_of(SPEC, Response("x", usage, reported_cost=bad)) == pytest.approx(0.002)


def test_reported_cost_parsing():
    assert reported_cost({"cost": "0.25"}) == 0.25
    assert reported_cost({"cost": 0}) == 0.0
    assert reported_cost({"cost": "free"}) is None
    assert reported_cost({"cost": float("inf")}) is None
    assert reported_cost({}) is None


# worst_case by hand: one message of 300 characters -> 300 // 3 + 8 * 1 + 8 = 116 input tokens; room 100.
LONG = [{"role": "user", "content": "x" * 300}]


def test_worst_case_counts_the_input_and_the_output_room():
    spec = ModelSpec("m", "p", "m", input_per_m=2.0, output_per_m=10.0)
    assert worst_case(spec, LONG, 100) == pytest.approx((116 * 2.0 + 100 * 10.0) / 1e6)   # 0.001232


def test_worst_case_uses_the_reasoning_price_when_it_is_higher():
    spec = ModelSpec("m", "p", "m", input_per_m=2.0, output_per_m=10.0, reasoning_per_m=30.0)
    assert worst_case(spec, LONG, 100) == pytest.approx((116 * 2.0 + 100 * 30.0) / 1e6)   # 0.003232


def test_worst_case_uses_the_cache_write_price_when_it_is_higher():
    spec = ModelSpec("m", "p", "m", input_per_m=2.0, output_per_m=10.0, cache_write_per_m=5.0)
    assert worst_case(spec, LONG, 100) == pytest.approx((116 * 5.0 + 100 * 10.0) / 1e6)   # 0.00158
