"""The real transports, against stand-in SDK clients: how replies and errors are read. No SDK, no network."""
from types import SimpleNamespace as NS

import pytest

from helpers import run

from llm_router import AnthropicTransport, ModelSpec, OpenAICompatTransport, TransportError, Usage

SPEC = ModelSpec("k", "p", "model-x")
MSG = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]


class StatusError(Exception):
    def __init__(self, status, retry_after=None):
        super().__init__(f"status {status}")
        self.status_code = status
        self.response = NS(headers={"retry-after": retry_after} if retry_after is not None else {})


class FakeCreate:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.kwargs = reply, error, None

    async def __call__(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.reply


def openai_with(create, **kw):
    client = NS(chat=NS(completions=NS(create=create)))
    made = []

    def factory(key, base_url):
        made.append((key, base_url))
        return client

    return OpenAICompatTransport("https://example.invalid/v1", client_factory=factory, **kw), made


def openai_reply(content="hello", finish="stop", refusal=None, usage=None):
    return NS(choices=[NS(message=NS(content=content, refusal=refusal), finish_reason=finish)], usage=usage)


def test_openai_reply_text_usage_and_parameter_name():
    create = FakeCreate(openai_reply(usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}))
    transport, made = openai_with(create, max_tokens_param="max_completion_tokens")
    r = run(transport(SPEC, "key-1", MSG, 77))
    assert (r.text, r.finish, r.usage) == ("hello", "stop", Usage(input=10, output=5))
    assert create.kwargs == {"model": "model-x", "messages": MSG, "max_completion_tokens": 77}
    run(transport(SPEC, "key-1", MSG, 77))
    assert made == [("key-1", "https://example.invalid/v1")]   # one client per key, reused


def test_openai_finish_reasons():
    t, _ = openai_with(FakeCreate(openai_reply("cut", finish="length")))
    assert run(t(SPEC, "k", MSG, 5)).finish == "length"
    t, _ = openai_with(FakeCreate(openai_reply(None, refusal="I can't help with that")))
    r = run(t(SPEC, "k", MSG, 5))
    assert (r.text, r.finish) == ("", "refusal")
    t, _ = openai_with(FakeCreate(NS(choices=[], usage=None)))
    assert run(t(SPEC, "k", MSG, 5)).text == ""


def test_openai_reported_cost_only_when_asked():
    usage = NS(model_dump=lambda: {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0042})
    t, _ = openai_with(FakeCreate(openai_reply(usage=usage)), use_reported_cost=True)
    assert run(t(SPEC, "k", MSG, 5)).reported_cost == 0.0042
    t, _ = openai_with(FakeCreate(openai_reply(usage=usage)))
    assert run(t(SPEC, "k", MSG, 5)).reported_cost is None


def test_openai_status_error_becomes_transport_error_with_retry_after():
    t, _ = openai_with(FakeCreate(error=StatusError(429, "7")))
    with pytest.raises(TransportError) as info:
        run(t(SPEC, "k", MSG, 5))
    assert (info.value.status, info.value.retry_after) == (429, 7.0)


def test_other_errors_pass_through():
    t, _ = openai_with(FakeCreate(error=ConnectionResetError("network")))
    with pytest.raises(ConnectionResetError):
        run(t(SPEC, "k", MSG, 5))


def test_anthropic_moves_system_and_reads_blocks():
    reply = NS(content=[NS(type="thinking", text="hmm"), NS(type="text", text="Hel"), NS(type="text", text="lo")],
               stop_reason="max_tokens",
               usage={"input_tokens": 3, "cache_read_input_tokens": 9, "output_tokens": 4})
    create = FakeCreate(reply)
    t = AnthropicTransport(client_factory=lambda key: NS(messages=NS(create=create)))
    r = run(t(SPEC, "k", MSG, 64))
    assert (r.text, r.finish, r.usage) == ("Hello", "length", Usage(input=3, cache_read=9, output=4))
    assert create.kwargs == {"model": "model-x", "messages": [MSG[1]], "max_tokens": 64, "system": "be brief"}


def test_anthropic_refusal_and_status_error():
    create = FakeCreate(NS(content=[], stop_reason="refusal", usage=None))
    t = AnthropicTransport(client_factory=lambda key: NS(messages=NS(create=create)))
    assert run(t(SPEC, "k", MSG[1:], 8)).finish == "refusal"
    assert "system" not in create.kwargs
    bad = AnthropicTransport(client_factory=lambda key: NS(messages=NS(create=FakeCreate(error=StatusError(529)))))
    with pytest.raises(TransportError) as info:
        run(bad(SPEC, "k", MSG, 8))
    assert info.value.status == 529
