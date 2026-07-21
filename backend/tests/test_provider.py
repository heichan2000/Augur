"""Tests for AnthropicProvider — TDD: written BEFORE provider.py exists.

All tests use fake Anthropic clients; no network calls are made.
"""
import types

import anthropic
import httpx
import pytest

from app.provider import (
    AnthropicProvider,
    ProviderError,
    ProviderRateLimitError,
    TextDelta,
    ToolUseRequested,
    TurnComplete,
)

MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Fake Anthropic stream infrastructure
# ---------------------------------------------------------------------------


class FakeAsyncIter:
    """Async-iterable over a plain list of fake events."""

    def __init__(self, events: list) -> None:
        self._events = events

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for event in self._events:
            yield event


class FakeStream:
    """Async context manager that yields a FakeAsyncIter."""

    def __init__(self, events: list) -> None:
        self._events = events

    async def __aenter__(self):
        return FakeAsyncIter(self._events)

    async def __aexit__(self, *args):
        pass


class FakeMessages:
    """Records kwargs passed to .stream() and returns a FakeStream."""

    def __init__(self, events: list) -> None:
        self._events = events
        self.last_kwargs: dict = {}

    def stream(self, **kwargs):
        self.last_kwargs = kwargs
        return FakeStream(self._events)


class FakeClient:
    def __init__(self, events: list) -> None:
        self.messages = FakeMessages(events)


# ---------------------------------------------------------------------------
# Event-shape helpers (SimpleNamespace)
# ---------------------------------------------------------------------------


def _message_start(input_tokens: int):
    return types.SimpleNamespace(
        type="message_start",
        message=types.SimpleNamespace(
            usage=types.SimpleNamespace(input_tokens=input_tokens)
        ),
    )


def _content_block_start_text(index: int):
    return types.SimpleNamespace(
        type="content_block_start",
        index=index,
        content_block=types.SimpleNamespace(type="text"),
    )


def _content_block_start_tool(index: int, id: str, name: str):
    return types.SimpleNamespace(
        type="content_block_start",
        index=index,
        content_block=types.SimpleNamespace(type="tool_use", id=id, name=name),
    )


def _text_delta(index: int, text: str):
    return types.SimpleNamespace(
        type="content_block_delta",
        index=index,
        delta=types.SimpleNamespace(type="text_delta", text=text),
    )


def _input_json_delta(index: int, partial_json: str):
    return types.SimpleNamespace(
        type="content_block_delta",
        index=index,
        delta=types.SimpleNamespace(type="input_json_delta", partial_json=partial_json),
    )


def _content_block_stop(index: int):
    return types.SimpleNamespace(type="content_block_stop", index=index)


def _message_delta(stop_reason: str, output_tokens: int):
    return types.SimpleNamespace(
        type="message_delta",
        delta=types.SimpleNamespace(stop_reason=stop_reason),
        usage=types.SimpleNamespace(output_tokens=output_tokens),
    )


def _message_stop():
    return types.SimpleNamespace(type="message_stop")


# ---------------------------------------------------------------------------
# Helper: collect all events from the async generator
# ---------------------------------------------------------------------------


async def collect(provider, **kwargs):
    return [event async for event in provider.stream_turn(**kwargs)]


def make_provider(client, *, max_tokens: int = 2048) -> AnthropicProvider:
    """Build a provider over a fake client.

    Centralised so the constructor can gain arguments without touching every
    test — 2048 matches the Settings default, so tests that do not care about
    the ceiling read as "the normal configuration".
    """
    return AnthropicProvider(client=client, model=MODEL, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Test 1: Text streaming
# ---------------------------------------------------------------------------


async def test_text_streaming_yields_text_deltas_and_turn_complete():
    events = [
        _message_start(input_tokens=5),
        _content_block_start_text(index=0),
        _text_delta(index=0, text="Hel"),
        _text_delta(index=0, text="lo"),
        _content_block_stop(index=0),
        _message_delta(stop_reason="end_turn", output_tokens=7),
        _message_stop(),
    ]
    client = FakeClient(events)
    provider = make_provider(client)

    results = await collect(provider, messages=[{"role": "user", "content": "hi"}])

    assert results == [
        TextDelta("Hel"),
        TextDelta("lo"),
        TurnComplete("end_turn", 5, 7),
    ]


# ---------------------------------------------------------------------------
# Test 2: Tool use with accumulated JSON
# ---------------------------------------------------------------------------


async def test_tool_use_yields_tool_use_requested_and_turn_complete():
    events = [
        _message_start(input_tokens=10),
        _content_block_start_tool(index=0, id="toolu_1", name="get_current_time"),
        _input_json_delta(index=0, partial_json='{"tz":'),
        _input_json_delta(index=0, partial_json='"utc"}'),
        _content_block_stop(index=0),
        _message_delta(stop_reason="tool_use", output_tokens=3),
        _message_stop(),
    ]
    client = FakeClient(events)
    provider = make_provider(client)

    results = await collect(provider, messages=[{"role": "user", "content": "what time?"}])

    assert results == [
        ToolUseRequested(id="toolu_1", name="get_current_time", input={"tz": "utc"}),
        TurnComplete("tool_use", 10, 3),
    ]


# ---------------------------------------------------------------------------
# Test 3: Empty tool input → parsed as {}
# ---------------------------------------------------------------------------


async def test_tool_use_with_no_input_json_delta_yields_empty_dict():
    events = [
        _message_start(input_tokens=8),
        _content_block_start_tool(index=0, id="toolu_2", name="ping"),
        # no input_json_delta events
        _content_block_stop(index=0),
        _message_delta(stop_reason="tool_use", output_tokens=2),
        _message_stop(),
    ]
    client = FakeClient(events)
    provider = make_provider(client)

    results = await collect(provider, messages=[{"role": "user", "content": "ping"}])

    assert results == [
        ToolUseRequested(id="toolu_2", name="ping", input={}),
        TurnComplete("tool_use", 8, 2),
    ]


# ---------------------------------------------------------------------------
# Test 4: Passthrough — kwargs forwarding
# ---------------------------------------------------------------------------


async def test_stream_kwargs_include_model_messages_system_and_tools():
    """Verify that stream() receives model, messages, system, and tools."""
    events = [
        _message_start(input_tokens=1),
        _message_delta(stop_reason="end_turn", output_tokens=1),
        _message_stop(),
    ]
    client = FakeClient(events)
    provider = make_provider(client)
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"name": "search", "description": "search", "input_schema": {"type": "object"}}]

    await collect(
        provider,
        messages=messages,
        system="sys prompt",
        tools=tools,
    )

    kwargs = client.messages.last_kwargs
    assert kwargs["model"] == MODEL
    assert kwargs["messages"] == messages
    assert kwargs["system"] == "sys prompt"
    assert kwargs["tools"] == tools


async def test_stream_kwargs_omit_system_and_tools_when_not_provided():
    """system and tools must be ABSENT from kwargs when not passed."""
    events = [
        _message_start(input_tokens=1),
        _message_delta(stop_reason="end_turn", output_tokens=1),
        _message_stop(),
    ]
    client = FakeClient(events)
    provider = make_provider(client)

    await collect(
        provider,
        messages=[{"role": "user", "content": "hello"}],
    )

    kwargs = client.messages.last_kwargs
    assert "system" not in kwargs
    assert "tools" not in kwargs


# ---------------------------------------------------------------------------
# Fake client that raises when .stream() is called
# ---------------------------------------------------------------------------


class FakeRaisingMessages:
    """Raises *exc* the moment .stream() is called (mirrors the real SDK,
    which raises from opening/iterating the stream context)."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def stream(self, **kwargs):
        raise self._exc


class FakeRaisingClient:
    def __init__(self, exc: Exception) -> None:
        self.messages = FakeRaisingMessages(exc)


# ---------------------------------------------------------------------------
# Test 5: Rate-limit maps to ProviderRateLimitError
# ---------------------------------------------------------------------------


async def test_rate_limit_error_maps_to_provider_rate_limit_error():
    req = httpx.Request("POST", "https://api.anthropic.com")
    resp = httpx.Response(429, request=req)
    exc = anthropic.RateLimitError("rate limited", response=resp, body=None)
    client = FakeRaisingClient(exc)
    provider = make_provider(client)

    with pytest.raises(ProviderRateLimitError):
        [e async for e in provider.stream_turn(messages=[{"role": "user", "content": "hi"}])]


# ---------------------------------------------------------------------------
# Test 6: Other API errors map to ProviderError (not ProviderRateLimitError)
# ---------------------------------------------------------------------------


async def test_api_connection_error_maps_to_provider_error():
    req = httpx.Request("POST", "https://api.anthropic.com")
    exc = anthropic.APIConnectionError(message="boom", request=req)
    client = FakeRaisingClient(exc)
    provider = make_provider(client)

    with pytest.raises(ProviderError) as excinfo:
        [e async for e in provider.stream_turn(messages=[{"role": "user", "content": "hi"}])]
    assert not isinstance(excinfo.value, ProviderRateLimitError)


# ---------------------------------------------------------------------------
# Test 7: Non-anthropic exceptions propagate unchanged
# ---------------------------------------------------------------------------


async def test_non_anthropic_exception_propagates_unchanged():
    client = FakeRaisingClient(RuntimeError("bug"))
    provider = make_provider(client)

    with pytest.raises(RuntimeError):
        [e async for e in provider.stream_turn(messages=[{"role": "user", "content": "hi"}])]


async def test_max_tokens_from_constructor_appears_in_stream_kwargs():
    events = [
        _message_start(input_tokens=1),
        _message_delta(stop_reason="end_turn", output_tokens=1),
        _message_stop(),
    ]
    client = FakeClient(events)
    provider = make_provider(client, max_tokens=4096)

    await collect(provider, messages=[{"role": "user", "content": "hello"}])

    assert client.messages.last_kwargs["max_tokens"] == 4096


# ---------------------------------------------------------------------------
# Truncation: a turn cut off by the token ceiling drops its tool blocks
# ---------------------------------------------------------------------------


async def test_truncated_mid_input_json_drops_the_tool_block():
    """A buffer holding half a JSON object must not raise, and must not be
    dispatched. Before the guard this raised JSONDecodeError out of the
    generator, which stream_chat turned into an `internal` error with no
    `done` and nothing persisted."""
    events = [
        _message_start(input_tokens=9),
        _content_block_start_tool(index=0, id="toolu_3", name="get_current_time"),
        _input_json_delta(index=0, partial_json='{"tz":'),
        _content_block_stop(index=0),
        _message_delta(stop_reason="max_tokens", output_tokens=4),
        _message_stop(),
    ]
    client = FakeClient(events)
    provider = make_provider(client)

    results = await collect(provider, messages=[{"role": "user", "content": "what time?"}])

    assert results == [TurnComplete("max_tokens", 9, 4)]


async def test_truncated_with_empty_tool_buffer_dispatches_nothing():
    """Truncated before any input_json_delta arrived. An empty buffer would
    otherwise parse to {} and dispatch a real tool with silently-empty
    input."""
    events = [
        _message_start(input_tokens=8),
        _content_block_start_tool(index=0, id="toolu_4", name="ping"),
        _content_block_stop(index=0),
        _message_delta(stop_reason="max_tokens", output_tokens=2),
        _message_stop(),
    ]
    client = FakeClient(events)
    provider = make_provider(client)

    results = await collect(provider, messages=[{"role": "user", "content": "ping"}])

    assert results == [TurnComplete("max_tokens", 8, 2)]


async def test_context_window_exceeded_drops_the_tool_block_too():
    events = [
        _message_start(input_tokens=8),
        _content_block_start_tool(index=0, id="toolu_5", name="ping"),
        _content_block_stop(index=0),
        _message_delta(stop_reason="model_context_window_exceeded", output_tokens=2),
        _message_stop(),
    ]
    client = FakeClient(events)
    provider = make_provider(client)

    results = await collect(provider, messages=[{"role": "user", "content": "ping"}])

    assert results == [TurnComplete("model_context_window_exceeded", 8, 2)]


async def test_streamed_text_survives_a_truncated_tool_block():
    """Text streamed before the cut-off is the real answer and must reach the
    caller — it is what gets persisted and shown."""
    events = [
        _message_start(input_tokens=9),
        _content_block_start_text(index=0),
        _text_delta(index=0, text="Let me check"),
        _content_block_stop(index=0),
        _content_block_start_tool(index=1, id="toolu_6", name="get_current_time"),
        _input_json_delta(index=1, partial_json='{"tz":'),
        _content_block_stop(index=1),
        _message_delta(stop_reason="max_tokens", output_tokens=4),
        _message_stop(),
    ]
    client = FakeClient(events)
    provider = make_provider(client)

    results = await collect(provider, messages=[{"role": "user", "content": "what time?"}])

    assert results == [TextDelta("Let me check"), TurnComplete("max_tokens", 9, 4)]


async def test_unparseable_buffer_is_dropped_and_the_turn_completes_normally():
    """Malformed JSON without truncation: drop the one block, keep the turn."""
    events = [
        _message_start(input_tokens=9),
        _content_block_start_tool(index=0, id="toolu_7", name="get_current_time"),
        _input_json_delta(index=0, partial_json="not json at all"),
        _content_block_stop(index=0),
        _message_delta(stop_reason="tool_use", output_tokens=4),
        _message_stop(),
    ]
    client = FakeClient(events)
    provider = make_provider(client)

    results = await collect(provider, messages=[{"role": "user", "content": "what time?"}])

    assert results == [TurnComplete("tool_use", 9, 4)]
