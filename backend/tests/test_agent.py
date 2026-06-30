"""Tests for run_turn — TDD: written BEFORE agent.py exists.

Uses a FAKE provider (scripted ProviderEvent lists, one list per
stream_turn call) and a real ToolRegistry with simple async handlers.
All test functions are async (asyncio_mode = "auto" handles the event loop).
"""
import pytest

from app.agent import run_turn
from app.provider import TextDelta, ToolUseRequested, TurnComplete
from app.tools import Tool, ToolRegistry


# ---------------------------------------------------------------------------
# Fake provider
# ---------------------------------------------------------------------------


class FakeProvider:
    """Pops one scripted event-list per stream_turn() call.

    Records the kwargs of every call for assertion.
    """

    def __init__(self, calls: list[list]) -> None:
        self._calls = list(calls)
        self.received_calls: list[dict] = []

    async def stream_turn(self, *, messages, system=None, tools=None, max_tokens=2048):
        self.received_calls.append(
            {
                "messages": [dict(m) for m in messages],  # snapshot at call time
                "system": system,
                "tools": tools,
                "max_tokens": max_tokens,
            }
        )
        events = self._calls.pop(0)
        for event in events:
            yield event


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _echo_handler(tool_input: dict) -> str:
    return f"echo:{tool_input}"


def _make_registry(handler=_echo_handler) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="echo",
            description="Echoes the input back.",
            input_schema={"type": "object", "properties": {}},
            handler=handler,
        )
    )
    return registry


# ---------------------------------------------------------------------------
# Test 1: No-tool turn
# ---------------------------------------------------------------------------


async def test_no_tool_turn_yields_text_then_turn_complete_and_appends_one_message():
    provider = FakeProvider(
        [
            [TextDelta("hi"), TurnComplete(stop_reason="end_turn", input_tokens=3, output_tokens=2)],
        ]
    )
    registry = ToolRegistry()
    messages = [{"role": "user", "content": "hello"}]

    events = [event async for event in run_turn(provider=provider, registry=registry, messages=messages)]

    assert events == [
        TextDelta("hi"),
        TurnComplete(stop_reason="end_turn", input_tokens=3, output_tokens=2),
    ]
    assert messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ]
    assert len(provider.received_calls) == 1


# ---------------------------------------------------------------------------
# Test 2: Single tool round-trip
# ---------------------------------------------------------------------------


async def test_single_tool_round_trip_dispatches_and_reinvokes_provider():
    received_input = {}

    async def handler(tool_input: dict) -> str:
        received_input.update(tool_input)
        return "echo-result"

    registry = _make_registry(handler=handler)
    provider = FakeProvider(
        [
            [
                ToolUseRequested(id="t1", name="echo", input={"v": 1}),
                TurnComplete(stop_reason="tool_use", input_tokens=5, output_tokens=4),
            ],
            [
                TextDelta("done"),
                TurnComplete(stop_reason="end_turn", input_tokens=6, output_tokens=2),
            ],
        ]
    )
    messages = [{"role": "user", "content": "use the tool"}]

    events = [event async for event in run_turn(provider=provider, registry=registry, messages=messages)]

    assert received_input == {"v": 1}
    assert len(provider.received_calls) == 2

    assert messages == [
        {"role": "user", "content": "use the tool"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "echo", "input": {"v": 1}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "echo-result"}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]

    # Second provider call received the grown messages (everything up to that point).
    second_call_messages = provider.received_calls[1]["messages"]
    assert second_call_messages == messages[:3]

    final = events[-1]
    assert isinstance(final, TurnComplete)
    assert final.stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# Test 3: Tool schemas forwarded
# ---------------------------------------------------------------------------


async def test_tool_schemas_forwarded_when_registry_non_empty():
    registry = _make_registry()
    provider = FakeProvider(
        [[TurnComplete(stop_reason="end_turn", input_tokens=1, output_tokens=1)]]
    )
    messages = [{"role": "user", "content": "hi"}]

    async for _ in run_turn(provider=provider, registry=registry, messages=messages):
        pass

    assert provider.received_calls[0]["tools"] == registry.schemas()


async def test_tools_none_when_registry_empty():
    registry = ToolRegistry()
    provider = FakeProvider(
        [[TurnComplete(stop_reason="end_turn", input_tokens=1, output_tokens=1)]]
    )
    messages = [{"role": "user", "content": "hi"}]

    async for _ in run_turn(provider=provider, registry=registry, messages=messages):
        pass

    assert provider.received_calls[0]["tools"] is None


# ---------------------------------------------------------------------------
# Test 4: tool_use notice passes through
# ---------------------------------------------------------------------------


async def test_tool_use_requested_passes_through_to_caller():
    registry = _make_registry()
    provider = FakeProvider(
        [
            [
                ToolUseRequested(id="t1", name="echo", input={}),
                TurnComplete(stop_reason="tool_use", input_tokens=1, output_tokens=1),
            ],
            [TurnComplete(stop_reason="end_turn", input_tokens=1, output_tokens=1)],
        ]
    )
    messages = [{"role": "user", "content": "hi"}]

    events = [event async for event in run_turn(provider=provider, registry=registry, messages=messages)]

    assert ToolUseRequested(id="t1", name="echo", input={}) in events[:-1]


# ---------------------------------------------------------------------------
# Test 5: Max-steps bound
# ---------------------------------------------------------------------------


async def test_max_steps_bound_terminates_without_hang_or_raise():
    dispatch_count = 0

    async def handler(tool_input: dict) -> str:
        nonlocal dispatch_count
        dispatch_count += 1
        return "echo-result"

    def always_tool_call_list():
        return [
            ToolUseRequested(id="loop", name="echo", input={}),
            TurnComplete(stop_reason="tool_use", input_tokens=1, output_tokens=1),
        ]

    provider = FakeProvider([always_tool_call_list() for _ in range(10)])
    registry = _make_registry(handler=handler)
    messages = [{"role": "user", "content": "loop forever"}]

    events = [
        event
        async for event in run_turn(provider=provider, registry=registry, messages=messages, max_steps=3)
    ]

    assert len(provider.received_calls) == 3
    final = events[-1]
    assert isinstance(final, TurnComplete)
    assert final.stop_reason == "tool_use"

    # The bound-hitting step must NOT dispatch a tool or append a tool_result —
    # the loop stops right after appending the last assistant message.
    assert dispatch_count == 2
    assert messages[-1] == {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "loop", "name": "echo", "input": {}}],
    }


# ---------------------------------------------------------------------------
# Test 6: Token totals summed
# ---------------------------------------------------------------------------


async def test_token_totals_summed_across_steps():
    registry = _make_registry()
    provider = FakeProvider(
        [
            [
                ToolUseRequested(id="t1", name="echo", input={}),
                TurnComplete(stop_reason="tool_use", input_tokens=10, output_tokens=5),
            ],
            [
                TextDelta("done"),
                TurnComplete(stop_reason="end_turn", input_tokens=7, output_tokens=3),
            ],
        ]
    )
    messages = [{"role": "user", "content": "hi"}]

    events = [event async for event in run_turn(provider=provider, registry=registry, messages=messages)]

    final = events[-1]
    assert isinstance(final, TurnComplete)
    assert final.input_tokens == 17
    assert final.output_tokens == 8
