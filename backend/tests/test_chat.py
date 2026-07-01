"""Tests for stream_chat — TDD: written BEFORE chat.py exists.

Uses a FAKE provider (scripted ProviderEvent lists, one list per
stream_turn() call, same style as tests/test_agent.py), a real
InMemoryConversationStore, and a real ToolRegistry with a throwaway
"echo" test tool. All test functions are async (asyncio_mode = "auto").
"""
import json

import pytest

from app.chat import stream_chat
from app.conversation import InMemoryConversationStore
from app.provider import TextDelta, ToolUseRequested, TurnComplete
from app.tools import Tool, ToolRegistry


# ---------------------------------------------------------------------------
# Fake provider (mirrors tests/test_agent.py's FakeProvider)
# ---------------------------------------------------------------------------


class FakeProvider:
    """Pops one scripted event-list (or raises) per stream_turn() call.

    Records the kwargs of every call for assertion.
    """

    def __init__(self, calls: list) -> None:
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
        if isinstance(events, Exception):
            raise events
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


def _parse_sse(chunks: list[str]) -> list[tuple[str, dict]]:
    """Parse yielded SSE wire strings into (event-name, data-dict) pairs."""
    parsed = []
    for chunk in chunks:
        lines = chunk.split("\n")
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        event_name = lines[0][len("event: "):]
        data = json.loads(lines[1][len("data: "):])
        parsed.append((event_name, data))
    return parsed


# ---------------------------------------------------------------------------
# Behavior 1: Simple turn
# ---------------------------------------------------------------------------


async def test_simple_turn_yields_token_then_done():
    provider = FakeProvider(
        [[TextDelta("Hi"), TurnComplete(stop_reason="end_turn", input_tokens=1, output_tokens=1)]]
    )
    registry = ToolRegistry()
    store = InMemoryConversationStore()

    chunks = [
        c
        async for c in stream_chat(
            provider=provider, registry=registry, store=store, session_id="s1", message="hello"
        )
    ]

    events = _parse_sse(chunks)
    assert events == [("token", {"text": "Hi"}), ("done", {})]


# ---------------------------------------------------------------------------
# Behavior 2: Always ends in exactly one done
# ---------------------------------------------------------------------------


async def test_always_ends_in_exactly_one_done():
    provider = FakeProvider(
        [
            [
                TextDelta("a"),
                TextDelta("b"),
                TurnComplete(stop_reason="end_turn", input_tokens=1, output_tokens=1),
            ]
        ]
    )
    registry = ToolRegistry()
    store = InMemoryConversationStore()

    chunks = [
        c
        async for c in stream_chat(
            provider=provider, registry=registry, store=store, session_id="s1", message="hello"
        )
    ]

    events = _parse_sse(chunks)
    done_events = [e for e in events if e[0] == "done"]
    assert len(done_events) == 1
    assert events[-1][0] == "done"


# ---------------------------------------------------------------------------
# Behavior 3: Tool path
# ---------------------------------------------------------------------------


async def test_tool_path_yields_tool_use_then_token_then_done():
    received_input = {}

    async def handler(tool_input: dict) -> str:
        received_input.update(tool_input)
        return "echo-result"

    registry = _make_registry(handler=handler)
    store = InMemoryConversationStore()
    provider = FakeProvider(
        [
            [
                ToolUseRequested(id="t1", name="echo", input={"v": 1}),
                TurnComplete(stop_reason="tool_use", input_tokens=1, output_tokens=1),
            ],
            [TextDelta("ok"), TurnComplete(stop_reason="end_turn", input_tokens=1, output_tokens=1)],
        ]
    )

    chunks = [
        c
        async for c in stream_chat(
            provider=provider, registry=registry, store=store, session_id="s1", message="use the tool"
        )
    ]

    events = _parse_sse(chunks)
    assert events == [
        ("tool_use", {"id": "t1", "name": "echo", "input": {"v": 1}}),
        ("token", {"text": "ok"}),
        ("done", {}),
    ]
    assert received_input == {"v": 1}

    # The persisted history must be the full, valid replay sequence for the
    # multi-step tool turn — not just a simple user+assistant pair. This
    # locks in that stream_chat persists the intermediate assistant
    # tool_use message and the user tool_result message produced by
    # run_turn (app/agent.py), in addition to the final assistant text.
    history = await store.get_history("s1")
    assert history == [
        {"role": "user", "content": "use the tool"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "echo", "input": {"v": 1}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "echo-result"}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
    ]


# ---------------------------------------------------------------------------
# Behavior 4: Persist + multi-turn
# ---------------------------------------------------------------------------


async def test_persists_turn_and_feeds_history_to_next_call():
    store = InMemoryConversationStore()
    registry = ToolRegistry()
    provider1 = FakeProvider(
        [[TextDelta("Hi"), TurnComplete(stop_reason="end_turn", input_tokens=1, output_tokens=1)]]
    )

    async for _ in stream_chat(
        provider=provider1, registry=registry, store=store, session_id="s1", message="hello"
    ):
        pass

    history = await store.get_history("s1")
    assert history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
    ]

    provider2 = FakeProvider(
        [[TextDelta("again"), TurnComplete(stop_reason="end_turn", input_tokens=1, output_tokens=1)]]
    )
    async for _ in stream_chat(
        provider=provider2, registry=registry, store=store, session_id="s1", message="second message"
    ):
        pass

    second_call_messages = provider2.received_calls[0]["messages"]
    assert second_call_messages[:2] == history


# ---------------------------------------------------------------------------
# Behavior 5: Empty-content guard
# ---------------------------------------------------------------------------


async def test_empty_content_assistant_message_not_persisted():
    provider = FakeProvider([[TurnComplete(stop_reason="end_turn", input_tokens=1, output_tokens=1)]])
    registry = ToolRegistry()
    store = InMemoryConversationStore()

    chunks = [
        c
        async for c in stream_chat(
            provider=provider, registry=registry, store=store, session_id="s1", message="hello"
        )
    ]

    events = _parse_sse(chunks)
    assert events[-1][0] == "done"

    history = await store.get_history("s1")
    assert history == [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# Behavior 6: Error path
# ---------------------------------------------------------------------------


async def test_error_during_turn_yields_single_error_event_and_persists_nothing():
    provider = FakeProvider([RuntimeError("boom")])
    registry = ToolRegistry()
    store = InMemoryConversationStore()

    chunks = [
        c
        async for c in stream_chat(
            provider=provider, registry=registry, store=store, session_id="s1", message="hello"
        )
    ]

    events = _parse_sse(chunks)
    assert events == [("error", {"type": "internal", "message": "An internal error occurred."})]

    history = await store.get_history("s1")
    assert history == []
