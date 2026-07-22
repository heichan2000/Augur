"""Tests for stream_chat — TDD: written BEFORE chat.py exists.

Uses a FAKE provider (scripted ProviderEvent lists, one list per
stream_turn() call, same style as tests/test_agent.py), a real
InMemoryConversationStore, and a real ToolRegistry with a throwaway
"echo" test tool. All test functions are async (asyncio_mode = "auto").
"""
import asyncio
import contextlib
import json
import logging

import pytest

from app.chat import stream_chat
from app.config import AGENT_MAX_STEPS
from app.conversation import InMemoryConversationStore
from app.observability import compute_cost
from app.provider import (
    ProviderError,
    ProviderRateLimitError,
    TextDelta,
    ToolUseRequested,
    TurnComplete,
)
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


async def _fixed_result_handler(tool_input: dict) -> str:
    """A tool whose result is the same every call, for exact assertions."""
    return "echo-result"


async def _stop_after(stream, count: int = 1) -> list[str]:
    """Consume *count* events, then go away — the client pressing Stop.

    Closing the generator is what a dropped connection looks like when the
    turn is parked at a yield nobody will read. The other shape — cancelled
    while suspended in its own await — is driven directly by the tests that
    need it, since only a task cancel produces it.
    """
    seen = []
    async for chunk in stream:
        seen.append(chunk)
        if len(seen) == count:
            break
    await stream.aclose()
    return seen


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
            provider=provider, registry=registry, store=store, session_id="s1", message="hello", model="claude-sonnet-4-6"
        )
    ]

    events = _parse_sse(chunks)
    assert events == [("token", {"text": "Hi"}), ("done", {"stop_reason": "end_turn"})]


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
            provider=provider, registry=registry, store=store, session_id="s1", message="hello", model="claude-sonnet-4-6"
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
            provider=provider, registry=registry, store=store, session_id="s1", message="use the tool", model="claude-sonnet-4-6"
        )
    ]

    events = _parse_sse(chunks)
    assert events == [
        ("tool_use", {"id": "t1", "name": "echo", "input": {"v": 1}}),
        ("token", {"text": "ok"}),
        ("done", {"stop_reason": "end_turn"}),
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
        provider=provider1, registry=registry, store=store, session_id="s1", message="hello", model="claude-sonnet-4-6"
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
        provider=provider2, registry=registry, store=store, session_id="s1", message="second message", model="claude-sonnet-4-6"
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
            provider=provider, registry=registry, store=store, session_id="s1", message="hello", model="claude-sonnet-4-6"
        )
    ]

    events = _parse_sse(chunks)
    assert events[-1][0] == "done"

    # The model returned nothing, so the assistant message is empty and is
    # not stored — an empty assistant turn is not a message the API would
    # accept back. The user's message stays: the turn completed, and a
    # completed turn is not the place to second-guess whether the exchange
    # happened. (A *stopped* turn with no answer does drop it — see
    # test_stopping_before_any_text_persists_nothing.)
    history = await store.get_history("s1")
    assert history == [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# Behavior 5b: A turn that exhausts the step bound still leaves valid history
# ---------------------------------------------------------------------------


async def test_turn_that_exhausts_step_bound_persists_valid_replay_sequence():
    # Every round requests a tool, so run_turn hits max_steps with tools
    # still pending and the last message it appends is an assistant
    # tool_use with no tool_result to answer it. That message must not
    # reach stored history — what we persist has to stay a sequence we
    # could replay to the model.
    registry = _make_registry(handler=_fixed_result_handler)
    store = InMemoryConversationStore()
    provider = FakeProvider(
        [
            [
                ToolUseRequested(id=f"t{i}", name="echo", input={"round": i}),
                TurnComplete(stop_reason="tool_use", input_tokens=1, output_tokens=1),
            ]
            for i in range(AGENT_MAX_STEPS)
        ]
    )

    chunks = [
        c
        async for c in stream_chat(
            provider=provider,
            registry=registry,
            store=store,
            session_id="s1",
            message="loop forever",
            model="claude-sonnet-4-6",
        )
    ]

    events = _parse_sse(chunks)
    assert events[-1][0] == "done"

    history = await store.get_history("s1")

    # Every round but the last completed with its results; the final
    # round's unanswered tool_use is dropped, so history ends on the
    # preceding tool_result.
    expected: list[dict] = [{"role": "user", "content": "loop forever"}]
    for i in range(AGENT_MAX_STEPS - 1):
        expected.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"t{i}", "name": "echo", "input": {"round": i}}
                ],
            }
        )
        expected.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"t{i}", "content": "echo-result"}
                ],
            }
        )
    assert history == expected


# ---------------------------------------------------------------------------
# Behavior 5c: An unanswered tool call costs only itself, not the answer
# ---------------------------------------------------------------------------


async def test_answer_survives_alongside_an_unanswered_tool_call():
    # Same exhausted bound, except the final round also streams text. The
    # user read that text, so it belongs in history; only the tool call it
    # sits beside is unanswerable. Dropping the whole message would lose an
    # answer the model can no longer recall.
    registry = _make_registry(handler=_fixed_result_handler)
    store = InMemoryConversationStore()
    rounds: list[list] = [
        [
            ToolUseRequested(id=f"t{i}", name="echo", input={"round": i}),
            TurnComplete(stop_reason="tool_use", input_tokens=1, output_tokens=1),
        ]
        for i in range(AGENT_MAX_STEPS - 1)
    ]
    rounds.append(
        [
            TextDelta("Here is what I found so far."),
            ToolUseRequested(id="t-last", name="echo", input={"round": "last"}),
            TurnComplete(stop_reason="tool_use", input_tokens=1, output_tokens=1),
        ]
    )
    provider = FakeProvider(rounds)

    chunks = [
        c
        async for c in stream_chat(
            provider=provider,
            registry=registry,
            store=store,
            session_id="s1",
            message="loop forever",
            model="claude-sonnet-4-6",
        )
    ]

    events = _parse_sse(chunks)
    assert events[-1][0] == "done"

    history = await store.get_history("s1")

    assert history[-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Here is what I found so far."}],
    }
    # Nothing anywhere in history may request a tool that never came back.
    persisted_calls = {
        block["id"]
        for message in history
        if isinstance(message["content"], list)
        for block in message["content"]
        if block.get("type") == "tool_use"
    }
    assert "t-last" not in persisted_calls
    assert persisted_calls == {f"t{i}" for i in range(AGENT_MAX_STEPS - 1)}


# ---------------------------------------------------------------------------
# Behavior 5d: A stopped turn keeps the answer the user already read
# ---------------------------------------------------------------------------


async def test_stopping_mid_answer_persists_the_partial_text():
    provider = FakeProvider(
        [
            [
                TextDelta("Half an "),
                TextDelta("answer"),
                TurnComplete(stop_reason="end_turn", input_tokens=3, output_tokens=5),
            ]
        ]
    )
    registry = ToolRegistry()
    store = InMemoryConversationStore()

    stream = stream_chat(
        provider=provider,
        registry=registry,
        store=store,
        session_id="s1",
        message="tell me",
        model="claude-sonnet-4-6",
    )
    seen = await _stop_after(stream)

    assert _parse_sse(seen) == [("token", {"text": "Half an "})]

    history = await store.get_history("s1")
    assert history == [
        {"role": "user", "content": "tell me"},
        {"role": "assistant", "content": [{"type": "text", "text": "Half an "}]},
    ]


async def test_stopping_before_any_text_persists_nothing():
    # Stopped while the first tool call was still running, so the turn
    # never produced an answer. Leaving the user's message behind on its
    # own would put a question in history that nothing answers — and cost
    # them Retry, which is withheld once a turn is persisted.
    registry = _make_registry(handler=_fixed_result_handler)
    store = InMemoryConversationStore()
    provider = FakeProvider(
        [
            [
                ToolUseRequested(id="t1", name="echo", input={}),
                TurnComplete(stop_reason="tool_use", input_tokens=2, output_tokens=1),
            ]
        ]
    )

    stream = stream_chat(
        provider=provider,
        registry=registry,
        store=store,
        session_id="s1",
        message="use the tool",
        model="claude-sonnet-4-6",
    )
    seen = await _stop_after(stream)

    assert _parse_sse(seen) == [("tool_use", {"id": "t1", "name": "echo", "input": {}})]

    history = await store.get_history("s1")
    assert history == []


async def test_stopping_while_a_tool_runs_drops_the_call_and_keeps_the_text():
    # The only way a stopped turn can carry an unanswered call: run_turn
    # commits the assistant message the moment a round's stream ends, then
    # awaits the tool before it can append the result. Cancelling in that
    # window leaves a tool_use in messages that nothing will ever answer.
    # The call cannot be replayed, but the text was on screen — so the call
    # goes and the answer stays.
    dispatching = asyncio.Event()

    async def handler(tool_input: dict) -> str:
        dispatching.set()
        await asyncio.sleep(10)  # cancelled here, mid-dispatch
        return "never returned"

    registry = _make_registry(handler=handler)
    store = InMemoryConversationStore()
    provider = FakeProvider(
        [
            [
                TextDelta("Looking that up."),
                ToolUseRequested(id="t1", name="echo", input={}),
                TurnComplete(stop_reason="tool_use", input_tokens=2, output_tokens=2),
            ]
        ]
    )

    async def consume() -> None:
        async for _ in stream_chat(
            provider=provider,
            registry=registry,
            store=store,
            session_id="s1",
            message="look it up",
            model="claude-sonnet-4-6",
        ):
            pass

    task = asyncio.create_task(consume())
    await dispatching.wait()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    history = await store.get_history("s1")
    assert history == [
        {"role": "user", "content": "look it up"},
        {"role": "assistant", "content": [{"type": "text", "text": "Looking that up."}]},
    ]


async def test_stopped_turn_persists_through_a_store_that_suspends():
    # The stopped path writes while a cancellation is unwinding. The
    # Phase-1 in-memory store never suspends, so the write always finishes
    # before anything can interrupt it — the guarantee is held by the
    # store, not by the code. A Phase-2 store suspends on every append,
    # and an await that suspends mid-unwind is cancelled by the *next*
    # cancellation, silently dropping the turn. (One cancel is delivered
    # once; it takes a second one — a shutdown timeout on top of the
    # client hangup — to interrupt the write itself.)
    #
    # So: a store that really suspends, and a second cancel aimed at the
    # window while it is suspended. Without the shield the write is lost
    # and this test fails.
    writing = asyncio.Event()

    class SuspendingStore(InMemoryConversationStore):
        async def append(self, session_id: str, message: dict) -> None:
            writing.set()
            await asyncio.sleep(0.05)  # a real await point, as a DB store has
            await super().append(session_id, message)

    dispatching = asyncio.Event()

    async def handler(tool_input: dict) -> str:
        dispatching.set()
        await asyncio.sleep(10)  # cancelled here, mid-dispatch
        return "never returned"

    registry = _make_registry(handler=handler)
    store = SuspendingStore()
    provider = FakeProvider(
        [
            [
                TextDelta("Partial answer."),
                ToolUseRequested(id="t1", name="echo", input={}),
                TurnComplete(stop_reason="tool_use", input_tokens=2, output_tokens=2),
            ]
        ]
    )

    async def consume() -> None:
        async for _ in stream_chat(
            provider=provider,
            registry=registry,
            store=store,
            session_id="s1",
            message="look it up",
            model="claude-sonnet-4-6",
        ):
            pass

    task = asyncio.create_task(consume())
    await dispatching.wait()
    task.cancel()  # the client hangs up

    # Wait until the stopped-path write is actually suspended, then cancel
    # again — this is the shutdown-timeout cancel that would kill an
    # unshielded write.
    await writing.wait()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    # The shielded write outlives the task, so let it finish landing.
    await asyncio.sleep(0.2)

    history = await store.get_history("s1")
    assert history == [
        {"role": "user", "content": "look it up"},
        {"role": "assistant", "content": [{"type": "text", "text": "Partial answer."}]},
    ]


async def test_assistant_can_refer_back_to_a_stopped_answer():
    store = InMemoryConversationStore()
    registry = ToolRegistry()
    stopped = FakeProvider(
        [
            [
                TextDelta("Point one is cost."),
                TextDelta(" Point two is speed."),
                TurnComplete(stop_reason="end_turn", input_tokens=4, output_tokens=6),
            ]
        ]
    )

    stream = stream_chat(
        provider=stopped,
        registry=registry,
        store=store,
        session_id="s1",
        message="list the tradeoffs",
        model="claude-sonnet-4-6",
    )
    await _stop_after(stream)

    follow_up = FakeProvider(
        [[TextDelta("Cost means..."), TurnComplete(stop_reason="end_turn", input_tokens=1, output_tokens=1)]]
    )
    async for _ in stream_chat(
        provider=follow_up,
        registry=registry,
        store=store,
        session_id="s1",
        message="expand on that last point",
        model="claude-sonnet-4-6",
    ):
        pass

    # The stopped turn is in the history the model is asked to continue
    # from, so "that last point" refers to something it can actually see.
    sent = follow_up.received_calls[0]["messages"]
    assert sent[:3] == [
        {"role": "user", "content": "list the tradeoffs"},
        {"role": "assistant", "content": [{"type": "text", "text": "Point one is cost."}]},
        {"role": "user", "content": "expand on that last point"},
    ]


async def test_stopped_turn_logs_the_rounds_already_billed(caplog):
    # A provider reports usage in its closing event, so a round cut off
    # part-way has nothing to report yet. Rounds that already finished do:
    # stopping in round three still costs money for rounds one and two, and
    # that is what has to reach the cost log.
    registry = _make_registry(handler=_fixed_result_handler)
    store = InMemoryConversationStore()
    provider = FakeProvider(
        [
            [
                ToolUseRequested(id="t1", name="echo", input={}),
                TurnComplete(stop_reason="tool_use", input_tokens=5, output_tokens=1),
            ],
            [
                ToolUseRequested(id="t2", name="echo", input={}),
                TurnComplete(stop_reason="tool_use", input_tokens=6, output_tokens=2),
            ],
            [
                TextDelta("still going"),
                TurnComplete(stop_reason="end_turn", input_tokens=99, output_tokens=99),
            ],
        ]
    )

    with caplog.at_level(logging.INFO, logger="augur.observability"):
        stream = stream_chat(
            provider=provider,
            registry=registry,
            store=store,
            session_id="s1",
            message="use the tools",
            model="claude-sonnet-4-6",
        )
        await _stop_after(stream, 2)

    usage_records = [
        r for r in caplog.records if r.name == "augur.observability" and r.msg == "turn usage"
    ]
    assert len(usage_records) == 1
    # Round one is closed and billed; round two is still open, and round
    # three never ran, so neither contributes.
    assert usage_records[0].input_tokens == 5
    assert usage_records[0].output_tokens == 1


# ---------------------------------------------------------------------------
# Behavior 6: Error path
# ---------------------------------------------------------------------------


async def test_error_during_turn_yields_single_error_event_and_persists_nothing():
    # A failed turn is atomic by choice, not by accident. A stopped turn
    # keeps whatever answer it produced, so "nothing survives an
    # incomplete turn" is no longer a property of the control flow — this
    # case has to stay atomic deliberately. A failed turn produced no
    # answer, so storing the user's message alone would leave a question
    # with nothing answering it and make a retry look like asking twice.
    provider = FakeProvider([RuntimeError("boom")])
    registry = ToolRegistry()
    store = InMemoryConversationStore()

    chunks = [
        c
        async for c in stream_chat(
            provider=provider, registry=registry, store=store, session_id="s1", message="hello", model="claude-sonnet-4-6"
        )
    ]

    events = _parse_sse(chunks)
    assert events == [("error", {"type": "internal", "message": "An internal error occurred."})]

    history = await store.get_history("s1")
    assert history == []


# ---------------------------------------------------------------------------
# Behavior 7: successful turn logs usage exactly once
# ---------------------------------------------------------------------------


async def test_successful_turn_logs_usage_once(caplog):
    provider = FakeProvider(
        [[TextDelta("Hi"), TurnComplete(stop_reason="end_turn", input_tokens=7, output_tokens=11)]]
    )
    registry = ToolRegistry()
    store = InMemoryConversationStore()

    with caplog.at_level(logging.INFO, logger="augur.observability"):
        chunks = [
            c
            async for c in stream_chat(
                provider=provider,
                registry=registry,
                store=store,
                session_id="s1",
                message="hello",
                model="claude-sonnet-4-6",
            )
        ]

    events = _parse_sse(chunks)
    assert events[-1][0] == "done"

    usage_records = [r for r in caplog.records if r.name == "augur.observability"]
    assert len(usage_records) == 1
    record = usage_records[0]
    assert record.input_tokens == 7
    assert record.output_tokens == 11
    assert record.cost_usd == pytest.approx(
        compute_cost(
            input_tokens=7,
            output_tokens=11,
            input_price_per_mtok=3.0,
            output_price_per_mtok=15.0,
        )
    )


# ---------------------------------------------------------------------------
# Behavior 8: errored turn does not log usage
# ---------------------------------------------------------------------------


async def test_errored_turn_does_not_log_usage(caplog):
    provider = FakeProvider([RuntimeError("boom")])
    registry = ToolRegistry()
    store = InMemoryConversationStore()

    with caplog.at_level(logging.INFO, logger="augur.observability"):
        chunks = [
            c
            async for c in stream_chat(
                provider=provider,
                registry=registry,
                store=store,
                session_id="s1",
                message="hello",
                model="claude-sonnet-4-6",
            )
        ]

    events = _parse_sse(chunks)
    assert events == [("error", {"type": "internal", "message": "An internal error occurred."})]

    # Filter to usage records specifically — the error path now also logs an
    # "augur.observability" WARNING record (see error-path tests below), so a
    # bare logger-name filter would no longer correctly assert "no usage log".
    usage_records = [
        r for r in caplog.records if r.name == "augur.observability" and r.msg == "turn usage"
    ]
    assert len(usage_records) == 0


# ---------------------------------------------------------------------------
# Behavior 9: rate-limit error maps to typed "rate_limit" SSE event and logs
# ---------------------------------------------------------------------------


async def test_rate_limit_error_yields_rate_limit_event_and_logs(caplog):
    provider = FakeProvider([ProviderRateLimitError("rate limited")])
    registry = ToolRegistry()
    store = InMemoryConversationStore()

    with caplog.at_level(logging.WARNING, logger="augur.observability"):
        chunks = [
            c
            async for c in stream_chat(
                provider=provider,
                registry=registry,
                store=store,
                session_id="s1",
                message="hello",
                model="claude-sonnet-4-6",
            )
        ]

    events = _parse_sse(chunks)
    assert len(events) == 1
    event_name, data = events[0]
    assert event_name == "error"
    assert data["type"] == "rate_limit"
    assert data["message"]

    history = await store.get_history("s1")
    assert history == []

    error_records = [r for r in caplog.records if r.name == "augur.observability"]
    assert len(error_records) == 1
    record = error_records[0]
    assert record.levelno == logging.WARNING
    assert record.session_id == "s1"
    assert record.error_type == "rate_limit"


# ---------------------------------------------------------------------------
# Behavior 10: generic ProviderError maps to typed "provider_error" SSE event
# ---------------------------------------------------------------------------


async def test_provider_error_yields_provider_error_event():
    provider = FakeProvider([ProviderError("upstream broke")])
    registry = ToolRegistry()
    store = InMemoryConversationStore()

    chunks = [
        c
        async for c in stream_chat(
            provider=provider,
            registry=registry,
            store=store,
            session_id="s1",
            message="hello",
            model="claude-sonnet-4-6",
        )
    ]

    events = _parse_sse(chunks)
    assert len(events) == 1
    event_name, data = events[0]
    assert event_name == "error"
    assert data["type"] == "provider_error"
    assert data["message"]

    history = await store.get_history("s1")
    assert history == []


# ---------------------------------------------------------------------------
# Behavior 11: unexpected (non-provider) exception maps to "internal"
# ---------------------------------------------------------------------------


async def test_unexpected_exception_yields_internal_event():
    provider = FakeProvider([RuntimeError("bug")])
    registry = ToolRegistry()
    store = InMemoryConversationStore()

    chunks = [
        c
        async for c in stream_chat(
            provider=provider,
            registry=registry,
            store=store,
            session_id="s1",
            message="hello",
            model="claude-sonnet-4-6",
        )
    ]

    events = _parse_sse(chunks)
    assert events == [("error", {"type": "internal", "message": "An internal error occurred."})]

    history = await store.get_history("s1")
    assert history == []


# ---------------------------------------------------------------------------
# Behavior: an unknown tool name completes the turn instead of erroring it
# ---------------------------------------------------------------------------


def _unknown_tool_provider() -> FakeProvider:
    """Step 1 requests an unregistered tool; step 2 answers in text."""
    return FakeProvider(
        [
            [
                ToolUseRequested(id="t1", name="nonexistent", input={}),
                TurnComplete(stop_reason="tool_use", input_tokens=5, output_tokens=4),
            ],
            [
                TextDelta("I don't have that tool."),
                TurnComplete(stop_reason="end_turn", input_tokens=6, output_tokens=2),
            ],
        ]
    )


async def test_unknown_tool_ends_in_done_with_no_error_event():
    provider = _unknown_tool_provider()
    registry = _make_registry()
    store = InMemoryConversationStore()

    chunks = [
        c
        async for c in stream_chat(
            provider=provider,
            registry=registry,
            store=store,
            session_id="s1",
            message="use a tool",
            model="claude-sonnet-4-6",
        )
    ]

    events = _parse_sse(chunks)
    assert [name for name, _ in events] == ["tool_use", "token", "done"]
    # The tool_use event is still emitted for the unregistered tool: it is
    # sent before dispatch, and that is accepted behaviour.
    assert events[0] == ("tool_use", {"id": "t1", "name": "nonexistent", "input": {}})


async def test_unknown_tool_turn_persists_the_call_and_its_error_result():
    provider = _unknown_tool_provider()
    registry = _make_registry()
    store = InMemoryConversationStore()

    [
        c
        async for c in stream_chat(
            provider=provider,
            registry=registry,
            store=store,
            session_id="s1",
            message="use a tool",
            model="claude-sonnet-4-6",
        )
    ]

    history = await store.get_history("s1")

    # The assistant message carrying the unknown tool_use survives
    # persistable_messages because it now has a matching tool_result.
    assert len(history) == 4
    assert history[0] == {"role": "user", "content": "use a tool"}
    assert history[1]["content"] == [
        {"type": "tool_use", "id": "t1", "name": "nonexistent", "input": {}}
    ]
    assert history[2]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": "Error: tool 'nonexistent' not found. Available tools: echo.",
            "is_error": True,
        }
    ]
    assert history[3]["content"] == [
        {"type": "text", "text": "I don't have that tool."}
    ]


async def test_unknown_tool_logs_one_structured_warning(caplog):
    provider = _unknown_tool_provider()
    registry = _make_registry()
    store = InMemoryConversationStore()

    with caplog.at_level(logging.WARNING, logger="augur.agent"):
        [
            c
            async for c in stream_chat(
                provider=provider,
                registry=registry,
                store=store,
                session_id="s1",
                message="use a tool",
                model="claude-sonnet-4-6",
            )
        ]

    records = [r for r in caplog.records if r.name == "augur.agent"]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].msg == "unknown tool requested"
    assert records[0].tool_name == "nonexistent"


async def test_unknown_tool_logs_usage_as_a_completed_turn_not_an_error(caplog):
    provider = _unknown_tool_provider()
    registry = _make_registry()
    store = InMemoryConversationStore()

    with caplog.at_level(logging.INFO, logger="augur.observability"):
        [
            c
            async for c in stream_chat(
                provider=provider,
                registry=registry,
                store=store,
                session_id="s1",
                message="use a tool",
                model="claude-sonnet-4-6",
            )
        ]

    records = [r for r in caplog.records if r.name == "augur.observability"]
    assert len(records) == 1
    # log_turn_usage, not log_turn_error — the turn did not fail.
    assert records[0].msg == "turn usage"
    assert not hasattr(records[0], "error_type")
    assert records[0].input_tokens == 11
    assert records[0].output_tokens == 6


async def test_done_carries_the_terminal_stop_reason_and_the_turn_persists():
    """A truncated turn is a *completed* turn: it streams, it persists, and it
    closes with a done that names why it stopped."""
    provider = FakeProvider(
        [[TextDelta("The three main "), TurnComplete(stop_reason="max_tokens", input_tokens=1, output_tokens=1)]]
    )
    registry = ToolRegistry()
    store = InMemoryConversationStore()

    chunks = [
        c
        async for c in stream_chat(
            provider=provider, registry=registry, store=store, session_id="s1", message="list them", model="claude-sonnet-4-6"
        )
    ]

    events = _parse_sse(chunks)
    assert events == [
        ("token", {"text": "The three main "}),
        ("done", {"stop_reason": "max_tokens"}),
    ]

    history = await store.get_history("s1")
    assert history == [
        {"role": "user", "content": "list them"},
        {"role": "assistant", "content": [{"type": "text", "text": "The three main "}]},
    ]


async def test_truncated_turn_with_no_text_or_tool_use_yields_bare_done():
    """The cross-module seam for the bug this branch fixes: a JSONDecodeError
    escaping the provider generator and being caught here as an "internal"
    error. Every other truncation test in this file (and all of
    tests/test_provider.py) sits below this seam — inside stream_turn or
    hand-feeding stream_chat the post-guard event stream — so none of them
    would catch a regression if the guard moved.

    This is exactly what the provider now emits for a turn truncated
    mid-tool_use with no preceding text: a single TurnComplete and nothing
    else — no ToolUseRequested (the half-written block was dropped) and no
    TextDelta (none had streamed yet).
    """
    provider = FakeProvider(
        [[TurnComplete(stop_reason="max_tokens", input_tokens=1, output_tokens=1)]]
    )
    registry = ToolRegistry()
    store = InMemoryConversationStore()

    chunks = [
        c
        async for c in stream_chat(
            provider=provider, registry=registry, store=store, session_id="s1", message="list them", model="claude-sonnet-4-6"
        )
    ]

    events = _parse_sse(chunks)
    assert events == [("done", {"stop_reason": "max_tokens"})]

    # The user's message persists; no empty assistant message is stored.
    history = await store.get_history("s1")
    assert history == [{"role": "user", "content": "list them"}]
