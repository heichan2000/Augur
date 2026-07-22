"""Tests for ConversationStore — TDD: written BEFORE conversation.py exists.

All tests construct InMemoryConversationStore() directly (not the singleton).
All test functions are async (asyncio_mode = "auto" handles the event loop).
"""
import pytest

from app.conversation import (
    ConversationStore,
    InMemoryConversationStore,
    TurnOutcome,
    persistable_messages,
)


# ---------------------------------------------------------------------------
# Test 1: Unknown session returns empty list
# ---------------------------------------------------------------------------


async def test_get_history_unknown_session_returns_empty_list():
    store = InMemoryConversationStore()
    result = await store.get_history("nope")
    assert result == []


# ---------------------------------------------------------------------------
# Test 2: Order preserved
# ---------------------------------------------------------------------------


async def test_append_preserves_insertion_order():
    store = InMemoryConversationStore()
    msg1 = {"role": "user", "content": "first"}
    msg2 = {"role": "assistant", "content": "second"}
    await store.append("s1", msg1)
    await store.append("s1", msg2)
    history = await store.get_history("s1")
    assert history == [msg1, msg2]


# ---------------------------------------------------------------------------
# Test 3: Multi-turn accumulation (client need not resend prior context)
# ---------------------------------------------------------------------------


async def test_multi_turn_accumulation():
    store = InMemoryConversationStore()
    user_msg = {"role": "user", "content": "Hello"}
    assistant_msg = {"role": "assistant", "content": "Hi there!"}
    await store.append("s1", user_msg)
    await store.append("s1", assistant_msg)
    history = await store.get_history("s1")
    assert history == [user_msg, assistant_msg]


# ---------------------------------------------------------------------------
# Test 4: Session isolation
# ---------------------------------------------------------------------------


async def test_session_isolation():
    store = InMemoryConversationStore()
    msg_s1 = {"role": "user", "content": "for s1"}
    await store.append("s1", msg_s1)
    history_s2 = await store.get_history("s2")
    assert history_s2 == []
    assert msg_s1 not in history_s2


# ---------------------------------------------------------------------------
# Test 5: get_history returns a copy (mutating returned list does not corrupt store)
# ---------------------------------------------------------------------------


async def test_get_history_returns_copy():
    store = InMemoryConversationStore()
    msg = {"role": "user", "content": "original"}
    await store.append("s1", msg)

    history1 = await store.get_history("s1")
    history1.append({"role": "assistant", "content": "injected"})

    history2 = await store.get_history("s1")
    assert history2 == [msg]


# ---------------------------------------------------------------------------
# Test 6: reset clears a session
# ---------------------------------------------------------------------------


async def test_reset_clears_session():
    store = InMemoryConversationStore()
    await store.append("s1", {"role": "user", "content": "hello"})
    await store.reset("s1")
    history = await store.get_history("s1")
    assert history == []


# ---------------------------------------------------------------------------
# Test 7: reset on unknown session raises no error
# ---------------------------------------------------------------------------


async def test_reset_unknown_session_no_error():
    store = InMemoryConversationStore()
    await store.reset("nonexistent")  # must not raise


# ---------------------------------------------------------------------------
# Test 8: Interface conformance — isinstance check and ABC non-instantiable
# ---------------------------------------------------------------------------


async def test_interface_conformance():
    store = InMemoryConversationStore()
    assert isinstance(store, ConversationStore)

    with pytest.raises(TypeError):
        ConversationStore()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Test 9: an empty-content assistant message is dropped, not persisted
# ---------------------------------------------------------------------------


def test_persistable_messages_drops_an_empty_content_assistant_message():
    # A truncated tool block with no preceding text (app/agent.py) leaves
    # exactly this message reachable: {"role": "assistant", "content": []}.
    # It must never reach storage — an empty-content assistant message is
    # not something the Anthropic API accepts back on a later turn.
    messages = [
        {"role": "user", "content": "list them"},
        {"role": "assistant", "content": []},
    ]

    result = persistable_messages(messages, outcome=TurnOutcome.COMPLETED)

    assert result == [{"role": "user", "content": "list them"}]
