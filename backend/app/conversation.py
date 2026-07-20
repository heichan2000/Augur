"""Server-side conversation store for Augur Phase 1.

Holds Anthropic-format message dicts keyed by session_id:
    {"role": "user"|"assistant", "content": <str | list of content blocks>}

The interface is async by design so Phase 2+ can swap in a persistent
(e.g. Postgres) store whose operations are genuinely async; call sites
already use ``await`` and require no changes on that swap.

``persistable_messages`` is the single guard a persistence path runs its
messages through, so a turn can never leave stored history in a state we
could not replay to the model.

Phase 1 limitations (future concerns, not implemented here):
- No locking/concurrency primitives — safe only for single-process in-memory use.
- No message validation beyond ``persistable_messages`` — callers are otherwise
  trusted to supply well-formed dicts.
- No history truncation/windowing — unbounded growth per session.
- Shallow copy returned by get_history — message dicts themselves are shared
  between the caller and internal storage; do not mutate dict contents.
"""
from __future__ import annotations

import abc
import functools
from typing import Any

Message = dict[str, Any]  # Anthropic-format: {"role": ..., "content": ...}


def _content_blocks(message: Message, block_type: str) -> list[dict[str, Any]]:
    """Return *message*'s content blocks of *block_type*.

    String content (a plain text message) holds no blocks.
    """
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [
        block
        for block in content
        if isinstance(block, dict) and block.get("type") == block_type
    ]


def _tool_result_ids(message: Message) -> set[str | None]:
    """Return the ids of the tool calls *message* answers."""
    return {
        block["tool_use_id"]
        for block in _content_blocks(message, "tool_result")
        if "tool_use_id" in block
    }


def _is_unanswered_call(block: Any, answered: set[str | None]) -> bool:
    """Is *block* a tool call that *answered* does not cover?

    A ``tool_use`` block with no ``id`` counts as unanswered — a call we
    cannot match to a result is one we cannot replay.
    """
    if not isinstance(block, dict) or block.get("type") != "tool_use":
        return False
    return block.get("id") not in answered


def _without_unanswered_calls(message: Message, answered: set[str | None]) -> Message | None:
    """Return *message* stripped of the tool calls *answered* leaves open.

    Returns the message untouched when every call it makes is answered, a
    copy carrying only the surviving blocks when some are not, and ``None``
    when no content survives at all. Never mutates the message passed in —
    stored history and the caller's working list share these dicts.
    """
    content = message.get("content")
    if not isinstance(content, list):
        return message if content else None

    kept = [block for block in content if not _is_unanswered_call(block, answered)]
    if not kept:
        return None
    if len(kept) == len(content):
        return message
    return {**message, "content": kept}


def persistable_messages(messages: list[Message]) -> list[Message]:
    """Return the part of *messages* that is safe to persist.

    Stored history is replayable only if every ``tool_use`` block is
    answered by a ``tool_result`` in the message that follows it. A call
    left open — as when the agent loop hits its step bound mid-round, or a
    turn is cut short while a tool is still running — is dropped.

    Only the call is dropped, not the turn: an assistant message that
    streamed an answer alongside an unanswerable call keeps that answer,
    because the user read it and the model should be able to recall it. A
    message left with no content at all is dropped entirely, and a sequence
    with no unanswered call is returned unchanged.

    A turn that leaves no assistant content at all persists nothing, not
    even the user's message. A question with nothing answering it is not
    worth storing: it would reach the model as an exchange that never
    happened, and re-asking would record it twice.
    """
    persistable: list[Message] = []
    for index, message in enumerate(messages):
        following = messages[index + 1] if index + 1 < len(messages) else None
        answered = _tool_result_ids(following) if following else set()
        kept = _without_unanswered_calls(message, answered)
        if kept is not None:
            persistable.append(kept)

    if not any(message.get("role") == "assistant" for message in persistable):
        return []
    return persistable


class ConversationStore(abc.ABC):
    """Abstract interface for a session-keyed conversation history store."""

    @abc.abstractmethod
    async def append(self, session_id: str, message: Message) -> None:
        """Append *message* to the ordered history for *session_id*.

        Creates the session's list on first append.
        """
        ...

    @abc.abstractmethod
    async def get_history(self, session_id: str) -> list[Message]:
        """Return the session's messages in insertion order.

        Returns an empty list for an unknown session (no error, no side-effect).
        The returned list is a shallow copy — mutating it does not affect stored
        history, but mutating the message dicts themselves does (Phase 1 caveat).
        """
        ...

    @abc.abstractmethod
    async def reset(self, session_id: str) -> None:
        """Drop all history for *session_id* (no error if it does not exist)."""
        ...


class InMemoryConversationStore(ConversationStore):
    """In-memory implementation of ConversationStore backed by a plain dict."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[Message]] = {}

    async def append(self, session_id: str, message: Message) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(message)

    async def get_history(self, session_id: str) -> list[Message]:
        return list(self._sessions.get(session_id, []))

    async def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


@functools.lru_cache(maxsize=1)
def get_conversation_store() -> ConversationStore:
    """Return the process-wide singleton InMemoryConversationStore.

    Use ``get_conversation_store.cache_clear()`` in tests that need isolation.
    Tests should construct ``InMemoryConversationStore()`` directly instead.
    """
    return InMemoryConversationStore()
