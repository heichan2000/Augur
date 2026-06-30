"""Transport-agnostic SSE body generator for the `/chat` endpoint.

``stream_chat`` composes the provider, the tool registry, and the
conversation store into a single async generator of SSE wire strings. It
has no FastAPI/HTTP dependency — the FastAPI route that adapts this to a
``StreamingResponse`` is a separate concern (#4 Task 2).

Persistence: the working ``messages`` list (prior history + the new user
message) is only persisted to the store after the turn completes
successfully, and only the messages new to this turn (skipping any with
empty ``content``) — so stored history always stays a valid, replayable
Anthropic message sequence. On any error during the turn, nothing is
persisted.

Phase-1 limitation (documented, not fixed here): if ``run_turn`` hits its
``max_steps`` bound while the model is still requesting tools, the final
appended message is an assistant ``tool_use`` block with no matching
``tool_result``. Persisting that message would leave the stored history in
a state that is not a valid replay sequence. This requires 8 tool rounds
in a single turn and is out of scope for #4.
"""
from __future__ import annotations

from typing import Any, AsyncIterator

from app.agent import run_turn
from app.conversation import ConversationStore, Message
from app.provider import TextDelta, ToolUseRequested
from app.sse import DoneEvent, ErrorEvent, ToolUseEvent, TokenEvent, format_sse
from app.tools import ToolRegistry


async def stream_chat(
    *,
    provider: Any,
    registry: ToolRegistry,
    store: ConversationStore,
    session_id: str,
    message: str,
    system: str | None = None,
) -> AsyncIterator[str]:
    """Run one chat turn and yield SSE wire strings for the response body.

    Loads prior history for *session_id*, runs the turn via ``run_turn``,
    and on success persists the turn's new messages before yielding the
    final ``done`` event. On any exception raised while producing the
    turn, yields a single generic ``error`` event and stops — no `done`,
    and nothing is persisted.
    """
    history = await store.get_history(session_id)
    user_msg: Message = {"role": "user", "content": message}
    messages: list[Message] = [*history, user_msg]

    try:
        async for event in run_turn(provider=provider, registry=registry, messages=messages, system=system):
            if isinstance(event, TextDelta):
                yield format_sse(TokenEvent(text=event.text))
            elif isinstance(event, ToolUseRequested):
                yield format_sse(ToolUseEvent(id=event.id, name=event.name, input=event.input))
            # TurnComplete ends the turn — no token emitted for it.
    except Exception:
        yield format_sse(ErrorEvent(type="internal", message="An internal error occurred."))
        return

    for new_message in messages[len(history):]:
        if new_message["content"] == []:
            continue
        await store.append(session_id, new_message)

    yield format_sse(DoneEvent())
