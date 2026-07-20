"""Transport-agnostic SSE body generator for the `/chat` endpoint.

``stream_chat`` composes the provider, the tool registry, and the
conversation store into a single async generator of SSE wire strings. It
has no FastAPI/HTTP dependency — the FastAPI route that adapts this to a
``StreamingResponse`` is a separate concern (#4 Task 2).

Persistence: the working ``messages`` list (prior history + the new user
message) is only persisted to the store after the turn completes
successfully, and only the messages new to this turn. On any error during
the turn, nothing is persisted.

Stored history is therefore always a valid, replayable Anthropic message
sequence. Two rules keep it so: messages with empty ``content`` are
skipped, and the new messages pass through
``conversation.replayable_prefix`` first — so a turn that ends
mid-tool-round (``run_turn`` hitting its ``max_steps`` bound while the
model is still requesting tools) drops its unanswered assistant
``tool_use`` message rather than persisting a history we could not
replay.

Atomicity note: the persistence loop below assumes ``store.append`` cannot
fail, which holds for the Phase-1 in-memory store. A persistent (Phase-2)
store whose ``append`` can raise mid-loop would need this revisited — a
failure partway through would leave a partial turn persisted while the
exception escapes the generator uncaught.
"""
from __future__ import annotations

from typing import Any, AsyncIterator

from app.agent import run_turn
from app.conversation import ConversationStore, Message, replayable_prefix
from app.observability import log_turn_error, log_turn_usage
from app.provider import (
    ProviderError,
    ProviderRateLimitError,
    TextDelta,
    ToolUseRequested,
    TurnComplete,
)
from app.sse import DoneEvent, ErrorEvent, ToolUseEvent, TokenEvent, format_sse
from app.tools import ToolRegistry


async def stream_chat(
    *,
    provider: Any,
    registry: ToolRegistry,
    store: ConversationStore,
    session_id: str,
    message: str,
    model: str,
    system: str | None = None,
) -> AsyncIterator[str]:
    """Run one chat turn and yield SSE wire strings for the response body.

    Loads prior history for *session_id*, runs the turn via ``run_turn``,
    and on success persists the turn's new messages — as far as they
    form a replayable sequence — before yielding the final ``done``
    event. On any exception raised while producing the
    turn, yields a single typed ``error`` event — ``rate_limit`` for
    ``ProviderRateLimitError``, ``provider_error`` for other
    ``ProviderError``s, ``internal`` for anything else — and stops: no
    `done`, and nothing is persisted.

    On success, logs one structured token/cost usage record (see
    ``app.observability``) keyed by *model* before yielding ``done``. On
    error, logs one structured ``log_turn_error`` record instead — no
    token counts are available there.
    """
    history = await store.get_history(session_id)
    user_msg: Message = {"role": "user", "content": message}
    messages: list[Message] = [*history, user_msg]

    final_turn: TurnComplete | None = None

    try:
        async for event in run_turn(provider=provider, registry=registry, messages=messages, system=system):
            if isinstance(event, TextDelta):
                yield format_sse(TokenEvent(text=event.text))
            elif isinstance(event, ToolUseRequested):
                yield format_sse(ToolUseEvent(id=event.id, name=event.name, input=event.input))
            elif isinstance(event, TurnComplete):
                final_turn = event
    except ProviderRateLimitError:
        log_turn_error(session_id=session_id, error_type="rate_limit")
        yield format_sse(
            ErrorEvent(
                type="rate_limit",
                message="The service is temporarily rate limited. Please retry shortly.",
            )
        )
        return
    except ProviderError:
        log_turn_error(session_id=session_id, error_type="provider_error")
        yield format_sse(
            ErrorEvent(
                type="provider_error",
                message="The upstream model provider returned an error.",
            )
        )
        return
    except Exception:
        log_turn_error(session_id=session_id, error_type="internal")
        yield format_sse(ErrorEvent(type="internal", message="An internal error occurred."))
        return

    for new_message in replayable_prefix(messages[len(history):]):
        if new_message["content"] == []:
            continue
        await store.append(session_id, new_message)

    if final_turn is not None:
        log_turn_usage(
            session_id=session_id,
            model=model,
            input_tokens=final_turn.input_tokens,
            output_tokens=final_turn.output_tokens,
        )

    yield format_sse(DoneEvent())
