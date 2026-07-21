"""Transport-agnostic SSE body generator for the `/chat` endpoint.

``stream_chat`` composes the provider, the tool registry, and the
conversation store into a single async generator of SSE wire strings. It
has no FastAPI/HTTP dependency — the FastAPI route that adapts this to a
``StreamingResponse`` is a separate concern (#4 Task 2).

Persistence works on the messages new to this turn — the working
``messages`` list (prior history + the new user message) minus the history
it started from. Each way a turn can end has one rule:

- **Completes** (``TurnOutcome.COMPLETED``) — persists in full, including
  the user's message on a turn where the model said nothing.
- **Fails** (provider error, rate limit, internal) — persists nothing, and
  never reaches ``persistable_messages`` at all. Deliberately atomic: no
  answer was produced, so storing the user's message alone would leave a
  question nothing answers and make a retry look like asking twice.
- **Stopped** (``TurnOutcome.STOPPED``, the client went away) — persists
  the user's message and whatever answer had been streamed, because the
  user read it. If no answer had been streamed, persists nothing.

The two persisting outcomes run the turn's new messages through
``conversation.persistable_messages``, which drops any tool call left
unanswered — as when ``run_turn`` hits its ``max_steps`` bound while the
model is still requesting tools — along with any message left empty. An
answer streamed alongside an unanswerable call survives; only the call is
dropped. Where the two outcomes differ is a turn that leaves no assistant
content at all: see that function for which keeps the user's message.

Atomicity note: the persistence loop below assumes ``store.append`` cannot
fail, which holds for the Phase-1 in-memory store. A persistent (Phase-2)
store whose ``append`` can raise mid-loop would need this revisited — a
failure partway through would leave a partial turn persisted while the
exception escapes the generator uncaught.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from app.agent import TurnProgress, run_turn
from app.config import PERSIST_ON_STOP_TIMEOUT_SECONDS
from app.conversation import (
    ConversationStore,
    Message,
    TurnOutcome,
    persistable_messages,
)
from app.observability import log_persist_timeout, log_turn_error, log_turn_usage
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

    Loads prior history for *session_id* and runs the turn via
    ``run_turn``. On success, persists the turn's new messages — as far as
    they form a replayable sequence — and logs one structured token/cost
    usage record (see ``app.observability``) keyed by *model*, before
    yielding the final ``done`` event.

    On any exception raised while producing the turn, yields a single
    typed ``error`` event — ``rate_limit`` for ``ProviderRateLimitError``,
    ``provider_error`` for other ``ProviderError``s, ``internal`` for
    anything else — and stops: no ``done``, nothing persisted, and one
    ``log_turn_error`` record instead (no token counts are available
    there).

    If the client goes away mid-turn the turn is *stopped*: whatever
    answer had already been streamed is persisted and whatever tokens the
    provider had already billed are logged, but nothing further is
    yielded — the connection is gone. See the module docstring for the
    full rule per outcome.
    """
    history = await store.get_history(session_id)
    user_msg: Message = {"role": "user", "content": message}
    messages: list[Message] = [*history, user_msg]

    final_turn: TurnComplete | None = None
    progress = TurnProgress()

    async def persist(new_messages: list[Message], *, outcome: TurnOutcome) -> None:
        for new_message in persistable_messages(new_messages, outcome=outcome):
            await store.append(session_id, new_message)

    try:
        async for event in run_turn(
            provider=provider,
            registry=registry,
            messages=messages,
            system=system,
            progress=progress,
        ):
            if isinstance(event, TextDelta):
                yield format_sse(TokenEvent(text=event.text))
            elif isinstance(event, ToolUseRequested):
                yield format_sse(ToolUseEvent(id=event.id, name=event.name, input=event.input))
            elif isinstance(event, TurnComplete):
                final_turn = event
    except (GeneratorExit, asyncio.CancelledError):
        # The client went away mid-turn. Which of the two arrives depends on
        # where the turn happened to be: parked at a yield nobody will read
        # gives GeneratorExit, suspended in its own await — waiting on the
        # provider or a running tool, where a turn spends most of its time —
        # gives CancelledError. Both are BaseExceptions, which is why the
        # handlers below never saw either.
        #
        # Persist what the client already read, log what the provider
        # already billed, then let the unwind proceed. Nothing may be
        # yielded here, and nothing needs to be: the connection is gone.
        stopped: list[Message] = list(messages[len(history):])
        if progress.partial_text:
            stopped.append(
                {"role": "assistant", "content": [{"type": "text", "text": progress.partial_text}]}
            )
        # Shielded because this await runs while a cancellation unwinds.
        # The Phase-1 in-memory store never suspends, so today the write
        # completes before anything can interrupt it — but a Phase-2 async
        # store introduces a real suspension point, and an await that
        # suspends mid-unwind gets cancelled again, silently losing the
        # write. The shield keeps the write running to completion; the
        # timeout keeps a store that never returns from hanging the unwind.
        try:
            await asyncio.wait_for(
                asyncio.shield(persist(stopped, outcome=TurnOutcome.STOPPED)),
                timeout=PERSIST_ON_STOP_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            log_persist_timeout(session_id=session_id, model=model)
        if progress.input_tokens or progress.output_tokens:
            log_turn_usage(
                session_id=session_id,
                model=model,
                input_tokens=progress.input_tokens,
                output_tokens=progress.output_tokens,
            )
        raise
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

    await persist(messages[len(history):], outcome=TurnOutcome.COMPLETED)

    if final_turn is not None:
        log_turn_usage(
            session_id=session_id,
            model=model,
            input_tokens=final_turn.input_tokens,
            output_tokens=final_turn.output_tokens,
        )

    yield format_sse(DoneEvent(stop_reason=final_turn.stop_reason if final_turn else None))
