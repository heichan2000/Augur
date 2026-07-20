"""Dispatch / agent loop — the multi-step turn driver.

Drives the provider/tool-registry tool-calling round trip: streams events
from the provider, dispatches any requested tools via the registry, feeds
the results back, and repeats until the model stops requesting tools (or
``max_steps`` is hit). Store-agnostic — operates on a plain ``messages``
list that the caller owns and persists.

Phase scope (future concerns, not implemented here):
- No catching of handler exceptions as model observations (Phase 3).
- No `/chat` SSE endpoint or conversation-store wiring (#4).
"""
from __future__ import annotations

import dataclasses
from typing import Any, AsyncIterator

from app.config import AGENT_MAX_STEPS
from app.provider import ProviderEvent, TextDelta, ToolUseRequested, TurnComplete
from app.tools import ToolRegistry

Message = dict[str, Any]  # Anthropic-format: {"role": ..., "content": ...}


@dataclasses.dataclass
class TurnProgress:
    """What a turn has produced so far, readable before it finishes.

    ``run_turn`` yields its totals only in the closing ``TurnComplete``, so
    a caller whose turn is cut short never sees them. Passing one of these
    in gives the caller an object it owns and can still read afterwards —
    the tokens the provider already billed, and any answer streamed since
    the last message ``run_turn`` committed to the messages list.

    ``partial_text`` is empty whenever the turn is between steps: the text
    it held has been appended to ``messages`` and is no longer partial.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    partial_text: str = ""


async def run_turn(
    *,
    provider: Any,
    registry: ToolRegistry,
    messages: list[Message],
    system: str | None = None,
    max_steps: int = AGENT_MAX_STEPS,
    progress: TurnProgress | None = None,
) -> AsyncIterator[ProviderEvent]:
    """Drive one turn to completion, streaming events as they arrive.

    Passes through each ``TextDelta``/``ToolUseRequested`` from the
    provider, dispatches requested tools, appends the resulting messages
    to *messages* in place, and yields one final ``TurnComplete`` whose
    ``stop_reason`` is the terminal step's and whose token counts are
    summed across all steps. Never raises or hangs on the ``max_steps``
    bound — it simply stops.

    Pass a *progress* object to read those totals — and any answer
    streamed since the last committed message — without waiting for the
    closing ``TurnComplete``. A caller whose turn is stopped part-way
    never receives that event, so this is the only way to see what the
    turn had produced by then.
    """
    progress = progress if progress is not None else TurnProgress()
    final_stop_reason: str | None = None
    tools = registry.schemas() or None

    for step in range(max_steps):
        text_parts: list[str] = []
        tool_uses: list[ToolUseRequested] = []

        async for event in provider.stream_turn(messages=messages, system=system, tools=tools):
            if isinstance(event, TextDelta):
                text_parts.append(event.text)
                progress.partial_text = "".join(text_parts)
                yield event
            elif isinstance(event, ToolUseRequested):
                tool_uses.append(event)
                yield event
            elif isinstance(event, TurnComplete):
                progress.input_tokens += event.input_tokens
                progress.output_tokens += event.output_tokens
                final_stop_reason = event.stop_reason

        content: list[dict[str, Any]] = []
        text = "".join(text_parts)
        if text:
            content.append({"type": "text", "text": text})
        for tool_use in tool_uses:
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_use.id,
                    "name": tool_use.name,
                    "input": tool_use.input,
                }
            )
        messages.append({"role": "assistant", "content": content})
        # Committed to *messages* — no longer the caller's to salvage.
        progress.partial_text = ""

        if not tool_uses:
            break

        if step == max_steps - 1:
            # Bound hit while the model is still requesting tools: stop right
            # after appending the assistant message above — do not dispatch
            # or make another provider call.
            break

        tool_results: list[dict[str, Any]] = []
        for tool_use in tool_uses:
            result = await registry.dispatch(tool_use.name, tool_use.input)
            tool_results.append(
                {"type": "tool_result", "tool_use_id": tool_use.id, "content": result}
            )
        messages.append({"role": "user", "content": tool_results})

    yield TurnComplete(
        stop_reason=final_stop_reason,
        input_tokens=progress.input_tokens,
        output_tokens=progress.output_tokens,
    )
