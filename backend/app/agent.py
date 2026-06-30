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

from typing import Any, AsyncIterator

from app.provider import ProviderEvent, TextDelta, ToolUseRequested, TurnComplete
from app.tools import ToolRegistry

Message = dict[str, Any]  # Anthropic-format: {"role": ..., "content": ...}


async def run_turn(
    *,
    provider: Any,
    registry: ToolRegistry,
    messages: list[Message],
    system: str | None = None,
    max_steps: int = 8,
) -> AsyncIterator[ProviderEvent]:
    """Drive one turn to completion, streaming events as they arrive.

    Passes through each ``TextDelta``/``ToolUseRequested`` from the
    provider, dispatches requested tools, appends the resulting messages
    to *messages* in place, and yields one final ``TurnComplete`` whose
    ``stop_reason`` is the terminal step's and whose token counts are
    summed across all steps. Never raises or hangs on the ``max_steps``
    bound — it simply stops.
    """
    total_input_tokens = 0
    total_output_tokens = 0
    final_stop_reason: str | None = None
    tools = registry.schemas() or None

    for step in range(max_steps):
        text_parts: list[str] = []
        tool_uses: list[ToolUseRequested] = []

        async for event in provider.stream_turn(messages=messages, system=system, tools=tools):
            if isinstance(event, TextDelta):
                text_parts.append(event.text)
                yield event
            elif isinstance(event, ToolUseRequested):
                tool_uses.append(event)
                yield event
            elif isinstance(event, TurnComplete):
                total_input_tokens += event.input_tokens
                total_output_tokens += event.output_tokens
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

        if not tool_uses:
            break

        if step == max_steps - 1:
            # Bound hit while the model is still requesting tools: stop right
            # after appending the assistant message above — do not dispatch
            # or make another provider call.
            break

        tool_results = []
        for tool_use in tool_uses:
            result = await registry.dispatch(tool_use.name, tool_use.input)
            tool_results.append(
                {"type": "tool_result", "tool_use_id": tool_use.id, "content": result}
            )
        messages.append({"role": "user", "content": tool_results})

    yield TurnComplete(
        stop_reason=final_stop_reason,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
    )
