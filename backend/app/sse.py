"""SSE event contract — typed event models and serializer.

Wire framing (UTF-8):
    event: <type>\n
    data: <compact-single-line-json>\n
    \n

Four event types:
  token    — incremental assistant text
  tool_use — assistant is invoking a tool
  error    — typed error; stream ends after this
  done     — end-of-stream sentinel; last event of a successful turn

Additive evolution rule: future phases MAY add fields to any payload.
Existing consumers must ignore unknown fields and keep working.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel


class SSEEvent(BaseModel):
    """Common base for all SSE events.

    Subclasses declare their event name as a class variable:
        event: ClassVar[str] = "..."
    """

    event: ClassVar[str]


class TokenEvent(SSEEvent):
    """Incremental assistant text token."""

    event: ClassVar[str] = "token"

    text: str


class ToolUseEvent(SSEEvent):
    """Notice that the assistant is invoking a tool."""

    event: ClassVar[str] = "tool_use"

    id: str
    name: str
    input: dict[str, Any]


class ErrorEvent(SSEEvent):
    """A typed error; the stream ends after this event.

    Stable machine-readable type codes:
        "rate_limit"       — upstream provider rate limit
        "provider_error"   — upstream provider returned an error
        "invalid_request"  — the request payload was rejected
        "internal"         — unexpected server-side failure
    """

    event: ClassVar[str] = "error"

    type: str
    message: str


class DoneEvent(SSEEvent):
    """End-of-stream sentinel; the last event of a successful turn."""

    event: ClassVar[str] = "done"


def format_sse(event: SSEEvent) -> str:
    """Serialize an SSE event to its wire representation.

    Returns exactly:
        f"event: {event.event}\\ndata: {event.model_dump_json()}\\n\\n"

    model_dump_json() emits compact JSON (no spaces) with embedded newlines
    JSON-escaped as \\n, so the output is always a single SSE message with
    exactly one blank-line terminator.
    """
    return f"event: {event.event}\ndata: {event.model_dump_json()}\n\n"


__all__ = [
    "SSEEvent",
    "TokenEvent",
    "ToolUseEvent",
    "ErrorEvent",
    "DoneEvent",
    "format_sse",
]
