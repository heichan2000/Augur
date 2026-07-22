"""Anthropic streaming provider — normalizes SDK stream events into domain events.

Transport-agnostic: no FastAPI, no SSE imports. The Anthropic client is
injected via the constructor so tests can pass a fake.
"""
from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator

import anthropic

from app.config import get_settings

logger = logging.getLogger("augur.provider")

# Stop reasons meaning the response was cut off rather than finished. A tool
# block that arrives on such a turn cannot be trusted — its JSON may be half
# written, or missing entirely — so it is dropped rather than dispatched.
#
# The Anthropic SDK's own two enums disagree on the full value set
# (anthropic/types/stop_reason.py lists six values and omits
# model_context_window_exceeded; the beta variant lists eight), which is why
# nothing else in this codebase validates stop_reason against an allow-list.
# This frozenset is the one place a value is interpreted, and it names only
# the two that mean "truncated".
TRUNCATION_STOP_REASONS = frozenset({"max_tokens", "model_context_window_exceeded"})


# ---------------------------------------------------------------------------
# Domain event types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolUseRequested:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class TurnComplete:
    stop_reason: str | None
    input_tokens: int
    output_tokens: int


# Union alias — the caller annotates with this type.
ProviderEvent = TextDelta | ToolUseRequested | TurnComplete


# ---------------------------------------------------------------------------
# Domain errors
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Base: an upstream provider call failed."""


class ProviderRateLimitError(ProviderError):
    """Upstream provider rate-limited the request."""


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class AnthropicProvider:
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        model: str,
        max_tokens: int,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    async def stream_turn(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        if system is not None:
            kwargs["system"] = system
        if tools is not None:
            kwargs["tools"] = tools

        # Accumulated state across events
        input_tokens: int = 0
        output_tokens: int = 0
        stop_reason: str | None = None

        # Per-block tracking: index → {"id": str, "name": str, "buf": str}
        tool_blocks: dict[int, dict[str, Any]] = {}

        # Parsed tool calls held back until stop_reason is known. The stream
        # reveals stop_reason (in message_delta) only *after* every
        # content_block_stop, so at block-close time an empty buffer is
        # ambiguous: a zero-argument tool and a call truncated before its
        # first input_json_delta look identical. Replaying after the loop
        # resolves that, and keeps a dead call from reaching the UI as a
        # tool_use SSE event that would be drawn as a row and settled "done".
        pending_tool_uses: list[ToolUseRequested] = []

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    etype = event.type

                    if etype == "message_start":
                        input_tokens = event.message.usage.input_tokens

                    elif etype == "content_block_start":
                        cb = event.content_block
                        if cb.type == "tool_use":
                            tool_blocks[event.index] = {
                                "id": cb.id,
                                "name": cb.name,
                                "buf": "",
                            }

                    elif etype == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield TextDelta(delta.text)
                        elif delta.type == "input_json_delta":
                            if event.index in tool_blocks:
                                tool_blocks[event.index]["buf"] += delta.partial_json

                    elif etype == "content_block_stop":
                        idx = event.index
                        if idx in tool_blocks:
                            block = tool_blocks.pop(idx)
                            raw = block["buf"]
                            try:
                                parsed = json.loads(raw) if raw else {}
                            except json.JSONDecodeError:
                                # Half-written arguments. Dropping the block
                                # loses one tool call; letting this raise
                                # would lose the whole turn, including text
                                # already streamed to the user.
                                logger.warning(
                                    "dropping tool block with unparseable input",
                                    extra={"tool_name": block["name"]},
                                )
                                continue
                            pending_tool_uses.append(
                                ToolUseRequested(
                                    id=block["id"],
                                    name=block["name"],
                                    input=parsed,
                                )
                            )

                    elif etype == "message_delta":
                        stop_reason = event.delta.stop_reason
                        output_tokens = event.usage.output_tokens

                    # message_stop — no fields used, intentionally ignored
        except anthropic.RateLimitError as exc:
            raise ProviderRateLimitError(str(exc)) from exc
        except anthropic.APIError as exc:  # status errors + connection/network
            raise ProviderError(str(exc)) from exc

        # Outside the try, alongside the TurnComplete that already lived here:
        # a stream that raised reaches neither.
        if stop_reason in TRUNCATION_STOP_REASONS:
            for tool_use in pending_tool_uses:
                logger.warning(
                    "dropping tool block from a truncated turn",
                    extra={"tool_name": tool_use.name},
                )
        else:
            for tool_use in pending_tool_uses:
                yield tool_use

        yield TurnComplete(
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


# ---------------------------------------------------------------------------
# Factory (production wiring — tests do not call this)
# ---------------------------------------------------------------------------


def build_provider() -> AnthropicProvider:
    settings = get_settings()
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return AnthropicProvider(
        client=client,
        model=settings.anthropic_model,
        max_tokens=settings.anthropic_max_tokens,
    )


@functools.lru_cache(maxsize=1)
def get_provider() -> AnthropicProvider:
    """Return the process-wide singleton AnthropicProvider.

    FastAPI dependency — tests override this via ``app.dependency_overrides``
    so the real Anthropic client is never constructed.
    """
    return build_provider()
