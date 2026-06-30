"""Anthropic streaming provider — normalizes SDK stream events into domain events.

Transport-agnostic: no FastAPI, no SSE imports. The Anthropic client is
injected via the constructor so tests can pass a fake.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator

import anthropic

from app.config import get_settings


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
# Provider
# ---------------------------------------------------------------------------


class AnthropicProvider:
    def __init__(self, client: anthropic.AsyncAnthropic, model: str) -> None:
        self._client = client
        self._model = model

    async def stream_turn(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
    ) -> AsyncIterator[ProviderEvent]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
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
                        parsed = json.loads(raw) if raw else {}
                        yield ToolUseRequested(
                            id=block["id"],
                            name=block["name"],
                            input=parsed,
                        )

                elif etype == "message_delta":
                    stop_reason = event.delta.stop_reason
                    output_tokens = event.usage.output_tokens

                # message_stop — no fields used, intentionally ignored

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
    return AnthropicProvider(client=client, model=settings.anthropic_model)


@functools.lru_cache(maxsize=1)
def get_provider() -> AnthropicProvider:
    """Return the process-wide singleton AnthropicProvider.

    FastAPI dependency — tests override this via ``app.dependency_overrides``
    so the real Anthropic client is never constructed.
    """
    return build_provider()
