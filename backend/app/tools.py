"""Tool registry for Augur's tool-calling spine.

Holds a schema + async handler per tool, keyed by name. Pure unit — no
provider, no FastAPI, no SSE imports. The dispatch loop (a later piece)
calls ``schemas()`` to advertise available tools to the model and
``dispatch()`` to run the handler the model selected.

Phase scope (future concerns, not implemented here):
- No catching of handler *exceptions* as model observations (Phase 3). An
  unknown tool *name* is a different case and is already handled: the
  dispatch loop checks ``names()`` and answers it as an error observation,
  so ``dispatch`` never sees one. Called directly with an unregistered
  name it still raises KeyError.
- No JSON-schema validation of tool_input against input_schema.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class Tool:
    """A single tool's schema and the async handler that executes it."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[str]]


class ToolRegistry:
    """Holds registered tools, keyed by name, in registration order."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add *tool* to the registry.

        Raises ValueError if a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        """Return each registered tool's schema in Anthropic format.

        Each entry is {"name", "description", "input_schema"} — the handler
        is excluded. Order matches registration order.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]

    def names(self) -> list[str]:
        """Return registered tool names in registration order."""
        return list(self._tools)

    async def dispatch(self, name: str, tool_input: dict[str, Any]) -> str:
        """Await the handler registered under *name* with *tool_input*.

        Raises KeyError if no tool is registered under *name*.
        """
        tool = self._tools[name]
        return await tool.handler(tool_input)


async def _get_current_time(tool_input: dict[str, Any]) -> str:
    """Return the current UTC date and time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


GET_CURRENT_TIME = Tool(
    name="get_current_time",
    description="Returns the current UTC date and time in ISO 8601 format.",
    input_schema={"type": "object", "properties": {}},
    handler=_get_current_time,
)


@functools.lru_cache(maxsize=1)
def get_registry() -> ToolRegistry:
    """Return the process-wide singleton ToolRegistry.

    Registers the seed ``get_current_time`` tool (#7) before returning.
    FastAPI dependency — tests override this via ``app.dependency_overrides``.
    """
    registry = ToolRegistry()
    registry.register(GET_CURRENT_TIME)
    return registry
