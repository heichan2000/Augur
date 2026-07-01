"""Tool registry for Augur's tool-calling spine.

Holds a schema + async handler per tool, keyed by name. Pure unit — no
provider, no FastAPI, no SSE imports. The dispatch loop (a later piece)
calls ``schemas()`` to advertise available tools to the model and
``dispatch()`` to run the handler the model selected.

Phase scope (future concerns, not implemented here):
- No catching of handler exceptions / error-as-observation (Phase 3).
- No JSON-schema validation of tool_input against input_schema.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
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

    async def dispatch(self, name: str, tool_input: dict[str, Any]) -> str:
        """Await the handler registered under *name* with *tool_input*.

        Raises KeyError if no tool is registered under *name*.
        """
        tool = self._tools[name]
        return await tool.handler(tool_input)


@functools.lru_cache(maxsize=1)
def get_registry() -> ToolRegistry:
    """Return the process-wide singleton ToolRegistry.

    Empty for now — a later piece (#7) registers concrete tools here.
    FastAPI dependency — tests override this via ``app.dependency_overrides``.
    """
    return ToolRegistry()
