"""Tests for ToolRegistry — TDD: written BEFORE tools.py exists.

All test functions are async (asyncio_mode = "auto" handles the event loop).
"""
import pytest

from app.tools import Tool, ToolRegistry


async def _echo_handler(tool_input: dict) -> str:
    return f"echo:{tool_input}"


def _make_tool(name: str = "echo", handler=_echo_handler) -> Tool:
    return Tool(
        name=name,
        description="Echoes the input back.",
        input_schema={"type": "object", "properties": {}},
        handler=handler,
    )


# ---------------------------------------------------------------------------
# Test 1: schemas() returns exactly {"name", "description", "input_schema"},
# handler excluded
# ---------------------------------------------------------------------------


async def test_schemas_excludes_handler_and_has_anthropic_keys():
    registry = ToolRegistry()
    registry.register(_make_tool())

    schemas = registry.schemas()

    assert len(schemas) == 1
    assert set(schemas[0].keys()) == {"name", "description", "input_schema"}
    assert schemas[0]["name"] == "echo"
    assert schemas[0]["description"] == "Echoes the input back."
    assert schemas[0]["input_schema"] == {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# Test 2: schemas() preserves registration order
# ---------------------------------------------------------------------------


async def test_schemas_preserves_registration_order():
    registry = ToolRegistry()
    registry.register(_make_tool(name="first"))
    registry.register(_make_tool(name="second"))
    registry.register(_make_tool(name="third"))

    names = [schema["name"] for schema in registry.schemas()]

    assert names == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# Test 3: register with a duplicate name raises ValueError
# ---------------------------------------------------------------------------


async def test_register_duplicate_name_raises_value_error():
    registry = ToolRegistry()
    registry.register(_make_tool(name="dup"))

    with pytest.raises(ValueError):
        registry.register(_make_tool(name="dup"))


# ---------------------------------------------------------------------------
# Test 4: dispatch awaits the matching tool's handler and returns its result
# ---------------------------------------------------------------------------


async def test_dispatch_invokes_handler_with_input_and_returns_result():
    received = {}

    async def handler(tool_input: dict) -> str:
        received.update(tool_input)
        return "handler result"

    registry = ToolRegistry()
    registry.register(_make_tool(name="capture", handler=handler))

    result = await registry.dispatch("capture", {"key": "value"})

    assert result == "handler result"
    assert received == {"key": "value"}


# ---------------------------------------------------------------------------
# Test 5: dispatch with an unregistered name raises KeyError
# ---------------------------------------------------------------------------


async def test_dispatch_unknown_name_raises_key_error():
    registry = ToolRegistry()

    with pytest.raises(KeyError):
        await registry.dispatch("nonexistent", {})


# ---------------------------------------------------------------------------
# Test 6: a fresh registry's schemas() is []
# ---------------------------------------------------------------------------


async def test_fresh_registry_schemas_is_empty():
    registry = ToolRegistry()
    assert registry.schemas() == []
