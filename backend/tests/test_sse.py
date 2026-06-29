"""Tests for the SSE event contract — typed events + serializer.

TDD: these tests are written BEFORE sse.py exists, so they will fail
on import until the module is created.
"""
import json
import pytest

from app.sse import (
    TokenEvent,
    ToolUseEvent,
    ErrorEvent,
    DoneEvent,
    SSEEvent,
    format_sse,
)


# ---------------------------------------------------------------------------
# Basic exact-wire-string assertions
# ---------------------------------------------------------------------------


def test_token_event_exact_wire_string():
    event = TokenEvent(text="Hi")
    result = format_sse(event)
    assert result == 'event: token\ndata: {"text":"Hi"}\n\n'


def test_tool_use_event_exact_wire_string():
    event = ToolUseEvent(id="call_abc", name="search", input={"query": "hello"})
    result = format_sse(event)
    assert result == 'event: tool_use\ndata: {"id":"call_abc","name":"search","input":{"query":"hello"}}\n\n'


def test_error_event_exact_wire_string():
    event = ErrorEvent(type="rate_limit", message="Too many requests")
    result = format_sse(event)
    assert result == 'event: error\ndata: {"type":"rate_limit","message":"Too many requests"}\n\n'


def test_done_event_exact_wire_string():
    event = DoneEvent()
    result = format_sse(event)
    assert result == "event: done\ndata: {}\n\n"


# ---------------------------------------------------------------------------
# Field-name assertions for tool_use
# ---------------------------------------------------------------------------


def test_tool_use_has_required_fields():
    event = ToolUseEvent(id="x", name="y", input={"a": 1})
    data = json.loads(event.model_dump_json())
    assert "id" in data
    assert "name" in data
    assert "input" in data
    assert data["input"] == {"a": 1}


def test_tool_use_nested_input_serializes():
    nested = {"key": "value", "nested": {"deep": [1, 2, 3]}}
    event = ToolUseEvent(id="t1", name="tool", input=nested)
    result = format_sse(event)
    # The data line should be valid JSON containing the nested structure
    data_line = result.split("\n")[1]
    assert data_line.startswith("data: ")
    parsed = json.loads(data_line[len("data: "):])
    assert parsed["input"] == nested


# ---------------------------------------------------------------------------
# Newline-escaping test (critical for SSE framing)
# ---------------------------------------------------------------------------


def test_token_with_newline_and_quote_no_raw_newline_in_data_line():
    """A token whose text contains a raw newline and a double-quote must not
    break SSE framing. The data: line must contain NO raw newline — the
    embedded newline must be JSON-escaped as \\n."""
    text_with_newline_and_quote = 'Hello\nWorld "quoted"'
    event = TokenEvent(text=text_with_newline_and_quote)
    result = format_sse(event)

    # Split on physical newlines
    lines = result.split("\n")
    # Expected lines: ["event: token", 'data: {"text":"Hello\\nWorld \\"quoted\\""}', "", ""]
    # The last split produces an empty string after the final \n
    assert lines[0] == "event: token"
    assert lines[1].startswith("data: ")
    # Line 2 must be blank (the SSE blank-line terminator)
    assert lines[2] == ""
    # The data line itself must not span multiple physical lines
    # i.e., there should be exactly 4 items when split by \n (event, data, blank, trailing empty)
    assert len(lines) == 4, f"Expected 4 lines, got {len(lines)}: {lines!r}"

    # Parse the JSON and verify the text round-trips correctly
    data_line = lines[1][len("data: "):]
    parsed = json.loads(data_line)
    assert parsed["text"] == text_with_newline_and_quote


# ---------------------------------------------------------------------------
# JSON compactness assertions
# ---------------------------------------------------------------------------


def test_token_json_is_compact_no_spaces():
    """The JSON output must be compact: no spaces after : or ,"""
    event = TokenEvent(text="Hi")
    result = format_sse(event)
    data_line = result.split("\n")[1]
    json_part = data_line[len("data: "):]
    assert json_part == '{"text":"Hi"}', f"Expected compact JSON, got: {json_part!r}"


def test_done_json_is_empty_object():
    """Done event must produce exactly {} — no fields."""
    event = DoneEvent()
    result = format_sse(event)
    data_line = result.split("\n")[1]
    json_part = data_line[len("data: "):]
    assert json_part == "{}", f"Expected {{}}, got: {json_part!r}"


def test_error_json_is_compact():
    """Error event JSON must be compact."""
    event = ErrorEvent(type="internal", message="oops")
    result = format_sse(event)
    data_line = result.split("\n")[1]
    json_part = data_line[len("data: "):]
    assert json_part == '{"type":"internal","message":"oops"}'


# ---------------------------------------------------------------------------
# Event-name inheritance
# ---------------------------------------------------------------------------


def test_event_names_match_contract():
    assert TokenEvent.event == "token"
    assert ToolUseEvent.event == "tool_use"
    assert ErrorEvent.event == "error"
    assert DoneEvent.event == "done"


def test_all_events_are_subclass_of_sse_event():
    for cls in (TokenEvent, ToolUseEvent, ErrorEvent, DoneEvent):
        assert issubclass(cls, SSEEvent), f"{cls} must subclass SSEEvent"


# ---------------------------------------------------------------------------
# format_sse output structure
# ---------------------------------------------------------------------------


def test_format_sse_always_ends_with_double_newline():
    for event in [
        TokenEvent(text="x"),
        ToolUseEvent(id="i", name="n", input={}),
        ErrorEvent(type="internal", message="e"),
        DoneEvent(),
    ]:
        result = format_sse(event)
        assert result.endswith("\n\n"), f"format_sse output must end with \\n\\n, got: {result!r}"


def test_format_sse_first_line_is_event_line():
    event = TokenEvent(text="test")
    result = format_sse(event)
    first_line = result.split("\n")[0]
    assert first_line == "event: token"


def test_format_sse_second_line_is_data_line():
    event = TokenEvent(text="test")
    result = format_sse(event)
    second_line = result.split("\n")[1]
    assert second_line.startswith("data: ")
