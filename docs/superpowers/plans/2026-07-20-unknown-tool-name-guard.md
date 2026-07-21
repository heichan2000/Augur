# Unknown Tool Name Guard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the model requests a tool that isn't registered, answer it as a `tool_result` with `is_error: True` and let the turn complete normally, instead of raising `KeyError` and failing the whole turn as `internal`.

**Architecture:** `ToolRegistry` gains a read-only `names()` accessor and stays a pure lookup — it gets no error-handling responsibility, and `dispatch` keeps raising `KeyError`. The agent's dispatch loop checks membership *before* dispatching and, on a miss, appends an error `tool_result` and `continue`s so every `tool_use` in a parallel batch still gets exactly one result. One structured WARNING per miss goes to a new `"augur.agent"` logger.

**Tech Stack:** Python 3.12+, pytest with `asyncio_mode = "auto"`, stdlib `logging`. No new dependencies.

**Spec:** GitHub issue [heichan2000/Augur#28](https://github.com/heichan2000/Augur/issues/28). Research backing the decision: `docs/research/claude-tool-calling.md`.

## Global Constraints

These come from the spec and apply to **every** task. Violating one is what causes the follow-on conflict with Phase 3.

- **Run tests with `uv run pytest` from the `backend/` directory.** Bare `python -m pytest` fails at collection with `ModuleNotFoundError: No module named 'anthropic'` — the dependencies live only in the uv venv. Baseline before any change: **88 passed**.
- **Check membership; never wrap `dispatch` in `try/except KeyError`.** A handler that raises `KeyError` internally is a Phase-3 concern and must keep propagating. Catching around `dispatch` would silently swallow a real handler bug as "tool not found".
- **`continue`, never `break`.** Every `tool_use` in a step gets exactly one `tool_result`, in the same order as `tool_uses`.
- **`is_error: True` goes on the error block only.** Successful `tool_result` blocks keep their exact current shape — no `is_error` key.
- **Do not change `Tool.handler`'s `-> Awaitable[str]` signature.** A richer handler return type is Phase-3 scope.
- **Do not call `log_turn_error`.** Its `error_type` mirrors an SSE `error` event type; the point of this change is that the turn no longer fails.
- **Do not change the SSE event contract.** No new event types or fields. The `tool_use` event for the unregistered tool is still emitted before dispatch (`agent.py:83`) — that is accepted, not a defect.
- **Error message format, exactly:** `Error: tool '<name>' not found. Available tools: <names>.` — names comma-separated in registration order; `(none)` when the registry is empty.
- Commit style is conventional commits, matching recent history (`fix(chat): ...`, `feat(chat): ...`).

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `backend/app/tools.py` | Modify | Add `names()`; amend the Phase-scope docstring. `dispatch` untouched. |
| `backend/app/observability.py` | Modify | Whitelist `tool_name` in `_STRUCTURED_FIELDS`. |
| `backend/app/agent.py` | Modify | Module logger + membership guard in `run_turn`'s dispatch loop; amend docstring. |
| `backend/tests/test_tools.py` | Modify | Unit tests for `names()`. |
| `backend/tests/test_observability.py` | Modify | Formatter test for `tool_name`. |
| `backend/tests/test_agent.py` | Modify | `tool_results` message shape at the faked provider boundary. |
| `backend/tests/test_chat.py` | Modify | Turn-level outcome: SSE events, persisted history, log records. |

Tasks are ordered so the log field is whitelisted (Task 2) before the code that emits it lands (Task 3).

---

### Task 1: `ToolRegistry.names()`

**Files:**
- Modify: `backend/app/tools.py` (docstring at lines 8–11; new method after `schemas()`, which ends at line 58)
- Test: `backend/tests/test_tools.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ToolRegistry.names() -> list[str]` — registered tool names in registration order, as a fresh list. Task 3 calls this.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_tools.py`. The existing `_make_tool` helper at the top of that file is used as-is.

```python
# ---------------------------------------------------------------------------
# Test 9: names() returns registered names in registration order
# ---------------------------------------------------------------------------


async def test_names_returns_registered_names_in_order():
    registry = ToolRegistry()
    registry.register(_make_tool(name="first"))
    registry.register(_make_tool(name="second"))
    registry.register(_make_tool(name="third"))

    assert registry.names() == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# Test 10: names() returns a fresh list — mutating it does not touch the registry
# ---------------------------------------------------------------------------


async def test_names_returns_a_copy_the_caller_may_mutate():
    registry = ToolRegistry()
    registry.register(_make_tool(name="echo"))

    names = registry.names()
    names.append("injected")

    assert registry.names() == ["echo"]


# ---------------------------------------------------------------------------
# Test 11: a fresh registry's names() is []
# ---------------------------------------------------------------------------


async def test_fresh_registry_names_is_empty():
    assert ToolRegistry().names() == []
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd backend && uv run pytest tests/test_tools.py -k names -v
```

Expected: 3 FAILED with `AttributeError: 'ToolRegistry' object has no attribute 'names'`.

- [ ] **Step 3: Add the method**

In `backend/app/tools.py`, insert directly after the `schemas()` method (after line 58, before `async def dispatch`):

```python
    def names(self) -> list[str]:
        """Return registered tool names in registration order."""
        return list(self._tools)
```

- [ ] **Step 4: Amend the module docstring**

In `backend/app/tools.py`, replace this line (line 9):

```python
- No catching of handler exceptions / error-as-observation (Phase 3).
```

with:

```python
- No catching of handler *exceptions* as model observations (Phase 3). An
  unknown tool *name* is a different case and is already handled: the
  dispatch loop checks ``names()`` and answers it as an error observation,
  so ``dispatch`` never sees one. Called directly with an unregistered
  name it still raises KeyError.
```

- [ ] **Step 5: Run the full suite**

```bash
cd backend && uv run pytest -q
```

Expected: `91 passed` (88 baseline + 3 new). In particular `test_dispatch_unknown_name_raises_key_error` still passes — `dispatch` was not touched.

- [ ] **Step 6: Commit**

```bash
git add backend/app/tools.py backend/tests/test_tools.py
git commit -m "feat(tools): expose registered tool names for the dispatch loop"
```

---

### Task 2: Whitelist `tool_name` in the structured formatter

**Files:**
- Modify: `backend/app/observability.py:21-28`
- Test: `backend/tests/test_observability.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `StructuredFormatter` renders a record's `tool_name` attribute into its JSON output. Task 3 emits records carrying that field.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_observability.py`. It uses the existing `_make_record` helper defined in that file.

```python
# ---------------------------------------------------------------------------
# Behavior 8: StructuredFormatter renders tool_name
# ---------------------------------------------------------------------------


def test_formatter_renders_tool_name_field():
    formatter = StructuredFormatter()
    record = _make_record(
        msg="unknown tool requested",
        level=logging.WARNING,
        extra={"tool_name": "nonexistent"},
    )

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "WARNING"
    assert payload["message"] == "unknown tool requested"
    assert payload["tool_name"] == "nonexistent"
    assert set(payload.keys()) == {"level", "logger", "message", "tool_name"}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd backend && uv run pytest tests/test_observability.py::test_formatter_renders_tool_name_field -v
```

Expected: FAILED with `KeyError: 'tool_name'` — the formatter drops the unwhitelisted field.

- [ ] **Step 3: Add the field to the whitelist**

In `backend/app/observability.py`, replace the `_STRUCTURED_FIELDS` tuple (lines 21–28):

```python
_STRUCTURED_FIELDS = (
    "session_id",
    "model",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "error_type",
    "tool_name",
)
```

- [ ] **Step 4: Run the observability tests**

```bash
cd backend && uv run pytest tests/test_observability.py -q
```

Expected: all pass — the file's own tests plus the new one. Full-suite total is now `92 passed`. `test_formatter_does_not_leak_unwhitelisted_attributes` still passes — it asserts on a record that has no `tool_name` attribute, and absent fields are omitted.

- [ ] **Step 5: Commit**

```bash
git add backend/app/observability.py backend/tests/test_observability.py
git commit -m "feat(observability): whitelist tool_name in structured log output"
```

---

### Task 3: The membership guard in `run_turn`

**Files:**
- Modify: `backend/app/agent.py` (docstring line 10; imports around line 13–20; dispatch loop at lines 115–121)
- Test: `backend/tests/test_agent.py`

**Interfaces:**
- Consumes: `ToolRegistry.names() -> list[str]` from Task 1; the `tool_name` whitelist from Task 2.
- Produces: `run_turn` appends, for an unregistered `tool_use`, the block
  `{"type": "tool_result", "tool_use_id": <id>, "content": "Error: tool '<name>' not found. Available tools: <names>.", "is_error": True}`
  and emits one WARNING `"unknown tool requested"` on logger `"augur.agent"` with `tool_name` set. Task 4 asserts on both.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_agent.py`. Uses the existing `FakeProvider` and `_make_registry` helpers from that file.

```python
# ---------------------------------------------------------------------------
# Test: an unknown tool name becomes an error observation, turn continues
# ---------------------------------------------------------------------------


async def test_unknown_tool_yields_error_tool_result_and_turn_continues():
    registry = _make_registry()
    provider = FakeProvider(
        [
            [
                ToolUseRequested(id="t1", name="nonexistent", input={}),
                TurnComplete(stop_reason="tool_use", input_tokens=5, output_tokens=4),
            ],
            [
                TextDelta("sorry"),
                TurnComplete(stop_reason="end_turn", input_tokens=6, output_tokens=2),
            ],
        ]
    )
    messages = [{"role": "user", "content": "use a tool"}]

    events = [
        event
        async for event in run_turn(provider=provider, registry=registry, messages=messages)
    ]

    # The turn ran a second step instead of blowing up.
    assert len(provider.received_calls) == 2
    assert events[-1] == TurnComplete(
        stop_reason="end_turn", input_tokens=11, output_tokens=6
    )
    assert messages[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": "Error: tool 'nonexistent' not found. Available tools: echo.",
                "is_error": True,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Test: a mixed batch answers both calls, in order, and still runs the known one
# ---------------------------------------------------------------------------


async def test_mixed_known_and_unknown_returns_both_results_in_order():
    handler_inputs = []

    async def handler(tool_input: dict) -> str:
        handler_inputs.append(tool_input)
        return "echo-result"

    registry = _make_registry(handler=handler)
    provider = FakeProvider(
        [
            [
                ToolUseRequested(id="t1", name="echo", input={"v": 1}),
                ToolUseRequested(id="t2", name="nope", input={}),
                TurnComplete(stop_reason="tool_use", input_tokens=1, output_tokens=1),
            ],
            [
                TextDelta("done"),
                TurnComplete(stop_reason="end_turn", input_tokens=1, output_tokens=1),
            ],
        ]
    )
    messages = [{"role": "user", "content": "use both"}]

    [event async for event in run_turn(provider=provider, registry=registry, messages=messages)]

    # The registered tool still ran, exactly once, with its input.
    assert handler_inputs == [{"v": 1}]
    # Two results, in tool_uses order. Exact equality also pins the successful
    # block's shape: no is_error key on it.
    assert messages[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "echo-result"},
        {
            "type": "tool_result",
            "tool_use_id": "t2",
            "content": "Error: tool 'nope' not found. Available tools: echo.",
            "is_error": True,
        },
    ]


# ---------------------------------------------------------------------------
# Test: with nothing registered, the message reads "(none)"
# ---------------------------------------------------------------------------


async def test_unknown_tool_with_empty_registry_lists_none():
    registry = ToolRegistry()
    provider = FakeProvider(
        [
            [
                ToolUseRequested(id="t1", name="nope", input={}),
                TurnComplete(stop_reason="tool_use", input_tokens=1, output_tokens=1),
            ],
            [
                TextDelta("done"),
                TurnComplete(stop_reason="end_turn", input_tokens=1, output_tokens=1),
            ],
        ]
    )
    messages = [{"role": "user", "content": "use a tool"}]

    [event async for event in run_turn(provider=provider, registry=registry, messages=messages)]

    assert messages[2]["content"][0]["content"] == (
        "Error: tool 'nope' not found. Available tools: (none)."
    )


# ---------------------------------------------------------------------------
# Test: a handler raising KeyError internally is NOT mistaken for unknown-tool
# ---------------------------------------------------------------------------


async def test_handler_raising_key_error_still_propagates():
    async def exploding_handler(tool_input: dict) -> str:
        raise KeyError("something the handler looked up")

    registry = _make_registry(handler=exploding_handler)
    provider = FakeProvider(
        [
            [
                ToolUseRequested(id="t1", name="echo", input={}),
                TurnComplete(stop_reason="tool_use", input_tokens=1, output_tokens=1),
            ],
        ]
    )
    messages = [{"role": "user", "content": "use the tool"}]

    with pytest.raises(KeyError):
        [
            event
            async for event in run_turn(
                provider=provider, registry=registry, messages=messages
            )
        ]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd backend && uv run pytest tests/test_agent.py -k "unknown or mixed or key_error" -v
```

Expected: the three unknown-tool tests FAIL with `KeyError: 'nonexistent'` / `KeyError: 'nope'` escaping `run_turn`. `test_handler_raising_key_error_still_propagates` **already passes** — it pins behaviour that must not regress.

- [ ] **Step 3: Add the module logger**

In `backend/app/agent.py`, add `import logging` to the stdlib import block and define the logger below the imports. After the change, lines 13–24 read:

```python
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator

from app.config import AGENT_MAX_STEPS
from app.provider import ProviderEvent, TextDelta, ToolUseRequested, TurnComplete
from app.tools import ToolRegistry

logger = logging.getLogger("augur.agent")

Message = dict[str, Any]  # Anthropic-format: {"role": ..., "content": ...}
```

`"augur.agent"` propagates to the `"augur"` logger that `configure_logging` already handles, so no wiring is needed.

- [ ] **Step 4: Replace the dispatch loop**

In `backend/app/agent.py`, replace lines 115–121:

```python
        tool_results: list[dict[str, Any]] = []
        for tool_use in tool_uses:
            result = await registry.dispatch(tool_use.name, tool_use.input)
            tool_results.append(
                {"type": "tool_result", "tool_use_id": tool_use.id, "content": result}
            )
        messages.append({"role": "user", "content": tool_results})
```

with:

```python
        tool_results: list[dict[str, Any]] = []
        known = registry.names()
        for tool_use in tool_uses:
            if tool_use.name not in known:
                # Answer the hallucinated name as an observation rather than
                # failing the turn: the model corrects itself from this, and
                # every sibling call in the batch still gets its result.
                logger.warning(
                    "unknown tool requested",
                    extra={"tool_name": tool_use.name},
                )
                available = ", ".join(known) or "(none)"
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": (
                            f"Error: tool '{tool_use.name}' not found. "
                            f"Available tools: {available}."
                        ),
                        "is_error": True,
                    }
                )
                continue
            result = await registry.dispatch(tool_use.name, tool_use.input)
            tool_results.append(
                {"type": "tool_result", "tool_use_id": tool_use.id, "content": result}
            )
        messages.append({"role": "user", "content": tool_results})
```

Note the f-string is split across two lines; the rendered message has exactly one space after the full stop.

- [ ] **Step 5: Amend the module docstring**

In `backend/app/agent.py`, replace this line (line 10):

```python
- No catching of handler exceptions as model observations (Phase 3).
```

with:

```python
- No catching of handler *exceptions* as model observations (Phase 3). An
  unknown tool *name* is handled here, not deferred: the dispatch loop
  checks membership before dispatching and feeds back a ``tool_result``
  with ``is_error`` set, so one hallucinated name cannot fail the turn.
```

- [ ] **Step 6: Run the agent tests, then the full suite**

```bash
cd backend && uv run pytest tests/test_agent.py -q && uv run pytest -q
```

Expected: agent tests all pass; full suite `96 passed` (92 after Task 2 + 4 new here, one of which already passed at Step 2).

- [ ] **Step 7: Commit**

```bash
git add backend/app/agent.py backend/tests/test_agent.py
git commit -m "fix(agent): answer an unknown tool name instead of failing the turn"
```

---

### Task 4: Turn-level behaviour through `stream_chat`

**Files:**
- Test only: `backend/tests/test_chat.py`

No production code changes. This task verifies the acceptance criteria that only exist at the turn boundary — the SSE contract, what reaches storage, and the log records — using the real `stream_chat`, a real `InMemoryConversationStore`, and a real `ToolRegistry`.

**Interfaces:**
- Consumes: the guard's behaviour from Task 3, plus the existing `FakeProvider`, `_make_registry`, and `_parse_sse` helpers in `test_chat.py`.
- Produces: nothing consumed downstream.

- [ ] **Step 1: Write the tests**

Append to `backend/tests/test_chat.py`. `logging` and `InMemoryConversationStore` are already imported at the top of that file.

```python
# ---------------------------------------------------------------------------
# Behavior: an unknown tool name completes the turn instead of erroring it
# ---------------------------------------------------------------------------


def _unknown_tool_provider() -> FakeProvider:
    """Step 1 requests an unregistered tool; step 2 answers in text."""
    return FakeProvider(
        [
            [
                ToolUseRequested(id="t1", name="nonexistent", input={}),
                TurnComplete(stop_reason="tool_use", input_tokens=5, output_tokens=4),
            ],
            [
                TextDelta("I don't have that tool."),
                TurnComplete(stop_reason="end_turn", input_tokens=6, output_tokens=2),
            ],
        ]
    )


async def test_unknown_tool_ends_in_done_with_no_error_event():
    provider = _unknown_tool_provider()
    registry = _make_registry()
    store = InMemoryConversationStore()

    chunks = [
        c
        async for c in stream_chat(
            provider=provider,
            registry=registry,
            store=store,
            session_id="s1",
            message="use a tool",
            model="claude-sonnet-4-6",
        )
    ]

    events = _parse_sse(chunks)
    assert [name for name, _ in events] == ["tool_use", "token", "done"]
    # The tool_use event is still emitted for the unregistered tool: it is
    # sent before dispatch, and that is accepted behaviour.
    assert events[0] == ("tool_use", {"id": "t1", "name": "nonexistent", "input": {}})


async def test_unknown_tool_turn_persists_the_call_and_its_error_result():
    provider = _unknown_tool_provider()
    registry = _make_registry()
    store = InMemoryConversationStore()

    [
        c
        async for c in stream_chat(
            provider=provider,
            registry=registry,
            store=store,
            session_id="s1",
            message="use a tool",
            model="claude-sonnet-4-6",
        )
    ]

    history = await store.get_history("s1")

    # The assistant message carrying the unknown tool_use survives
    # persistable_messages because it now has a matching tool_result.
    assert len(history) == 4
    assert history[0] == {"role": "user", "content": "use a tool"}
    assert history[1]["content"] == [
        {"type": "tool_use", "id": "t1", "name": "nonexistent", "input": {}}
    ]
    assert history[2]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": "Error: tool 'nonexistent' not found. Available tools: echo.",
            "is_error": True,
        }
    ]
    assert history[3]["content"] == [
        {"type": "text", "text": "I don't have that tool."}
    ]


async def test_unknown_tool_logs_one_structured_warning(caplog):
    provider = _unknown_tool_provider()
    registry = _make_registry()
    store = InMemoryConversationStore()

    with caplog.at_level(logging.WARNING, logger="augur.agent"):
        [
            c
            async for c in stream_chat(
                provider=provider,
                registry=registry,
                store=store,
                session_id="s1",
                message="use a tool",
                model="claude-sonnet-4-6",
            )
        ]

    records = [r for r in caplog.records if r.name == "augur.agent"]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].msg == "unknown tool requested"
    assert records[0].tool_name == "nonexistent"


async def test_unknown_tool_logs_usage_as_a_completed_turn_not_an_error(caplog):
    provider = _unknown_tool_provider()
    registry = _make_registry()
    store = InMemoryConversationStore()

    with caplog.at_level(logging.INFO, logger="augur.observability"):
        [
            c
            async for c in stream_chat(
                provider=provider,
                registry=registry,
                store=store,
                session_id="s1",
                message="use a tool",
                model="claude-sonnet-4-6",
            )
        ]

    records = [r for r in caplog.records if r.name == "augur.observability"]
    assert len(records) == 1
    # log_turn_usage, not log_turn_error — the turn did not fail.
    assert records[0].msg == "turn usage"
    assert not hasattr(records[0], "error_type")
    assert records[0].input_tokens == 11
    assert records[0].output_tokens == 6
```

- [ ] **Step 2: Run them**

```bash
cd backend && uv run pytest tests/test_chat.py -k unknown_tool -v
```

Expected: 4 PASSED. These describe behaviour Task 3 already implemented, so they pass on first run — they are the acceptance-criteria net, not a red-green cycle. **If any fails, the guard is wrong; fix `agent.py`, not the test.**

- [ ] **Step 3: Run the full suite**

```bash
cd backend && uv run pytest -q
```

Expected: `100 passed` (96 after Task 3 + 4 new). Zero failures — every pre-existing test passes unchanged.

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_chat.py
git commit -m "test(chat): cover the unknown-tool turn end to end"
```

---

## Acceptance Criteria Coverage

Each criterion from issue #28, mapped to where it is verified.

| Acceptance criterion | Verified by |
|---|---|
| Turn completes with `done`, no `error` event | Task 4 · `test_unknown_tool_ends_in_done_with_no_error_event` |
| `is_error: True` + exact message, names in registration order | Task 3 · `test_unknown_tool_yields_error_tool_result_and_turn_continues` |
| Empty registry reads `Available tools: (none).` | Task 3 · `test_unknown_tool_with_empty_registry_lists_none` |
| Mixed batch → two results in order, known handler still runs | Task 3 · `test_mixed_known_and_unknown_returns_both_results_in_order` |
| Handler `KeyError` still fails the turn as `internal` | Task 3 · `test_handler_raising_key_error_still_propagates` (escapes `run_turn`) + pre-existing `test_unexpected_exception_yields_internal_event` (`stream_chat` maps it to `internal`) |
| Successful `tool_result` byte-identical (no `is_error`) | Task 3 · `test_mixed_known_and_unknown_...` (exact dict equality) |
| One structured WARNING with `tool_name`, surviving the formatter | Task 4 · `test_unknown_tool_logs_one_structured_warning` + Task 2 · `test_formatter_renders_tool_name_field` |
| `log_turn_error` not called; usage logged as a completed turn | Task 4 · `test_unknown_tool_logs_usage_as_a_completed_turn_not_an_error` |
| Assistant message survives `persistable_messages` | Task 4 · `test_unknown_tool_turn_persists_the_call_and_its_error_result` |
| No SSE contract change; `tool_use` event still emitted | Task 4 · `test_unknown_tool_ends_in_done_with_no_error_event` (asserts the exact event list) |
| `dispatch` still raises `KeyError`; `handler` still `-> Awaitable[str]` | Existing `test_dispatch_unknown_name_raises_key_error`, unchanged; Task 1 leaves `dispatch` untouched |
| Existing tests in `backend/tests/` pass unchanged | Task 4 · Step 3 full-suite run (`100 passed`) |

## Known Interaction, Not a Defect

If the unknown tool arrives on the **final** step, `agent.py:109-113` breaks out of the loop before the dispatch block, so no `tool_result` is written and `persistable_messages` drops the unanswered call. This is pre-existing `max_steps` behaviour, unchanged by this work, and consistent with the criteria above. Do not "fix" it here.

## Out of Scope

Do not widen the change into any of these — they would collide with Phase 3:

- Catching **handler** exceptions as observations (Phase 3, tracked by the amended docstrings)
- Changing `Tool.handler`'s return type, or adding `is_error` to successful results
- JSON-schema validation of `tool_input`, or `strict: true` tool definitions
- Concurrency — dispatch stays a sequential `for` loop
- Anything touching `stop_reason` or truncation — that is #27
