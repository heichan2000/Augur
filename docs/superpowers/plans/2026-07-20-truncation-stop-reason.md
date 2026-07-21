# Surface `max_tokens` Truncation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A turn cut off by the token ceiling tells the user it was cut off, and a tool call truncated mid-JSON no longer crashes the stream or dispatches with empty input.

**Architecture:** The provider stops yielding tool blocks the moment it parses them; it stashes them and replays them only after `message_delta` has revealed `stop_reason`, dropping them if the turn was truncated. The terminal `stop_reason` then rides out on the `done` SSE event verbatim as `str | None`, and the frontend keeps it in a field orthogonal to the turn's status, so an unrecognised value renders as a normal complete turn.

**Tech Stack:** Python 3.12 / FastAPI / pydantic-settings / pytest (`asyncio_mode = "auto"`); Next.js 16 / React 19 / TypeScript / Vitest / Testing Library.

## Global Constraints

- The truncation value set is exactly `{"max_tokens", "model_context_window_exceeded"}` — spelled identically in `backend/app/provider.py` and `frontend/src/components/turn-error.tsx`.
- `stop_reason` is passed through verbatim as `str | None`. Never derive a boolean, never widen `AssistantTurnStatus`, never validate against an allow-list on the wire.
- An unrecognised `stop_reason` must render as a normal complete turn.
- Default `max_tokens` is `2048`. Behaviour must be unchanged out of the box.
- The truncation notice is informational: no Retry button, no Continue button, no `role="alert"`, and no `UnsavedNotice` (the turn *is* saved).
- Notice copy, verbatim: `response cut off at the length limit · saved to the conversation`
- Backend tests run as `cd backend && .venv/Scripts/python.exe -m pytest <path> -v`.
- Frontend tests run as `cd frontend && npm test -- <path>`.

---

## File Structure

**Backend**

| File | Responsibility | Change |
| --- | --- | --- |
| `backend/app/config.py` | Settings | Add `anthropic_max_tokens` |
| `backend/.env.example` | Document env vars | Add `ANTHROPIC_MAX_TOKENS` |
| `backend/app/provider.py` | Normalize SDK stream → domain events | Own the truncation rule; stash-and-replay tool blocks; guard the JSON parse |
| `backend/app/agent.py` | Agent loop | Docstring only |
| `backend/app/sse.py` | Wire contract | `DoneEvent.stop_reason` |
| `backend/app/chat.py` | SSE body generator | Pass the terminal stop reason to `DoneEvent` |
| `docs/sse-contract.md` | The contract of record | Document the new field |

**Frontend**

| File | Responsibility | Change |
| --- | --- | --- |
| `frontend/src/lib/sse.ts` | Wire types | `done` payload gains optional `stop_reason` |
| `frontend/src/lib/chat-state.ts` | Reducer | `AssistantTurn.stopReason` |
| `frontend/src/components/turn-error.tsx` | Turn-ending notices | `TruncatedNotice` + `isTruncated` |
| `frontend/src/components/turns.tsx` | Assistant turn layout | Render the notice |

The truncation predicate lives beside the component that consumes it (`turn-error.tsx`), not in `chat-state.ts`: the reducer stores the value, the view decides what it means. That is the whole point of keeping `stopReason` out of the status enum.

---

### Task 1: `max_tokens` becomes configurable

`max_tokens` is currently a hardcoded default on `stream_turn` that no call site passes. Move it onto the provider where `model` already lives, and source it from settings.

**Files:**
- Modify: `backend/app/config.py:41-45`
- Modify: `backend/.env.example`
- Modify: `backend/app/provider.py:64-81`, `backend/app/provider.py:154-157`
- Test: `backend/tests/test_config.py`, `backend/tests/test_provider.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `Settings.anthropic_max_tokens: int` (default `2048`)
  - `AnthropicProvider.__init__(self, client: anthropic.AsyncAnthropic, model: str, max_tokens: int) -> None` — `max_tokens` is **required**, mirroring `model`
  - `AnthropicProvider.stream_turn(*, messages, system=None, tools=None)` — the `max_tokens` parameter is **removed**
  - `backend/tests/test_provider.py::make_provider(client, *, max_tokens: int = 2048) -> AnthropicProvider`

- [ ] **Step 1: Write the failing config tests**

Append to `backend/tests/test_config.py`:

```python
def test_max_tokens_defaults_when_unset(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-abc")
    monkeypatch.delenv("ANTHROPIC_MAX_TOKENS", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.anthropic_max_tokens == 2048


def test_max_tokens_overridden_by_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-abc")
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "4096")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.anthropic_max_tokens == 4096
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && .venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: both new tests FAIL with `AttributeError: 'Settings' object has no attribute 'anthropic_max_tokens'`

- [ ] **Step 3: Add the setting**

In `backend/app/config.py`, replace the `Settings` class body:

```python
class Settings(BaseSettings):
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-4-6"
    # Ceiling on a single provider response. A turn that hits it comes back
    # with stop_reason "max_tokens" and is surfaced to the user as truncated
    # (see app.provider.TRUNCATION_STOP_REASONS) rather than passed off as a
    # finished answer.
    anthropic_max_tokens: int = 2048

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
```

- [ ] **Step 4: Run the config tests to verify they pass**

Run: `cd backend && .venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: 5 passed

- [ ] **Step 5: Document the env var**

Replace the whole of `backend/.env.example` with:

```
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-4-6
# Ceiling on a single provider response, in tokens. A turn that hits this is
# reported to the user as cut off rather than presented as a complete answer.
ANTHROPIC_MAX_TOKENS=2048
```

- [ ] **Step 6: Add the failing provider test**

First add the helper. In `backend/tests/test_provider.py`, immediately after the `collect` helper (currently at line 139-140), add:

```python
def make_provider(client, *, max_tokens: int = 2048) -> AnthropicProvider:
    """Build a provider over a fake client.

    Centralised so the constructor can gain arguments without touching every
    test — 2048 matches the Settings default, so tests that do not care about
    the ceiling read as "the normal configuration".
    """
    return AnthropicProvider(client=client, model=MODEL, max_tokens=max_tokens)
```

Then append this test at the end of the file:

```python
async def test_max_tokens_from_constructor_appears_in_stream_kwargs():
    events = [
        _message_start(input_tokens=1),
        _message_delta(stop_reason="end_turn", output_tokens=1),
        _message_stop(),
    ]
    client = FakeClient(events)
    provider = make_provider(client, max_tokens=4096)

    await collect(provider, messages=[{"role": "user", "content": "hello"}])

    assert client.messages.last_kwargs["max_tokens"] == 4096
```

- [ ] **Step 7: Run it to verify it fails**

Run: `cd backend && .venv/Scripts/python.exe -m pytest tests/test_provider.py::test_max_tokens_from_constructor_appears_in_stream_kwargs -v`
Expected: FAIL with `TypeError: AnthropicProvider.__init__() got an unexpected keyword argument 'max_tokens'`

- [ ] **Step 8: Move `max_tokens` onto the provider**

In `backend/app/provider.py`, replace lines 64-81 (the `__init__` and the head of `stream_turn`) with:

```python
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
```

And replace `build_provider`'s return (line 157) with:

```python
    return AnthropicProvider(
        client=client,
        model=settings.anthropic_model,
        max_tokens=settings.anthropic_max_tokens,
    )
```

Removing the `max_tokens` parameter from `stream_turn` breaks nothing: `run_turn` calls `stream_turn(messages=…, system=…, tools=…)` and no other call site passes it. The fake providers in `test_agent.py`, `test_chat.py` and `test_chat_endpoint.py` declare `max_tokens=2048` in their own signatures, but they are independent stubs rather than subclasses and need no change.

- [ ] **Step 9: Route every test construction through the helper**

In `backend/tests/test_provider.py`, replace each of the eight occurrences of

```python
    provider = AnthropicProvider(client=client, model=MODEL)
```

(at lines 159, 186, 211, 234, 260, 303, 318, 332) with

```python
    provider = make_provider(client)
```

- [ ] **Step 10: Run the full backend suite**

Run: `cd backend && .venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass, no errors

- [ ] **Step 11: Commit**

```bash
git add backend/app/config.py backend/app/provider.py backend/.env.example backend/tests/test_config.py backend/tests/test_provider.py
git commit -m "feat(config): make the response token ceiling configurable"
```

---

### Task 2: The provider drops tool blocks from a truncated turn

The event order is `content_block_stop` → `message_delta` (carrying `stop_reason`) → `message_stop`. At block-close time the provider cannot tell whether an empty buffer means truncation or a genuinely zero-argument tool. So it stashes tool blocks and replays them once `stop_reason` is known.

`run_turn` could decide instead, but by then `stream_chat` has already emitted a `tool_use` SSE event for the dead call and the UI has drawn a tool row that `closeTurn` settles to `"done"` — a tool that never ran, shown as having succeeded. Keeping the rule in the provider means one place owns it and no phantom tool row reaches the UI.

**Files:**
- Modify: `backend/app/provider.py:1-16` (imports + logger), `backend/app/provider.py:87-146` (the stream loop)
- Modify: `backend/app/agent.py:9-15` (docstring)
- Test: `backend/tests/test_provider.py`

**Interfaces:**
- Consumes: `make_provider(client, *, max_tokens=2048)` from Task 1.
- Produces: `app.provider.TRUNCATION_STOP_REASONS: frozenset[str]` — exported for anyone who needs the same rule. `ToolUseRequested`, `TextDelta`, `TurnComplete` are unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_provider.py`:

```python
# ---------------------------------------------------------------------------
# Truncation: a turn cut off by the token ceiling drops its tool blocks
# ---------------------------------------------------------------------------


async def test_truncated_mid_input_json_drops_the_tool_block():
    """A buffer holding half a JSON object must not raise, and must not be
    dispatched. Before the guard this raised JSONDecodeError out of the
    generator, which stream_chat turned into an `internal` error with no
    `done` and nothing persisted."""
    events = [
        _message_start(input_tokens=9),
        _content_block_start_tool(index=0, id="toolu_3", name="get_current_time"),
        _input_json_delta(index=0, partial_json='{"tz":'),
        _content_block_stop(index=0),
        _message_delta(stop_reason="max_tokens", output_tokens=4),
        _message_stop(),
    ]
    client = FakeClient(events)
    provider = make_provider(client)

    results = await collect(provider, messages=[{"role": "user", "content": "what time?"}])

    assert results == [TurnComplete("max_tokens", 9, 4)]


async def test_truncated_with_empty_tool_buffer_dispatches_nothing():
    """Truncated before any input_json_delta arrived. An empty buffer would
    otherwise parse to {} and dispatch a real tool with silently-empty
    input."""
    events = [
        _message_start(input_tokens=8),
        _content_block_start_tool(index=0, id="toolu_4", name="ping"),
        _content_block_stop(index=0),
        _message_delta(stop_reason="max_tokens", output_tokens=2),
        _message_stop(),
    ]
    client = FakeClient(events)
    provider = make_provider(client)

    results = await collect(provider, messages=[{"role": "user", "content": "ping"}])

    assert results == [TurnComplete("max_tokens", 8, 2)]


async def test_context_window_exceeded_drops_the_tool_block_too():
    events = [
        _message_start(input_tokens=8),
        _content_block_start_tool(index=0, id="toolu_5", name="ping"),
        _content_block_stop(index=0),
        _message_delta(stop_reason="model_context_window_exceeded", output_tokens=2),
        _message_stop(),
    ]
    client = FakeClient(events)
    provider = make_provider(client)

    results = await collect(provider, messages=[{"role": "user", "content": "ping"}])

    assert results == [TurnComplete("model_context_window_exceeded", 8, 2)]


async def test_streamed_text_survives_a_truncated_tool_block():
    """Text streamed before the cut-off is the real answer and must reach the
    caller — it is what gets persisted and shown."""
    events = [
        _message_start(input_tokens=9),
        _content_block_start_text(index=0),
        _text_delta(index=0, text="Let me check"),
        _content_block_stop(index=0),
        _content_block_start_tool(index=1, id="toolu_6", name="get_current_time"),
        _input_json_delta(index=1, partial_json='{"tz":'),
        _content_block_stop(index=1),
        _message_delta(stop_reason="max_tokens", output_tokens=4),
        _message_stop(),
    ]
    client = FakeClient(events)
    provider = make_provider(client)

    results = await collect(provider, messages=[{"role": "user", "content": "what time?"}])

    assert results == [TextDelta("Let me check"), TurnComplete("max_tokens", 9, 4)]


async def test_unparseable_buffer_is_dropped_and_the_turn_completes_normally():
    """Malformed JSON without truncation: drop the one block, keep the turn."""
    events = [
        _message_start(input_tokens=9),
        _content_block_start_tool(index=0, id="toolu_7", name="get_current_time"),
        _input_json_delta(index=0, partial_json="not json at all"),
        _content_block_stop(index=0),
        _message_delta(stop_reason="tool_use", output_tokens=4),
        _message_stop(),
    ]
    client = FakeClient(events)
    provider = make_provider(client)

    results = await collect(provider, messages=[{"role": "user", "content": "what time?"}])

    assert results == [TurnComplete("tool_use", 9, 4)]
```

Note what is **not** added here: `test_tool_use_with_no_input_json_delta_yields_empty_dict` (line 201) already pins that an empty buffer with `stop_reason="tool_use"` still yields `ToolUseRequested(input={})`. That existing test is the one the new guard could most easily break, and it is why the guard keys on `stop_reason` rather than on the buffer being empty. Leave it exactly as it is — it is now a regression guard, and so is `test_tool_use_yields_tool_use_requested_and_turn_complete` (line 175) for the stash-and-replay path.

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && .venv/Scripts/python.exe -m pytest tests/test_provider.py -v -k "truncated or unparseable or context_window"`
Expected: 5 tests. `test_truncated_mid_input_json_drops_the_tool_block`, `test_streamed_text_survives_a_truncated_tool_block` and `test_unparseable_buffer_is_dropped_and_the_turn_completes_normally` FAIL with `json.decoder.JSONDecodeError`. `test_truncated_with_empty_tool_buffer_dispatches_nothing` and `test_context_window_exceeded_drops_the_tool_block_too` FAIL on the assertion, with an unexpected `ToolUseRequested(id='toolu_4', name='ping', input={})` in the results.

- [ ] **Step 3: Add the logger and the constant**

In `backend/app/provider.py`, replace lines 6-15 (the imports block) with:

```python
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
```

- [ ] **Step 4: Stash the tool blocks and guard the parse**

In `backend/app/provider.py`, replace lines 87-93 (the accumulated-state comment block) with:

```python
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
```

Then replace the `content_block_stop` branch (lines 120-130) with:

```python
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
```

Finally replace the closing `yield TurnComplete(...)` (lines 142-146) with:

```python
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
```

`TextDelta` still yields live inside the loop, so streaming is unaffected. The cost of stashing is that within a step every `ToolUseRequested` now lands after every `TextDelta`; in practice the API already emits text blocks before `tool_use` blocks, so observable ordering is unchanged.

`tool_name` is already whitelisted in `backend/app/observability.py:31`, so both warnings carry a structured field without further work.

- [ ] **Step 5: Run the provider tests**

Run: `cd backend && .venv/Scripts/python.exe -m pytest tests/test_provider.py -v`
Expected: all pass, including the pre-existing `test_tool_use_with_no_input_json_delta_yields_empty_dict` and `test_tool_use_yields_tool_use_requested_and_turn_complete`

- [ ] **Step 6: Correct the agent docstring**

No code changes in `backend/app/agent.py`. On truncation the provider yields no `ToolUseRequested`, so `tool_uses` is empty, `run_turn` appends the assistant message and breaks, and `final_stop_reason` is already the terminal step's — the existing behaviour is exactly right. Say so, so the next reader does not go looking for a truncation branch that isn't there.

In `backend/app/agent.py`, replace lines 9-15 (the "Phase scope" block, ending with the closing `"""`) with:

```python
Phase scope (future concerns, not implemented here):
- No catching of handler *exceptions* as model observations (Phase 3). An
  unknown tool *name* is handled here, not deferred: the dispatch loop
  checks membership before dispatching and feeds back a ``tool_result``
  with ``is_error`` set, so one hallucinated name cannot fail the turn.
- No `/chat` SSE endpoint or conversation-store wiring (#4).

Truncation needs no branch here. A step cut off by the token ceiling
arrives from the provider as a step that requested no tools — the provider
drops tool blocks it cannot trust (see ``app.provider``) — so the loop
appends the assistant message, breaks, and reports that step's
``stop_reason`` as the turn's. The caller decides what to do with it.
"""
```

- [ ] **Step 7: Run the full backend suite**

Run: `cd backend && .venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass

- [ ] **Step 8: Commit**

```bash
git add backend/app/provider.py backend/app/agent.py backend/tests/test_provider.py
git commit -m "fix(provider): drop tool blocks from a turn cut off mid-stream"
```

---

### Task 3: `done` carries the terminal stop reason

**Files:**
- Modify: `backend/app/sse.py:69-72`
- Modify: `backend/app/chat.py:188`
- Modify: `docs/sse-contract.md:115-131`, `docs/sse-contract.md:154`
- Test: `backend/tests/test_sse.py:41-45`, `backend/tests/test_sse.py:118-125`, `backend/tests/test_chat.py:137`, `backend/tests/test_chat.py:206`, `backend/tests/test_chat_endpoint.py:129`, `backend/tests/test_chat_endpoint.py:219`

**Interfaces:**
- Consumes: `TurnComplete.stop_reason` (unchanged) and `TRUNCATION_STOP_REASONS` from Task 2 — used by neither, but Task 2 must land first so the value reaching the wire is already the guarded one.
- Produces: `DoneEvent(stop_reason: str | None = None)`. Wire payload becomes `{"stop_reason": ...}`, always present, `null` when unknown.

- [ ] **Step 1: Write the failing SSE tests**

In `backend/tests/test_sse.py`, replace `test_done_event_exact_wire_string` (lines 41-45) with:

```python
def test_done_event_exact_wire_string():
    event = DoneEvent()
    result = format_sse(event)
    assert result == 'event: done\ndata: {"stop_reason":null}\n\n'


def test_done_event_carries_a_stop_reason():
    event = DoneEvent(stop_reason="max_tokens")
    result = format_sse(event)
    assert result == 'event: done\ndata: {"stop_reason":"max_tokens"}\n\n'
```

And replace `test_done_json_is_empty_object` (lines 118-125) with:

```python
def test_done_json_carries_stop_reason_and_nothing_else():
    """Done event must produce exactly one field: stop_reason."""
    event = DoneEvent()
    result = format_sse(event)
    data_line = result.split("\n")[1]
    json_part = data_line[len("data: "):]
    assert json_part == '{"stop_reason":null}', f"Unexpected payload: {json_part!r}"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd backend && .venv/Scripts/python.exe -m pytest tests/test_sse.py -v`
Expected: `test_done_event_exact_wire_string` and `test_done_json_carries_stop_reason_and_nothing_else` FAIL comparing `{}` against `{"stop_reason":null}`; `test_done_event_carries_a_stop_reason` FAILs with a pydantic error about an unexpected keyword argument

- [ ] **Step 3: Add the field**

In `backend/app/sse.py`, replace the `DoneEvent` class (lines 69-72) with:

```python
class DoneEvent(SSEEvent):
    """End-of-stream sentinel; the last event of a successful turn.

    ``stop_reason`` is the terminal step's stop reason, passed through from
    the provider verbatim and *not* interpreted here. The value set is
    open-ended — the Anthropic SDK's own enums disagree on it — so consumers
    must treat an unrecognised value as a normal completion. ``None`` means
    the turn produced no ``TurnComplete`` to read one from.
    """

    event: ClassVar[str] = "done"

    stop_reason: str | None = None
```

- [ ] **Step 4: Run the SSE tests to verify they pass**

Run: `cd backend && .venv/Scripts/python.exe -m pytest tests/test_sse.py -v`
Expected: all pass

- [ ] **Step 5: Write the failing chat test**

Append to `backend/tests/test_chat.py`:

```python
async def test_done_carries_the_terminal_stop_reason_and_the_turn_persists():
    """A truncated turn is a *completed* turn: it streams, it persists, and it
    closes with a done that names why it stopped."""
    provider = FakeProvider(
        [[TextDelta("The three main "), TurnComplete(stop_reason="max_tokens", input_tokens=1, output_tokens=1)]]
    )
    registry = ToolRegistry()
    store = InMemoryConversationStore()

    chunks = [
        c
        async for c in stream_chat(
            provider=provider, registry=registry, store=store, session_id="s1", message="list them", model="claude-sonnet-4-6"
        )
    ]

    events = _parse_sse(chunks)
    assert events == [
        ("token", {"text": "The three main "}),
        ("done", {"stop_reason": "max_tokens"}),
    ]

    history = await store.get_history("s1")
    assert history == [
        {"role": "user", "content": "list them"},
        {"role": "assistant", "content": [{"type": "text", "text": "The three main "}]},
    ]
```

- [ ] **Step 6: Run it to verify it fails**

Run: `cd backend && .venv/Scripts/python.exe -m pytest tests/test_chat.py::test_done_carries_the_terminal_stop_reason_and_the_turn_persists -v`
Expected: FAIL — the `done` payload is `{"stop_reason": None}`, not `{"stop_reason": "max_tokens"}`

- [ ] **Step 7: Pass the stop reason through**

In `backend/app/chat.py`, replace line 188:

```python
    yield format_sse(DoneEvent(stop_reason=final_turn.stop_reason if final_turn else None))
```

- [ ] **Step 8: Update the four exact-equality assertions**

All four fakes end their final step with `stop_reason="end_turn"`.

`backend/tests/test_chat.py:137` — replace:

```python
    assert events == [("token", {"text": "Hi"}), ("done", {"stop_reason": "end_turn"})]
```

`backend/tests/test_chat.py:206` — replace `("done", {}),` with:

```python
        ("done", {"stop_reason": "end_turn"}),
```

`backend/tests/test_chat_endpoint.py:129` — replace:

```python
    assert events == [("token", {"text": "Hi"}), ("done", {"stop_reason": "end_turn"})]
```

`backend/tests/test_chat_endpoint.py:219` — replace `("done", {}),` with:

```python
        ("done", {"stop_reason": "end_turn"}),
```

- [ ] **Step 9: Run the full backend suite**

Run: `cd backend && .venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass

- [ ] **Step 10: Update the contract**

In `docs/sse-contract.md`, replace the `done` section (lines 115-131, from the `### 4.` heading through the closing fence of the example) with:

````markdown
### 4. `done` — End-of-stream sentinel

The last event of a successful turn. The client should treat the stream as complete after
receiving this event.

| Field | Type | Description |
|-------|------|-------------|
| `stop_reason` | `string \| null` | Why the model stopped, passed through from the provider verbatim. `null` when unknown. |

**The `stop_reason` value set is open-ended.** It is whatever the upstream provider sent;
it is not validated, mapped, or reduced to a flag anywhere in the backend. Consumers
**MUST** treat an unrecognised value as a normal completion. Two values currently mean the
response was cut off rather than finished, and warrant surfacing to the user:

- `max_tokens` — the response hit the configured token ceiling
- `model_context_window_exceeded` — the conversation outgrew the model's context window

**Example** (exact bytes):

```
event: done
data: {"stop_reason":"end_turn"}

```
````

And replace the summary-table row at line 154:

```markdown
| `done`     | Successful end-of-turn (final)  | `stop_reason: string \| null`             |
```

- [ ] **Step 11: Commit**

```bash
git add backend/app/sse.py backend/app/chat.py backend/tests/test_sse.py backend/tests/test_chat.py backend/tests/test_chat_endpoint.py docs/sse-contract.md
git commit -m "feat(sse): carry the terminal stop reason on done"
```

---

### Task 4: The frontend reducer keeps `stopReason`

`stopReason` is a separate field, **not** a new `AssistantTurnStatus`. The status vocabulary is closed and describes how the stream ended — `done` genuinely arrived, so the turn is `complete`. `stop_reason` is open-ended data about what the model did. Folding an open set into a closed enum means every future stop reason pressures the enum. Keeping them orthogonal also makes "an unrecognised value renders as a normal complete turn" fall out for free.

**Files:**
- Modify: `frontend/src/lib/sse.ts:22`
- Modify: `frontend/src/lib/chat-state.ts:48-55`, `:148-165`, `:201-202`, `:206-223`, `:236-243`
- Test: `frontend/src/lib/chat-state.test.ts`, `frontend/src/lib/sse-parser.test.ts`

**Interfaces:**
- Consumes: the `done` wire payload from Task 3 — `{ stop_reason: string | null }`.
- Produces:
  - `AssistantTurn.stopReason: string | null`
  - `closeTurn(state, assistantTurnId, status, error = null, stopReason = null)` — module-private
  - `SSEEvent` `done` variant: `{ type: "done"; data: { stop_reason?: string | null } }`

- [ ] **Step 1: Write the failing reducer tests**

Append to `frontend/src/lib/chat-state.test.ts`, inside the existing `describe("done", …)` block (which begins at line 122):

```ts
  it("records a truncation stop reason on an otherwise complete turn", () => {
    const state = run(
      send,
      { type: "sse", assistantTurnId: "a1", event: { type: "token", data: { text: "The three " } } },
      {
        type: "sse",
        assistantTurnId: "a1",
        event: { type: "done", data: { stop_reason: "max_tokens" } },
      },
    );

    expect(assistant(state).status).toBe("complete");
    expect(assistant(state).stopReason).toBe("max_tokens");
  });

  it("treats a done with no stop_reason field as a null stop reason", () => {
    const state = run(send, {
      type: "sse",
      assistantTurnId: "a1",
      event: { type: "done", data: {} },
    });

    expect(assistant(state).status).toBe("complete");
    expect(assistant(state).stopReason).toBeNull();
  });

  it("closes the turn normally on a stop reason it does not recognise", () => {
    const state = run(send, {
      type: "sse",
      assistantTurnId: "a1",
      event: { type: "done", data: { stop_reason: "banana" } },
    });

    expect(assistant(state).status).toBe("complete");
    expect(assistant(state).stopReason).toBe("banana");
    expect(state.status).toBe("idle");
  });

  it("clears the stop reason when the turn is retried", () => {
    const state = run(
      send,
      {
        type: "sse",
        assistantTurnId: "a1",
        event: { type: "done", data: { stop_reason: "max_tokens" } },
      },
      { type: "retry", assistantTurnId: "a1" },
    );

    expect(assistant(state).stopReason).toBeNull();
    expect(assistant(state).status).toBe("awaiting");
  });
```

Then update the exact-equality assertion in `describe("send", …)` at `frontend/src/lib/chat-state.test.ts:33-43` — the `toEqual` there enumerates every field of a fresh turn:

```ts
    expect(state.turns).toEqual([
      { kind: "user", id: "u1", text: "How do I add dependency injection?" },
      {
        kind: "assistant",
        id: "a1",
        text: "",
        toolCalls: [],
        status: "awaiting",
        error: null,
        stopReason: null,
      },
    ]);
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd frontend && npm test -- src/lib/chat-state.test.ts`
Expected: the four new tests FAIL (`stopReason` is `undefined`, not `null`/`"max_tokens"`), and the `send` test FAILs on the missing `stopReason` key

- [ ] **Step 3: Widen the wire type**

In `frontend/src/lib/sse.ts`, replace line 22:

```ts
  | { type: "done"; data: { stop_reason?: string | null } };
```

Optional, so an old-style `{}` payload still type-checks. No parser change is needed: `sse-parser.ts` passes unknown fields through untouched.

- [ ] **Step 4: Add the field to the reducer**

In `frontend/src/lib/chat-state.ts`, replace the `AssistantTurn` type (lines 48-55):

```ts
export type AssistantTurn = {
  kind: "assistant";
  id: string;
  text: string;
  toolCalls: ToolCall[];
  status: AssistantTurnStatus;
  error: TurnError | null;
  /**
   * Why the model stopped, verbatim from the provider — see the `done` event
   * in docs/sse-contract.md. Deliberately *not* folded into
   * `AssistantTurnStatus`: that vocabulary is closed and describes how the
   * stream ended, whereas this value set is open-ended and describes what the
   * model did. Keeping them orthogonal means a stop reason the UI has never
   * heard of renders as an ordinary complete turn.
   */
  stopReason: string | null;
};
```

Replace `closeTurn` (lines 148-165):

```ts
function closeTurn(
  state: ChatState,
  assistantTurnId: string,
  status: Extract<
    AssistantTurnStatus,
    "complete" | "failed" | "interrupted" | "stopped"
  >,
  error: TurnError | null = null,
  stopReason: string | null = null,
): ChatState {
  const next = updateOpenTurn(state, assistantTurnId, (turn) => ({
    ...turn,
    status,
    toolCalls: settleToolCalls(turn.toolCalls),
    error,
    stopReason,
  }));
  if (next === state) return state;
  return { ...next, status: "idle" };
}
```

Replace the `done` case (lines 201-202):

```ts
    case "done":
      // `?? null` collapses a missing field and an explicit null to the same
      // thing, so an old-style `{}` payload is indistinguishable from a turn
      // whose stop reason was unknown.
      return closeTurn(state, assistantTurnId, "complete", null, event.data.stop_reason ?? null);
```

In the `send` case, add `stopReason: null` to the new assistant turn (after `error: null,` at line 220):

```ts
          {
            kind: "assistant",
            id: action.assistantTurnId,
            text: "",
            toolCalls: [],
            status: "awaiting",
            error: null,
            stopReason: null,
          },
```

And in the `retry` case, reset it (line 239):

```ts
          ? {
              ...turn,
              text: "",
              toolCalls: [],
              status: "awaiting" as const,
              error: null,
              stopReason: null,
            }
```

- [ ] **Step 5: Run the reducer tests**

Run: `cd frontend && npm test -- src/lib/chat-state.test.ts`
Expected: all pass

- [ ] **Step 6: Write and run the parser test**

Append to `frontend/src/lib/sse-parser.test.ts`, inside the `describe("SSEFrameParser", …)` block:

```ts
  it("passes stop_reason through on a done frame", () => {
    const parser = new SSEFrameParser();

    const events = parser.push('event: done\ndata: {"stop_reason":"max_tokens"}\n\n');

    expect(events).toEqual([{ type: "done", data: { stop_reason: "max_tokens" } }]);
  });
```

Run: `cd frontend && npm test -- src/lib/sse-parser.test.ts`
Expected: all pass. This one may pass on the first run — the parser already forwards unknown fields, so it is a characterization test pinning behaviour the contract now depends on, not a driver of new code.

- [ ] **Step 7: Typecheck**

Run: `cd frontend && npm run typecheck`
Expected: no errors. If `turns.test.tsx` fails here on a missing `stopReason` in its `turnWith` helper, leave it — Task 5 fixes it.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/lib/sse.ts frontend/src/lib/chat-state.ts frontend/src/lib/chat-state.test.ts frontend/src/lib/sse-parser.test.ts
git commit -m "feat(chat-state): keep the terminal stop reason on the turn"
```

---

### Task 5: The truncation notice

**Files:**
- Modify: `frontend/src/components/turn-error.tsx` (append `isTruncated` + `TruncatedNotice`)
- Modify: `frontend/src/components/turns.tsx:7`, `:78`
- Test: `frontend/src/components/turns.test.tsx:11-21` (helper), plus a new `describe`

**Interfaces:**
- Consumes: `AssistantTurn.stopReason` from Task 4.
- Produces:
  - `isTruncated(stopReason: string | null): boolean`
  - `TruncatedNotice()` — takes no props

- [ ] **Step 1: Give the test helper the new field**

In `frontend/src/components/turns.test.tsx`, replace `turnWith` (lines 11-21):

```tsx
function turnWith(overrides: Partial<AssistantTurn> = {}): AssistantTurn {
  return {
    kind: "assistant",
    id: "a1",
    text: "",
    toolCalls: [],
    status: "awaiting",
    error: null,
    stopReason: null,
    ...overrides,
  };
}
```

- [ ] **Step 2: Write the failing tests**

Append to `frontend/src/components/turns.test.tsx`:

```tsx
describe("truncated turns", () => {
  it("says the answer was cut off at the length limit", () => {
    renderTurn(turnWith({ status: "complete", text: "The three main ", stopReason: "max_tokens" }));

    expect(screen.getByText(/cut off at the length limit/)).toBeInTheDocument();
  });

  it("shows the same notice when the context window was exceeded", () => {
    renderTurn(
      turnWith({
        status: "complete",
        text: "The three main ",
        stopReason: "model_context_window_exceeded",
      }),
    );

    expect(screen.getByText(/cut off at the length limit/)).toBeInTheDocument();
  });

  it("says the truncated answer was saved, and offers no way to act on it", () => {
    renderTurn(turnWith({ status: "complete", text: "The three main ", stopReason: "max_tokens" }));

    expect(screen.getByText(/saved to the conversation/)).toBeInTheDocument();
    expect(screen.queryByText(/not saved to the conversation/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("does not dim a truncated answer — the text is the real answer", () => {
    renderTurn(turnWith({ status: "complete", text: "The three main ", stopReason: "max_tokens" }));

    expect(screen.getByText("The three main ").closest(".opacity-65")).toBeNull();
  });

  it("shows nothing for a turn that ended normally", () => {
    renderTurn(turnWith({ status: "complete", text: "Done.", stopReason: "end_turn" }));

    expect(screen.queryByText(/cut off at the length limit/)).not.toBeInTheDocument();
  });

  it("shows nothing when there is no stop reason", () => {
    renderTurn(turnWith({ status: "complete", text: "Done.", stopReason: null }));

    expect(screen.queryByText(/cut off at the length limit/)).not.toBeInTheDocument();
  });

  it("shows nothing for a stop reason it does not recognise", () => {
    renderTurn(turnWith({ status: "complete", text: "Done.", stopReason: "banana" }));

    expect(screen.queryByText(/cut off at the length limit/)).not.toBeInTheDocument();
  });

  it("shows nothing on a turn still streaming toward that stop reason", () => {
    renderTurn(turnWith({ status: "streaming", text: "The three ", stopReason: null }));

    expect(screen.queryByText(/cut off at the length limit/)).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 3: Run them to verify they fail**

Run: `cd frontend && npm test -- src/components/turns.test.tsx`
Expected: the first four FAIL (`Unable to find an element with the text: /cut off at the length limit/`); the last four pass already, since nothing renders the notice yet — they are the guards that keep the coming implementation from over-matching

- [ ] **Step 4: Add the predicate and the notice**

Append to `frontend/src/components/turn-error.tsx`:

```tsx
/**
 * The two stop reasons that mean the response was cut off rather than
 * finished. Mirrors `TRUNCATION_STOP_REASONS` in `backend/app/provider.py`.
 *
 * The wire value set is open-ended (see the `done` event in
 * docs/sse-contract.md), so this is a positive match on the two values that
 * mean truncation, never a negative match on "not end_turn". A stop reason a
 * future model introduces renders as an ordinary complete turn until someone
 * decides otherwise.
 */
const TRUNCATION_STOP_REASONS = new Set(["max_tokens", "model_context_window_exceeded"]);

export function isTruncated(stopReason: string | null): boolean {
  return stopReason !== null && TRUNCATION_STOP_REASONS.has(stopReason);
}

/**
 * The model ran out of room mid-answer. Shaped like `StoppedNotice` — this is
 * not a failure and gets no colour, no alert role, and no buttons.
 *
 * The wording inverts the family's usual message because this is the one
 * notice whose turn *is* saved. Retry would re-send the original prompt and
 * leave the model seeing the same question twice; a canned "Continue" button
 * would put a user turn in the transcript that the user never typed. A user
 * who wants more types "continue", which works precisely because the
 * truncated turn is already in history.
 */
export function TruncatedNotice() {
  return (
    <div className="flex items-center gap-2.5 font-mono text-[11px] text-faint">
      <span aria-hidden className="size-2 shrink-0 rounded-full bg-faint" />
      response cut off at the length limit · saved to the conversation
    </div>
  );
}
```

- [ ] **Step 5: Render it**

In `frontend/src/components/turns.tsx`, replace the import on line 7:

```tsx
import { InterruptedNotice, isTruncated, StoppedNotice, TruncatedNotice, TurnError } from "./turn-error";
```

And add the notice after the `stopped` line (line 78):

```tsx
      {turn.status === "stopped" && <StoppedNotice />}

      {turn.status === "complete" && isTruncated(turn.stopReason) && <TruncatedNotice />}
```

Guarded on both: the status check keeps the notice off a turn that is still streaming, and `isTruncated` keeps it off a turn that finished normally.

Two deliberate non-changes in this file. `isIncomplete` (line 36) stays `interrupted || stopped` — those dim to `opacity-65` because their text may be discarded, whereas truncated text is saved and *is* the real answer; dimming it would read as "this doesn't count". And no `UnsavedNotice`, which would be a lie here.

- [ ] **Step 6: Run the component tests**

Run: `cd frontend && npm test -- src/components/turns.test.tsx`
Expected: all pass

- [ ] **Step 7: Run the whole frontend suite plus typecheck and lint**

Run: `cd frontend && npm test && npm run typecheck && npm run lint`
Expected: all tests pass, no type errors, no lint errors

- [ ] **Step 8: Run the whole backend suite once more**

Run: `cd backend && .venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass

- [ ] **Step 9: Commit**

```bash
git add frontend/src/components/turn-error.tsx frontend/src/components/turns.tsx frontend/src/components/turns.test.tsx
git commit -m "feat(ui): tell the user when an answer was cut off at the length limit"
```

---

## Acceptance criteria → task map

| Criterion | Task |
| --- | --- |
| `done` gains an additive field carrying the terminal stop reason verbatim | 3 |
| `docs/sse-contract.md` documents it and notes the value set is open-ended | 3 |
| `max_tokens` shows a truncation indicator; `end_turn` renders as today | 5 |
| `model_context_window_exceeded` shows the same indicator | 5 |
| An unrecognised value renders as a normal complete turn | 4, 5 |
| Old-style `done` with no field still parses and closes the turn complete | 4 |
| `max_tokens` configurable, defaulting to 2048; `.env.example` documents it | 1 |
| Persistence unchanged — a truncated turn still persists | 3 (asserted in the new `test_chat.py` test) |
| Truncated mid-`tool_use` with partial JSON no longer errors; streamed text survives; turn closes with `done` | 2, 3 |
| Truncated before any `input_json_delta` does not dispatch with empty input | 2 |
| Tests at the existing seams, plus the `test_provider.py` mid-`input_json_delta` case | 1–5 |

**Amended from issue #27:** the issue's preferred Option 1 — a sentinel `ToolUseRequested` plus an `INVALID_JSON` `tool_result` round trip — is not implemented. It contradicts the issue's own criterion that the turn closes with a `done` carrying the truncation stop reason: recovery costs another provider round trip, and `run_turn` reports the *last* step's `stop_reason`, so a repaired turn would end `end_turn` and the user would see no truncation signal at all. See *A truncated turn ends* in the design doc.

## Notes for the implementer

**Two test sites the design doc did not list.** It named four exact-equality assertions on the `done` payload; there are six. `backend/tests/test_sse.py` has two of its own (`test_done_event_exact_wire_string`, `test_done_json_is_empty_object`), and `frontend/src/lib/chat-state.test.ts:33` has a `toEqual` enumerating every field of a fresh assistant turn. All are handled in Tasks 3 and 4 respectively.

**If `content_block_stop` never arrives.** A tool block truncated before its closing event stays in `tool_blocks` and is never moved to `pending_tool_uses`, so it is dropped either way. That is the safe outcome and needs no extra handling — worth knowing when reading the loop.
