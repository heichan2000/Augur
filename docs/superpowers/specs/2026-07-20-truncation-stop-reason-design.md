# Surface `max_tokens` truncation instead of presenting a complete turn

Design for [issue #27](https://github.com/heichan2000/Augur/issues/27). Date: 2026-07-20.

## Problem

The provider collects `stop_reason` from the stream and `run_turn` carries it on
`TurnComplete` — and then nobody reads it. A turn cut off by the token ceiling streams
normally, persists normally, ends with a normal `done`, and renders as a finished answer.
The user gets no signal that the tail is missing, and a follow-up silently builds on a
truncated message.

Truncation mid-`tool_use` is worse than cosmetic. `provider.py:125` is an unguarded
`json.loads(raw) if raw else {}`, which gives two live failures:

| Truncated buffer | Behaviour today |
| --- | --- |
| Partial JSON (e.g. `{"tz":`) | `JSONDecodeError` propagates to `stream_chat`'s catch-all → SSE `error` / `internal`, no `done`, nothing persisted, streamed text discarded |
| Empty (truncated before any `input_json_delta`) | `raw` falsy → `parsed = {}` → a real tool call is dispatched with silently-empty input |

## Decisions

### Pass `stop_reason` through verbatim; do not derive a boolean

The issue argued this from the seven documented values. The installed SDK makes the case
harder still — the two enums already disagree:

- `anthropic/types/stop_reason.py` — six values, **no** `model_context_window_exceeded`
- `anthropic/types/beta/beta_stop_reason.py` — eight, including `compaction`, which the
  issue's list of seven does not mention

The value set drifted between the issue being written and this design. The wire field is
typed `str | None` and the frontend decides which values warrant a notice.

### A truncated turn ends; it does not attempt recovery

**This overrides the issue's stated preference.** Issue #27 preferred Option 1 — yield a
sentinel `ToolUseRequested`, have `run_turn` feed back an `INVALID_JSON` `is_error`
`tool_result`, and let the model repair its own call. That option contradicts the issue's
own acceptance criterion that *"the turn closes with a `done` carrying the truncation stop
reason"*: recovery costs another provider round trip, and `run_turn` reports the **last**
step's `stop_reason`, so a repaired turn ends `end_turn` and the user sees no truncation
signal at all.

Given the fork, we take the issue's Option 2 — drop the unusable block and end the turn.
The user is told the answer was cut off and decides what to do. This also avoids an
unresolved problem in Option 1: to pair an `INVALID_JSON` `tool_result` with anything, the
assistant message needs an API-legal `tool_use` block, and an unparseable input cannot
produce one.

If the retry-with-higher-`max_tokens` path is ever wanted, revisit this decision — it is
the only thing standing between here and Option 1.

### The guard lives in the provider, which buffers tool blocks

Event order is `content_block_stop` → `message_delta` (carrying `stop_reason`) →
`message_stop`. The provider therefore cannot tell, at block-close time, whether an empty
buffer means truncation or a genuinely zero-argument tool.

`run_turn` *could* decide instead — it collects `tool_uses` during the stream and
dispatches only after the loop, by which point `TurnComplete` has arrived. Rejected: by
then `stream_chat` has already emitted a `tool_use` SSE event for the dead call, so the UI
draws a tool row and `closeTurn` settles it to `"done"` — a tool that never ran, shown as
having succeeded.

Instead the provider stashes tool blocks and replays them only once `stop_reason` is
known. One place owns the rule, `run_turn` needs no change, and no phantom tool row
reaches the UI. The cost is that within a step all `ToolUseRequested` events land after
all `TextDelta` events; in practice the API already emits text blocks before `tool_use`
blocks, so observable ordering is unchanged.

### The truncation notice is informational only

No Retry button. The truncated turn **is** persisted, so Retry would re-send the original
prompt and leave the model seeing the same question twice. No canned "Continue" button
either — it would add a send path and put a user turn in the transcript that the user
never typed. A user who wants more types "continue", which works precisely because the
truncated turn is in history.

## Backend

**`config.py`** — `Settings` gains `anthropic_max_tokens: int = 2048`. Behaviour is
unchanged out of the box. `.env.example` documents it.

**`provider.py`**

1. `AnthropicProvider.__init__` takes `max_tokens`, held the same way `model` already is.
   `stream_turn` drops its `max_tokens: int = 2048` parameter and reads `self._max_tokens`.
   `build_provider()` passes `settings.anthropic_max_tokens`.

   No call site passes `max_tokens` today — `run_turn` calls
   `stream_turn(messages=…, system=…, tools=…)` — so removing the parameter breaks
   nothing. The fake providers in `test_agent.py`, `test_chat.py` and
   `test_chat_endpoint.py` declare `max_tokens=2048` in their own signatures; they are
   independent stubs, not subclasses, and need no change.

2. A module constant:

   ```python
   TRUNCATION_STOP_REASONS = frozenset({"max_tokens", "model_context_window_exceeded"})
   ```

3. Tool blocks are stashed rather than yielded, and the parse is guarded:

   ```
   content_block_stop → try json.loads(raw) if raw else {}
                        JSONDecodeError → log warning, drop the block
                        success         → stash ToolUseRequested
   message_delta      → stop_reason now known
   after the stream   → if stop_reason in TRUNCATION_STOP_REASONS: drop all stashed
                        else: yield each stashed
                      → yield TurnComplete(stop_reason=…)
   ```

   The replay and the `TurnComplete` both sit *outside* the `try`/`except
   anthropic.APIError` block, where `yield TurnComplete(...)` already lives today — a
   stream that raised never reaches either.

   `TextDelta` still yields live inside the loop; streaming is unaffected. The module gains
   a `logging.getLogger("augur.provider")`. `tool_name` is already whitelisted in
   `observability.py` from #28, so the warning reuses that field.

**`agent.py`** — no code change. On truncation the provider yields no `ToolUseRequested`,
so `tool_uses` is empty, `run_turn` appends the assistant message and breaks, and
`final_stop_reason` is already the terminal step's. The module docstring gains a line
noting that a truncated step arrives here as a step that requested no tools.

**`sse.py`** — `DoneEvent` gains `stop_reason: str | None = None`.

**`chat.py`** — one line:
`DoneEvent(stop_reason=final_turn.stop_reason if final_turn else None)`.

**Persistence** — untouched. A truncated turn is `COMPLETED` and persists. Truncation
leaves no `tool_use` block in the assistant message at all, so `persistable_messages` has
nothing to strip. If no text was streamed either, the empty-content message is dropped and
the user's message is kept — existing `COMPLETED` behaviour.

**`docs/sse-contract.md`** — document `stop_reason` on `done`, and state that the value set
is open-ended: consumers must treat an unrecognised value as a normal completion.

### Wire consequence

`done` now always carries `stop_reason`, including `null`. Four tests assert the payload is
exactly `{}` — `test_chat.py:137`, `test_chat.py:206`, `test_chat_endpoint.py:129`,
`test_chat_endpoint.py:219` — and must be updated. This is a legal additive change under
the contract, but it is not invisible.

## Frontend

**`lib/sse.ts`** — the `done` variant becomes
`{ type: "done"; data: { stop_reason?: string | null } }`. Optional, so an old-style `{}`
payload still type-checks. No parser change: `sse-parser.ts` already passes unknown fields
through untouched.

**`lib/chat-state.ts`** — `AssistantTurn` gains `stopReason: string | null`.

`stopReason` is a separate field, **not** a new `AssistantTurnStatus`. The status
vocabulary is closed and describes how the stream ended — `done` genuinely arrived, so the
turn is `complete`. `stop_reason` is open-ended data about what the model did. Folding an
open set into a closed enum means every future stop reason pressures the enum, which is the
same mistake the passthrough decision avoids on the wire. Keeping them orthogonal also
makes "an unrecognised value renders as a normal complete turn" fall out for free.

- `closeTurn` takes a `stopReason` argument defaulting to `null`
- the `done` case passes `event.data.stop_reason ?? null`, collapsing a missing field and
  an explicit `null` to the same thing
- `send` and `retry` initialise and reset it to `null`

**`components/turn-error.tsx`** — a new `TruncatedNotice` in `StoppedNotice`'s shape: dot,
mono 11px, `text-faint`, no `role="alert"`, no buttons. The wording inverts the family's
usual message, because this is the one notice whose turn *is* saved:

```
● response cut off at the length limit · saved to the conversation
```

The predicate lives beside it:

```ts
const TRUNCATION_STOP_REASONS = new Set(["max_tokens", "model_context_window_exceeded"]);
export function isTruncated(stopReason: string | null): boolean
```

**`components/turns.tsx`** — one line, guarded on both:

```tsx
{turn.status === "complete" && isTruncated(turn.stopReason) && <TruncatedNotice />}
```

**Two deliberate non-changes.** `isIncomplete` stays `interrupted || stopped`: those dim to
`opacity-65` because their text may be discarded, whereas truncated text is saved and is
the real answer — dimming it would read as "this doesn't count". And no `UnsavedNotice`,
which would be false here.

## Testing

**`backend/tests/test_provider.py`** — scripted `FakeClient`, no network:

| Case | Asserts |
| --- | --- |
| Stops mid-`input_json_delta` (buffer `{"tz":`), `stop_reason: max_tokens` | no exception; no `ToolUseRequested`; `TurnComplete.stop_reason == "max_tokens"` |
| Empty tool buffer, `stop_reason: max_tokens` | no `ToolUseRequested` — not dispatched with empty input |
| Empty tool buffer, `stop_reason: tool_use` | `ToolUseRequested(input={})` still yielded — a zero-argument tool keeps working |
| Unparseable buffer, `stop_reason: tool_use` | block dropped, turn completes normally |
| Normal tool use | yielded as today — regression guard on the stash-and-replay path |
| `max_tokens` from the constructor | appears in the client kwargs |

The third row is the one that earns its keep: it is the case the new guard could most
easily break, and it is why the guard keys on `stop_reason` rather than on the buffer being
empty. The existing `test_tool_use_with_no_input_json_delta_yields_empty_dict` already pins
part of it.

**`test_config.py`** — default is 2048; the env var overrides it.

**`test_sse.py`** — `DoneEvent()` serialises with `stop_reason` null;
`DoneEvent(stop_reason="max_tokens")` round-trips.

**`test_chat.py` / `test_chat_endpoint.py`** — `done` carries the terminal stop reason; a
truncated turn still persists; plus the four exact-equality updates noted above.

**`test_agent.py`** — no new tests. `run_turn` is unchanged, and a truncated step is
indistinguishable to it from a step that requested no tools, which is already covered.

**`frontend/src/lib/chat-state.test.ts`** — `done` with `max_tokens` sets `stopReason` and
status `complete`; `done` with no `stop_reason` field yields `stopReason: null` and closes
the turn complete; an unrecognised value (`"banana"`) closes the turn normally without
crashing; `retry` clears `stopReason`.

**`frontend/src/lib/sse-parser.test.ts`** — a `done` frame carrying `stop_reason` parses
with the field intact.

**`frontend/src/components/turns.test.tsx`** — the notice renders for `max_tokens` and
`model_context_window_exceeded`; does not render for `end_turn`, `null`, or an unrecognised
value; a truncated turn is not dimmed and shows no `UnsavedNotice`.

## Error handling

No new error path. The guard converts a crash into a normal terminal state: the
`JSONDecodeError` that previously escaped to `internal` no longer occurs, and nothing new
reaches `stream_chat`'s handlers. The dropped block is logged at `warning` rather than
swallowed silently.

## Acceptance criteria

Mapped from issue #27. All are covered by this design, with one amendment.

- [x] `done` gains an additive field carrying the terminal stop reason verbatim — `stop_reason: str | None`
- [x] `docs/sse-contract.md` documents it and notes the value set is open-ended
- [x] `stop_reason == "max_tokens"` shows a truncation indicator; `end_turn` renders as today
- [x] `model_context_window_exceeded` shows the same indicator
- [x] An unrecognised value renders as a normal complete turn
- [x] Old-style `done` events with no field still parse and close the turn complete
- [x] `max_tokens` is configurable via settings, defaulting to 2048; `.env.example` documents it
- [x] Persistence unchanged — a truncated turn still persists
- [x] A stream truncated mid-`tool_use` with a partial JSON buffer no longer produces an `internal` error; streamed text survives and the turn closes with a `done` carrying the truncation stop reason
- [x] A stream truncated before any `input_json_delta` does not dispatch a tool call with silently-empty input
- [x] Tests at the existing seams, plus the `test_provider.py` mid-`input_json_delta` case

**Amended:** the issue's preferred Option 1 (sentinel `ToolUseRequested` +
`INVALID_JSON` `tool_result` round trip) is not implemented. See *A truncated turn ends*
above for why it is incompatible with the criterion it shares the issue with.
