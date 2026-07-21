# Amendment: "persists nothing" criterion — issue #24, not #23

## The criterion is in #24

The review filed this against **#23**, but the wording quoted —
*"Stopping while a tool call is outstanding persists nothing"* — does not appear
in #23. It is the third acceptance criterion of **#24** ("Persist a stopped
turn's partial answer"), which is open. Issue #23 is closed, and its equivalent
criterion is already accurate:

> Persistence refuses to store an assistant message that ends in a tool call
> with no matching tool result

So the edit below belongs on **#24**. It has not been applied — apply it there,
or say the word and I will.

## Replace this criterion in #24

```
- [ ] Stopping while a tool call is outstanding persists nothing, leaving history a valid replay sequence
```

## With this

```
- [ ] Stopping while a tool call is outstanding persists any streamed assistant text and drops the unfinished `tool_use` block — a dangling `tool_use` must never be persisted, per the API's `tool_use`/`tool_result` pairing requirement — leaving history a valid replay sequence
```

## Why the code is right and the criterion is wrong

**The API requires the pairing.** Anthropic's docs are explicit that
`tool_result` blocks "must immediately follow their corresponding tool use
blocks in the message history," and that a mismatch produces an error of the
form *"tool_use ids were found without tool_result blocks immediately after."*
A dangling `tool_use` persisted to history therefore makes the conversation
unresumable — the next request on that session fails with a
`400 invalid_request_error`. Dropping the call is not a preference; it is the
only way to keep stored history replayable.
(Source: `docs/research/claude-tool-calling.md` §1.2, T1 first-party docs. The
400 status code specifically is an inference — the docs quote the error string
but not the status.)

**Keeping the text matches what the user saw.** Content the user watched stream
in should survive a refresh. Dropping the whole assistant message to get rid of
the call would discard an answer the user read and the model can no longer
recall — a strictly worse outcome that the API does not ask for. Only the call
is unreplayable; the text beside it is fine.

**This is the same rule #23 already chose.** `persistable_messages` strips
unanswered calls per-message and keeps surviving content, on every path. The
stopped path reuses it rather than reimplementing it, which #24 also requires
("The dangling-tool-call check from #23 is reused, not reimplemented").

Covered by `test_stopping_while_a_tool_runs_drops_the_call_and_keeps_the_text`.

## Possible follow-up (Phase 3)

Rather than dropping the call, persist it alongside a synthetic `tool_result`
marking it cancelled by the user. That satisfies the pairing rule *and* leaves
the model able to see that a tool was started and abandoned — better context on
resume. It needs an `is_error` / `content` convention the codebase does not have
yet (`run_turn` builds `tool_result` blocks with neither), so it is out of scope
here. Noted in the code at `backend/app/conversation.py`, in
`_is_unanswered_call`'s docstring.
