# Claude tool calling: the API contract, error-as-observation, and what it means for Augur

Research date: 2026-07-20. Scope: the Anthropic Messages API tool-use contract, the SDK's
Tool Runner error path, a short RAG section, and how all three map onto Augur issues #28, #27,
and #23.

---

## 0. Source-trust framing (read this first)

The premise "there's open source on GitHub that revealed Claude's code" is **not accurate as
stated**. Here is what actually exists, in descending order of trust. Every citation in this
document is tagged with its tier.

| Tier | What it is | Trust | Examples |
| --- | --- | --- | --- |
| **T1 — First-party docs** | Anthropic's published API documentation | Authoritative for the wire contract | [platform.claude.com/docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview) |
| **T1 — First-party SDK source** | Genuinely open source, Apache-2.0, Anthropic-published | Authoritative for SDK behaviour | [anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python), [anthropic-sdk-typescript](https://github.com/anthropics/anthropic-sdk-typescript) |
| **T1 — MCP spec** | Open source and first-party | Authoritative for MCP | [modelcontextprotocol.io](https://modelcontextprotocol.io), [github.com/modelcontextprotocol](https://github.com/modelcontextprotocol) |
| **T3 — Derived artifacts** | Community deobfuscations of the minified `claude-code` npm bundle, scraped system prompts, extracted tool descriptions | Low. Not Anthropic-published. Version-drifts silently. | (deliberately not cited below) |
| **Not published anywhere** | The *model's* retrieval behaviour and tool-selection policy — server-side, closed | No source can speak to this | — |

**Nothing in this document rests on a T3 source.** Where a claim could only come from a T3
source, it is not made. The `claude-code` npm package ships as minified JS; community repos that
publish deobfuscated dumps are reverse-engineered artifacts, not source releases, and a repo
claiming to reveal the model's RAG or tool-selection internals is claiming something Anthropic
has not published.

Sections below separate **what the API guarantees** (T1-cited) from **what I inferred** (labelled
inline).

---

## 1. Tool calling per the Anthropic API

### 1.1 The content-block contract

A client-tool round trip is three blocks across three messages
([T1 docs — Handle tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)):

- **Assistant** returns `stop_reason: "tool_use"` and one or more `tool_use` blocks carrying
  `id`, `name`, `input`.
- **User** replies with `tool_result` blocks carrying `tool_use_id`, optional `content`
  (string, or a list of `text` / `image` / `document` / `search_result` blocks), and optional
  `is_error`.

There is no `tool` role. Quoting the docs
([T1](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)):

> Unlike APIs that separate tool use or use special roles like `tool` or `function`, the Claude
> API integrates tools directly into the `user` and `assistant` message structure.

### 1.2 Every `tool_use` must be answered — and the adjacency rule is stricter than "eventually"

Verbatim formatting requirements
([T1](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)):

> - Tool result blocks must immediately follow their corresponding tool use blocks in the message
>   history. You cannot include any messages between the assistant's tool use message and the
>   user's tool result message.
> - In the user message containing tool results, the tool_result blocks must come FIRST in the
>   content array. Any text must come AFTER all tool results.

And the failure mode, quoted:

> If you receive an error like "tool_use ids were found without tool_result blocks immediately
> after", check that your tool results are formatted correctly.

**What the API guarantees:** an unanswered `tool_use` in the message array is a **400 on the next
request**, not a degraded response. This is the rule #23 exists to protect, and Augur's guard
implements exactly it (see §4.3).

**What I inferred:** the error text is a `400 invalid_request_error`; the docs quote the message
string but not the status code. The 400 classification is consistent with the sibling examples on
the same page (text-before-`tool_result` is documented as "this will cause a 400 error").

### 1.3 `stop_reason` values

From [T1 — Handling stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons):

| Value | Meaning | Handling |
| --- | --- | --- |
| `end_turn` | Finished naturally | Use as-is |
| `max_tokens` | Hit the `max_tokens` ceiling | Raise the limit or continue |
| `stop_sequence` | Hit a custom stop sequence | Read the `stop_sequence` field |
| `tool_use` | Claude is calling a tool | Run it, return `tool_result` |
| `pause_turn` | Server-tool loop hit its iteration limit | Send the response back to continue |
| `refusal` | Claude declined | Read `stop_details`; retry on a fallback model |
| `model_context_window_exceeded` | Context window filled | Treat the response as truncated |

Note `model_context_window_exceeded` — a seventh value the issue text does not list, and one
Augur's `TurnComplete.stop_reason` will pass through untouched.

On truncation mid-tool-call, the docs are explicit
([T1](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)):

> If Claude's response is cut off because it hit the `max_tokens` limit, and the truncated
> response contains an incomplete tool use block, you'll need to retry the request with a higher
> `max_tokens` value to get the full tool use.

The recommended detection is: on `stop_reason == "max_tokens"`, check whether the **last content
block** is a `tool_use`; an incomplete tool use cannot be executed.

### 1.4 `is_error` on `tool_result`

`is_error: true` is the sanctioned channel for handler failures — the turn continues, the model
sees the failure as an observation
([T1](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)):

> If the tool itself throws an error during execution (for example, a network error when fetching
> weather data), you can return the error message in the `content` along with `"is_error": true`
> […] Claude will then incorporate this error into its response to the user.

Plus a directly actionable prompt-engineering note:

> Write instructive error messages. Instead of generic errors like `"failed"`, include what went
> wrong and what Claude should try next, e.g., `"Rate limit exceeded. Retry after 60 seconds."`

### 1.5 Parallel tool calls

From [T1 — Parallel tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use):

> Whichever strategy you use, return one `tool_result` for each `tool_use` block, all together in
> the next user message. Match each result to its call with `tool_use_id`, and put every
> `tool_result` block before any text content in that message.

And, critically for the #28 decision:

> If you choose not to run a particular call (for example, because you ran the batch sequentially
> and an earlier call failed), still return a `tool_result` for it with `is_error: true` and a
> brief explanation.

```json
{
  "type": "tool_result",
  "tool_use_id": "toolu_02",
  "is_error": true,
  "content": "Not executed: the preceding write_file call failed."
}
```

The same page notes that mis-formatting results in history "teaches" Claude to stop making
parallel calls — i.e. dropping a result is not merely a validation error, it degrades future
behaviour. Parallelism is on by default; `tool_choice: {"type": "auto", "disable_parallel_tool_use": true}`
turns it off.

### 1.6 Unknown and malformed tool names

The docs address **invalid tool use** (missing/wrong parameters) directly
([T1](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)):

> However, you can also continue the conversation forward with a `tool_result` that indicates the
> error, and Claude will try to use the tool again with the missing information filled in […]
> If a tool request is invalid or missing parameters, Claude will retry 2-3 times with corrections
> before apologizing to the user.

The docs do **not** contain a paragraph specifically about a *hallucinated tool name*. The
first-party answer for that case comes from the SDK source instead — see §2, which is
unambiguous.

---

## 2. Error-as-observation: what Anthropic's own SDK actually does

This is the decisive evidence for #28, and it comes from code, not a blog post.

### 2.1 Documented behaviour

[T1 docs — Tool Runner](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-runner),
verbatim:

> When a tool throws an exception, the tool runner catches it and returns the error to Claude as a
> tool result with `is_error: true`. The tool result carries the exception's message (in Python,
> its type and message), not the full stack trace.

> By default, tool errors are passed back to Claude, which can then respond appropriately.
> However, you might want to detect errors and handle them differently, for example, to stop
> execution early or implement custom error handling.

The "stop the loop" behaviour Augur has today is documented as the *opt-in override*, reachable
only by explicitly inspecting `generate_tool_call_response()` and raising. It is not the default.

### 2.2 The actual code path

From [T1 SDK source — `anthropic-sdk-python`, `src/anthropic/lib/tools/_beta_runner.py`](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/lib/tools/_beta_runner.py),
inside `BaseSyncToolRunner._generate_tool_call_response()` (the async variant
`BaseAsyncToolRunner` implements identical logic):

**Unknown tool name — warn, then answer it:**

```python
if tool is None:
    warnings.warn(
        f"Tool '{tool_use.name}' not found in tool runner. "
        f"Available tools: {list(self._tools_by_name.keys())}. "
        ...
    )
    results.append(
        {
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": f"Error: Tool '{tool_use.name}' not found",
            "is_error": True,
        }
    )
```

**Handler exception — log, then answer it:**

```python
except ToolError as exc:
    results.append({"type": "tool_result", "tool_use_id": tool_use.id,
                    "content": tool_error_content(exc), "is_error": True})
except Exception as exc:
    log.exception(f"Error occurred while calling tool: {tool.name}", exc_info=exc)
    results.append({"type": "tool_result", "tool_use_id": tool_use.id,
                    "content": tool_error_content(exc), "is_error": True})
```

Three things to note:

1. **Unknown tool name is handled identically to a handler exception** — a `tool_result` with
   `is_error: true`. It is not a special case; it is the same recovery path.
2. **The catch is bare `except Exception`.** Anthropic's own reference implementation does not
   enumerate recoverable exception types; anything short of a `BaseException` becomes an
   observation.
3. **The `warnings.warn` / `log.exception` calls are the operator signal**, kept entirely separate
   from the model-facing result. Observability and recovery are decoupled — the developer learns
   about it, the turn survives.

The same default is documented across the other SDKs on the Tool Runner page
([T1](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-runner)) — Go: "converts
an error returned from your handler into a tool result with `is_error: true` internally"; Java:
"catches any exception thrown from a tool's `get()` method and converts it into a tool result with
`is_error: true` automatically"; PHP and C# likewise. Error-as-observation is the cross-SDK
default, not a Python quirk.

**Answer to the research question:** yes, unambiguously. Anthropic's own agent loop feeds handler
exceptions *and* unknown tool names back as `tool_result` / `is_error: true` rather than failing
the turn.

---

## 3. RAG — what Anthropic actually documents

Short by design; secondary to the above.

- **Tool-based search is the documented primary path.** Retrieval is modelled as a tool the
  model calls, not as stuffed context. Server-side `web_search` returns cited results directly
  ([T1](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)); for custom
  corpora you define your own search tool and return results in a `tool_result`.
- **`search_result` content blocks** are the first-party mechanism for custom-corpus citations.
  Quoting [T1 — Search results](https://platform.claude.com/docs/en/build-with-claude/search-results):
  "Search result content blocks enable natural citations with proper source attribution, bringing
  web search-quality citations to your custom applications. This feature is particularly powerful
  for RAG (Retrieval-Augmented Generation) applications where you need Claude to cite sources
  accurately." They can appear either **inside a `tool_result`** (dynamic RAG) or **as top-level
  user content** (pre-fetched data) — "Flexible integration: Use in tool returns for dynamic RAG
  or as top-level content for pre-fetched data."
- **Citations API** — `citations: {enabled: true}` on `document` blocks splits the response into
  cited `text` blocks carrying `cited_text` and a location
  ([T1 — Citations](https://platform.claude.com/docs/en/build-with-claude/citations)).
- **Contextual Retrieval** is an Anthropic *engineering technique*, not an API feature — a
  first-party post ([T1-blog](https://www.anthropic.com/engineering/contextual-retrieval),
  2024-09-19) describing prepending chunk-specific explanatory context before embedding, combining
  contextual embeddings with contextual BM25. Reported retrieval-failure-rate reduction: 35%
  (embeddings alone), 49% (embeddings + BM25), 67% (plus reranking). It is a preprocessing recipe
  you implement; there is no API parameter for it.
- **Not published:** how the model itself decides *when* to retrieve or *which* tool to select.
  That is server-side. No source — first-party or otherwise — documents it, and any repo claiming
  to should be treated as speculation.

---

## 4. Mapping onto Augur

Files read: `backend/app/tools.py`, `backend/app/agent.py`, `backend/app/conversation.py`,
`backend/app/provider.py`, `backend/app/chat.py`, and `backend/tests/`.

### 4.1 #28 — unknown tool name fails the whole turn as `internal`

**Current path.** `ToolRegistry.dispatch` (`backend/app/tools.py:60-66`) does a bare dict lookup:

```python
tool = self._tools[name]
return await tool.handler(tool_input)
```

`run_turn` (`backend/app/agent.py:117`) awaits it unguarded inside the tool-dispatch loop.
`stream_chat`'s catch-all (`backend/app/chat.py:151-154`) converts the resulting `KeyError` into
`ErrorEvent(type="internal")`, returns without a `done`, and persists nothing — discarding text
the user already read.

**What the contract says.** Anthropic's own reference loop treats "tool not found" as a
`tool_result` with `is_error: true` and a message naming the missing tool and listing the
available ones (§2.2). The API-level requirement points the same way: the assistant message
carrying that `tool_use` is *already in Augur's `messages` list*. Under §1.2 the only two valid
continuations are (a) answer it with a `tool_result`, or (b) never persist it. Augur currently does
(b) — via `persistable_messages` — which is *valid*, but it is the strictly worse of the two: it
throws away a recoverable turn.

**Recommendation: option 2 — minimal guard now.** Reasoning:

1. **The contract already specifies the shape.** This is not a design question awaiting Phase 3;
   §1.5 and §2.2 give the exact block to emit. Phase 3 generalises the mechanism, it does not
   change this case's answer.
2. **The parallel-tool rule makes it load-bearing sooner than the issue assumes.** The issue notes
   the blast radius is small because only one tool is registered. But parallelism is on by default
   (§1.5), so a single hallucinated name in a batch of otherwise-valid calls kills the whole batch
   today — and the docs explicitly say to return `is_error: true` for calls you didn't run rather
   than drop them. Blast radius grows with the second registered tool, not with Phase 3.
3. **It is a smaller change than the Phase-3 work.** A guard in `dispatch` (or, better, in
   `run_turn`'s dispatch loop) that catches `KeyError` and substitutes an error string is a few
   lines and requires no new event types or SSE contract changes.
4. **It removes an availability cliff.** Today a model hallucination is a 500-class outcome for
   the user. Under the guard it is a self-correcting round trip — the docs note Claude "will retry
   2-3 times with corrections" when handed an error result (§1.6).

**Suggested shape** — keep the observability, change only the recovery. Follow the SDK's split:
log the operator signal, return the model-facing observation. Make the message instructive per
§1.4 (name the missing tool, list the available ones — the SDK does exactly this). Where to put it
is a judgement call:

- In `run_turn`'s loop: keeps `ToolRegistry` a pure lookup, and puts the recovery next to the
  `tool_result` construction it feeds. Phase 3 will need a `try/except` there anyway for handler
  exceptions, so this is the seam that generalises.
- In `dispatch`: smaller diff, but conflates "does this tool exist" with "did it succeed", and
  Phase 3 would likely move it back out.

I'd put it in `run_turn`. One caveat either way: Augur's `tool_result` blocks
(`backend/app/agent.py:118-120`) have no `is_error` field at all — see §4.4.

### 4.2 #27 — `max_tokens` truncation presented as a complete turn

The issue's framing is correct and the contract backs it: `max_tokens` is a distinct terminal
state the API expects callers to branch on (§1.3), and Augur collects `stop_reason` into
`TurnComplete` (`backend/app/provider.py:133`, `backend/app/agent.py:87`) then never reads it.

Two additions from the research the issue does not currently cover:

**(a) There are seven stop reasons, not four.** `model_context_window_exceeded` (§1.3) is also a
truncation signal and is *also* silently rendered as a complete answer today. If the `done` event
gains a `stop_reason` passthrough field rather than a boolean `truncated`, both are covered for
free and future values need no backend change — which is the better fit for the additive-evolution
rule the issue cites.

**(b) Truncation mid-`tool_use` is the dangling-call shape, and Augur's handling of it is
currently unverified.** The docs say an incomplete `tool_use` block cannot be executed and the
request must be retried with a higher `max_tokens` (§1.3). Augur's provider buffers
`input_json_delta` into `tool_blocks[index]["buf"]` and only emits `ToolUseRequested` on
`content_block_stop`, where it calls `json.loads(raw)` (`backend/app/provider.py:120-130`).

*This is inferred, not verified* — I did not run a truncated stream against the real API — but
there appear to be two possible outcomes, both bad:

- If truncation means no `content_block_stop` arrives for the open tool block, the buffered call is
  silently dropped. `run_turn` sees `tool_uses == []`, breaks, and persists an assistant message
  containing only the partial text — a truncated turn stored as complete, with no signal.
- If `content_block_stop` does arrive with a partial JSON buffer, `json.loads` raises
  `JSONDecodeError`, which propagates through `run_turn` into the same `chat.py` catch-all as #28
  and produces a generic `internal` error, discarding the streamed text.

**Recommendation:** worth a test at the provider seam (`backend/tests/test_provider.py`) driving a
fake stream that stops mid-`input_json_delta`, to establish which branch is real before #27 is
implemented. Whichever it is, #27's `stop_reason` passthrough is the signal that makes it
diagnosable — and the second branch is the same failure family as #28, which strengthens the
case for handling dispatch-path exceptions as observations rather than turn-killers.

### 4.3 #23 (closed, current branch) — does the shipped guard match the API's validity rules?

**Verdict: it matches, and it does not over- or under-strip.** Details:

`persistable_messages` (`backend/app/conversation.py:86-115`) checks, for each message, only the
**immediately following** message's `tool_result` ids (`following = messages[index + 1]`, line 107).
That is precisely the API rule — "must immediately follow their corresponding tool use blocks […]
You cannot include any messages between" (§1.2). A guard that searched the whole remaining history
would be *too permissive* and would persist sequences the API rejects; this one does not.

Specific checks:

- **`tool_use` with no `id` counts as unanswered** (`_is_unanswered_call`, lines 55-63). Correct
  and defensive: the API matches on `tool_use_id`, so a call with no id is unreplayable by
  construction.
- **Partial strip preserves the streamed answer** (`_without_unanswered_calls`, lines 66-83). This
  is valid under the contract — the constraint is on `tool_use` blocks, not on `text` blocks in the
  same assistant message. Nothing is over-stripped.
- **No orphan-`tool_result` hazard.** The one way to create a `tool_result` with no preceding
  `tool_use` would be for an assistant message to be dropped entirely while its answering user
  message survives. That cannot happen: an assistant message is dropped only when *no* content
  survives, which means *none* of its calls were answered, which means the following message
  contains no matching `tool_result` blocks. Verified by construction, and
  `backend/tests/test_chat.py` covers the max-steps and stopped-turn cases (lines ~302, ~335, ~363,
  ~489).
- **Non-list content is handled** (line 80-81): a plain-string message passes through, empty
  content is dropped.

**Two things it does not do, neither of which is a bug in #23's scope:**

1. It does not enforce "`tool_result` blocks must come FIRST in the content array" (§1.2). Augur
   never mixes text into a tool-result message (`backend/app/agent.py:121` builds a
   `tool_results`-only user message), so this cannot currently be violated — but it is an
   unenforced invariant that a future path could break silently.
2. It runs only over the turn's *new* messages (`chat.py:119`, `chat.py:156`). Prior history is
   trusted. That is fine while the only writer is this guard, but a Phase-2 persistent store
   populated by anything else inherits no validation.

### 4.4 Divergences from the API contract that will bite later

Beyond the three issues:

- **Augur never emits `is_error` at all.** `run_turn` builds `{"type": "tool_result",
  "tool_use_id": ..., "content": result}` (`backend/app/agent.py:118-120`) with no `is_error`
  field, and `Tool.handler` is typed `-> Awaitable[str]` (`backend/app/tools.py:27`) with no way to
  signal failure. Any error-as-observation work — #28's guard and Phase 3 alike — needs the block
  builder to carry `is_error`, and probably needs the handler contract to return something richer
  than `str`. Worth deciding before #28 ships a guard that would otherwise hard-code
  `is_error`-less error strings.
- **`max_tokens=2048` is hardcoded** (`backend/app/provider.py:75`). #27 already covers moving it
  to settings; noting here that it is also what makes mid-`tool_use` truncation reachable at
  realistic tool-call sizes.
- **`pause_turn` is unhandled.** Not reachable today (no server-side tools), but the moment a
  server tool such as `web_search` is added, `pause_turn` (§1.3) requires re-sending the assistant
  turn to continue. `run_turn` breaks out of its loop whenever `tool_uses` is empty
  (`backend/app/agent.py:106-107`), so a paused turn would silently render as a finished, truncated
  answer — exactly the #27 failure mode, arriving through a different door.
- **No JSON-schema validation of `tool_input`** (documented as out of scope in
  `backend/app/tools.py:10`). The contract's answer is `strict: true` on tool definitions
  ([T1](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use)), which
  guarantees inputs match the schema and removes a whole class of invalid-tool-use round trips.
  Cheap to adopt when tool count grows; requires `additionalProperties: false` and `required` in
  each schema. Augur's `GET_CURRENT_TIME` schema (`backend/app/tools.py:77`) has neither.
- **Parallel tool calls are structurally supported but untested.** `run_turn` collects all
  `tool_uses` for a step and returns all `tool_results` in one user message
  (`backend/app/agent.py:115-121`) — correct per §1.5. But it dispatches them **sequentially** in a
  `for` loop, and if any one raises, none of the results are returned. Under §1.5 the right
  behaviour when a batch partially fails is to return `is_error: true` results for the calls that
  did not run, not to drop the batch. Same fix as #28, same place.

---

## 5. Bottom line for #28

Take **option 2**. The Anthropic contract does not merely permit returning an error `tool_result`
for an unknown tool — Anthropic's own SDK does exactly that, by default, in a bare
`except Exception`, with a message that names the missing tool and lists the available ones. Phase 3
generalises the mechanism; it does not change the answer for this case, and waiting keeps a
recoverable model hallucination classified as an internal server error. Put the guard in
`run_turn`'s dispatch loop (the seam Phase 3 will reuse), keep the operator log separate from the
model-facing observation, and add `is_error` to the `tool_result` builder while you are there.
