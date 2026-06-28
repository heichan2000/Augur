# Augur — Full Project Specs (All Phases)

**Project:** Augur — a developer-docs AI assistant built over the FastAPI documentation corpus.

**Goal:** A portfolio project for an entry-level AI engineer role, built one phase at a time, each phase independently demoable so the project is resume-worthy even if stopped early.

**Locked architecture:**
- Backend: FastAPI, async, SSE streaming, server-side history + tool orchestration
- Frontend: Next.js + TypeScript, thin client over the SSE endpoint
- Provider: Anthropic (Claude)
- Vector store: pgvector (Postgres, Dockerized)
- Corpus: FastAPI docs, Markdown from GitHub (MIT)
- Orchestration: hand-built RAG + agent loop through Phase 2; LangGraph added deliberately in Phase 3
- Eval: golden set + retrieval metrics + LLM-as-judge, starting Phase 2
- Phase 3 tools: search_docs, search_github_issues (committed); run_python sandbox (stretch)

**Hiring-priority order if time runs short:** Phase 2 (RAG + eval) > Phase 1 (deployed chat) > Phase 3 (agent) > Phase 3 stretch (run_python).

---

# Phase 1 — Streaming Chat Backbone & Tool-Calling Spine

**Project:** Augur (developer-docs AI assistant, FastAPI corpus)
**Demoable outcome:** A deployed streaming chatbot with a polished Next.js UI that already routes through a tool-calling layer — the durable spine every later phase extends.

---

## Problem Statement

As an aspiring AI engineer, I need a real, deployed conversational app — not a notebook demo — that proves I can wire an LLM into a production-shaped backend with streaming, server-managed conversation state, and clean error/cost handling. It must be built so that retrieval (Phase 2) and agentic tools (Phase 3) are *additions*, not rewrites.

## Solution

A FastAPI backend exposing a streaming `/chat` SSE endpoint backed by the Anthropic Messages API, with conversation history managed server-side and a tool-calling dispatch layer present from day one (seeded with one trivial tool). A Next.js + TypeScript frontend renders the token stream in a clean chat UI. Token usage and errors are logged in a structured way from the first commit.

## User Stories

1. As a user, I want to send a message and see the assistant's reply stream in token-by-token, so that the app feels responsive.
2. As a user, I want my prior messages in a session remembered, so that I can have a multi-turn conversation without re-stating context.
3. As a user, I want a clean, modern chat interface, so that the product looks credible in a portfolio.
4. As a user, I want clear feedback when something fails (rate limit, network, model error), so that I'm not staring at a frozen screen.
5. As a developer, I want conversation history managed on the server, so that the frontend stays a thin client and the backend is the source of truth.
6. As a developer, I want the Anthropic message-array contract handled correctly (roles, content blocks), so that tool use and multi-turn work cleanly later.
7. As a developer, I want a tool-calling dispatch layer in place from Phase 1, so that adding RAG and real tools later requires no changes to the agent loop or API.
8. As a developer, I want a single trivial tool (e.g. `get_current_time`) registered, so that the tool path is exercised end-to-end before real tools exist.
9. As a developer, I want token counts and cost logged per request, so that I can speak to cost-awareness in interviews and reuse the logging in later phases.
10. As a developer, I want structured error handling around the model call, so that failures are typed and surfaced consistently.
11. As a developer, I want the streaming contract (SSE event shape) defined once, so that the frontend and all future phases consume the same format.
12. As a hiring manager reviewing the repo, I want a clear README and a live URL, so that I can try it in 30 seconds.

## Implementation Decisions

- **Backend:** FastAPI, fully async. Single streaming `/chat` endpoint using Server-Sent Events.
- **Provider:** Anthropic Messages API, streaming mode. Message array (system / user / assistant, content blocks) handled explicitly rather than via a wrapper framework — orchestration is hand-built in Phases 1–2.
- **Conversation state:** managed server-side, keyed by a session id. In-memory store is acceptable for Phase 1; the interface should allow swapping to a persistent store later without touching the endpoint.
- **Tool-calling spine:** a tool registry (schema + handler per tool) and a dispatch loop that detects `tool_use` blocks, executes the handler, appends the `tool_result`, and continues the turn. Seeded with one trivial tool. This is the load-bearing abstraction of the whole project — Phase 2's `search_docs` and Phase 3's tools register here unchanged.
- **SSE event contract:** a defined event schema (e.g. token delta, tool-call notice, error, done) consumed identically by the frontend now and in later phases.
- **Frontend:** Next.js + TypeScript, a thin client over the SSE endpoint. Renders streaming tokens and (optionally) surfaces tool-call activity. No business logic in the client; backend remains source of truth.
- **Observability:** structured logging of input/output token counts and computed cost per request; structured (typed) error handling around the provider call.
- **Config:** API keys and model name via environment variables; no secrets in the repo.

## Testing Decisions

- Test external behavior at the highest seam: the `/chat` endpoint. Given a request, assert the SSE stream produces well-formed events ending in `done`, and that a message triggering the trivial tool produces a tool-call → tool-result → final-answer sequence.
- Mock the Anthropic provider at the SDK boundary so tests are deterministic and free — assert the app's handling of streamed deltas and `tool_use` blocks, not the model's content.
- Test the tool dispatch layer in isolation: a registered tool's schema is exposed correctly and its handler is invoked with parsed arguments.
- Do not test implementation details (internal function names, private history representation) — only the endpoint contract and the tool-dispatch contract.

## Out of Scope

- Retrieval, embeddings, vector store (Phase 2).
- Real external tools and the LangGraph orchestration layer (Phase 3).
- Docker, CI/CD, auth, persistent DB-backed history, guardrails (Phase 4 / later).
- Multi-user accounts.

## Further Notes

The non-negotiable Phase 1 principle: **make Phases 2 and 3 extensions, not rewrites.** The tool-calling spine and the SSE contract are the two pieces that buy that. Default corpus/target is FastAPI docs but no corpus work happens in this phase.


---

# Phase 2 — RAG Over the Docs Corpus + Eval Harness

**Project:** Augur
**Demoable outcome:** A "chat with the FastAPI docs" assistant that answers with citations, backed by pgvector, with a golden-set evaluation harness producing before/after retrieval and answer-quality numbers.

---

## Problem Statement

As an aspiring AI engineer, I need to demonstrate the single most-asked-about skill in entry AI-engineer interviews — Retrieval-Augmented Generation — and, crucially, that I can *measure* its quality. Most portfolios show a RAG demo with no evaluation; the differentiator is a documented before/after eval story.

## Solution

An ingestion pipeline that pulls the FastAPI docs (Markdown from GitHub), chunks them with awareness of headings and code blocks, embeds them, and stores vectors in pgvector. The chat turn retrieves top-k relevant chunks and augments the prompt, and answers cite the source sections. A golden evaluation set (~20–30 hand-written Q&A pairs) measures retrieval hit-rate and LLM-as-judge answer quality, re-run on every tuning change with results recorded.

## User Stories

1. As a user, I want to ask a question about FastAPI and get an answer grounded in the actual docs, so that I can trust it over the model's stale training memory.
2. As a user, I want answers to cite which doc section they came from, so that I can verify and read more.
3. As a user, I want the assistant to say when the docs don't cover something, so that I'm not given a confident hallucination.
4. As a developer, I want docs ingested from a `git clone` of the repo, so that there is no scraper to maintain and the source is MIT-licensed.
5. As a developer, I want chunking that respects Markdown headings and keeps code blocks intact, so that retrieved chunks are coherent and runnable.
6. As a developer, I want embeddings stored in pgvector, so that retrieval is inspectable SQL and the storage layer is something I understand end-to-end.
7. As a developer, I want retrieval exposed through the Phase 1 tool spine as a `search_docs` capability, so that it plugs in without changing the agent loop.
8. As a developer, I want a golden eval set written before I tune anything, so that I have an honest baseline.
9. As a developer, I want retrieval metrics (is the correct chunk in top-k?), so that I can isolate retrieval quality from answer quality.
10. As a developer, I want LLM-as-judge scoring of faithfulness and answer relevance, so that I can quantify answer quality.
11. As a developer, I want to re-run the eval and record numbers whenever I change chunk size, embedding model, or top-k, so that I have a before/after table for my README and interviews.
12. As a developer, I want the augmentation prompt to instruct citation and abstention, so that grounding and honesty are enforced.
13. As a hiring manager, I want to see an eval results table in the README, so that I can tell this candidate measures their work.

## Implementation Decisions

- **Corpus:** FastAPI docs, Markdown, sourced from a clone of the GitHub repo (MIT). Default; swappable for another Markdown-native library via config.
- **Ingestion:** parse Markdown, chunk on heading boundaries with a size target and overlap, never split inside a fenced code block. Store chunk text + source metadata (file, heading path) for citations.
- **Embeddings:** an embedding model called via API; model name in config so it can be swapped and the swap measured by the eval harness.
- **Vector store:** pgvector in Postgres (Docker container). Similarity query via SQL; top-k configurable.
- **Retrieval as a tool:** implemented as `search_docs`, registered in the Phase 1 tool registry. In Phase 2 the chat turn always retrieves; in Phase 3 the agent will *choose* when to call it — same handler, no rewrite.
- **Augmentation:** retrieved chunks injected into the prompt with explicit instructions to cite source sections and to abstain when coverage is insufficient.
- **Eval harness:** a golden set of ~20–30 Q&A pairs with known correct source sections. Computes (a) retrieval hit-rate / rank of the correct chunk in top-k, and (b) LLM-as-judge faithfulness + answer-relevance scores. Outputs a results table; runs are recorded so tuning changes are comparable.
- **Stretch:** add a second Markdown-native library (e.g. Pydantic or dbt) to demonstrate multi-source retrieval and disambiguation.

## Testing Decisions

- Test the ingestion contract: given a known Markdown input, chunks respect heading boundaries and never split code fences, and metadata is attached.
- Test retrieval behavior at the `search_docs` seam: for a known query, the expected chunk appears in top-k. This overlaps with the eval harness, which *is* the primary behavioral test of retrieval quality.
- Test the chat endpoint's grounded-answer path: with retrieval mocked to return known chunks, the answer includes citations and abstains when given irrelevant chunks.
- The eval harness itself is the headline test artifact — it tests real external behavior (retrieval correctness, answer quality) rather than implementation details, and its recorded outputs double as portfolio evidence.

## Out of Scope

- Agentic tool *selection* and LangGraph (Phase 3) — Phase 2 retrieves on every turn.
- External live tools like GitHub issue search (Phase 3).
- Deployment, Docker Compose for the full app, CI/CD (Phase 4) — a local Postgres container is fine here.
- Re-ranking models and hybrid (keyword + vector) search unless pursued as an optional extension.

## Further Notes

The eval harness is the highest-leverage half-day of the whole month — it is what makes the portfolio read as engineering rather than a tutorial. Build the golden set *before* tuning so the baseline is honest. Keep the before/after numbers; they are interview ammunition.


---

# Phase 3 — Agentic Tools & LangGraph Orchestration

**Project:** Augur
**Demoable outcome:** The assistant becomes an agent that reasons about which tool to call — choosing between doc retrieval and a live GitHub issue search — orchestrated with LangGraph, with a stretch sandboxed code-execution tool.

---

## Problem Statement

As an aspiring AI engineer, I need to show I understand the agent loop (reason → act → observe → repeat) from first principles *and* that I can use the industry-standard orchestration tooling. Retrieving on every turn (Phase 2) isn't agentic; a real agent decides when retrieval is even the right move and can reach for live data the model's training never saw.

## Solution

Re-architect the turn as an agent loop where the model chooses among registered tools: `search_docs` (Phase 2 retrieval, now agent-selected) and `search_github_issues` (a live GitHub API call against the FastAPI repo for current bug/issue context). The loop is re-expressed in LangGraph as a deliberate "I can also use standard tooling" layer, having first been hand-built in Phases 1–2. A sandboxed `run_python` tool that verifies snippets actually run is the stretch goal.

## User Stories

1. As a user, I want the assistant to decide whether to search the docs, search GitHub issues, or just answer, so that I get the most relevant grounding for my question.
2. As a user, I want to ask "is this a known bug?" and get an answer informed by current GitHub issues, so that I get information newer than the model's training cutoff.
3. As a user, I want to see which tools the agent used to reach its answer, so that the reasoning is transparent.
4. As a user (stretch), I want the assistant to verify a code snippet runs before recommending it, so that I trust the answer.
5. As a developer, I want the agent to select tools rather than always retrieving, so that the system is genuinely agentic.
6. As a developer, I want `search_docs` reused unchanged from Phase 2, so that the pluggable-tool design pays off.
7. As a developer, I want `search_github_issues` to call the live GitHub API against the FastAPI repo, so that the agent has access to current data.
8. As a developer, I want the loop re-expressed in LangGraph, so that I have the framework on my resume and can speak to the build-by-hand-vs-framework tradeoff.
9. As a developer, I want adding a future tool to require only a handler + schema registration, so that the design is demonstrably extensible.
10. As a developer, I want multi-step agent runs (call a tool, observe, possibly call another) to work and terminate safely with a max-step bound, so that the agent can't loop forever.
11. As a developer (stretch), I want a sandboxed `run_python` executor isolated from the host, so that code verification doesn't create a security hole.
12. As a developer, I want tool-call traces logged, so that I can debug and demo the agent's decisions.
13. As a hiring manager, I want to see both a from-scratch agent loop and a LangGraph version in the history/README, so that I can assess depth and tooling fluency.

## Implementation Decisions

- **Agent loop:** the model is given the tool schemas and chooses which to call. The loop executes the chosen tool, feeds the result back, and continues until the model produces a final answer or a max-step bound is hit. Built on the Phase 1 dispatch spine.
- **Committed tools:**
  - `search_docs` — the Phase 2 pgvector retrieval, unchanged, now invoked at the agent's discretion rather than every turn.
  - `search_github_issues` — live call to api.github.com against the FastAPI repo; returns matching issues for "known bug?" style questions. Free, no scraping, surfaces post-training data.
- **Orchestration:** re-express the loop in LangGraph (nodes for model-call, tool-execution, and routing). This is deliberate and additive — the hand-built version from Phases 1–2 stays in history so the contrast is visible. Be able to articulate *why* LangGraph (graph structure, state, conditional edges) over raw loops.
- **Safety/termination:** max-step bound on the agent loop; tool errors are caught and returned to the model as observations rather than crashing the turn.
- **Transparency:** tool-call activity surfaced over the existing SSE contract so the frontend can show which tools ran.
- **Stretch — `run_python`:** a sandboxed executor (isolated from the host filesystem/network, resource-bounded) so the agent can verify snippets. Treated as stretch specifically because sandboxing is the real security surface of the project; not committed scope.
- **Extensibility (free from the design):** new tools = handler + schema registration only; no changes to the loop, API, or frontend.

## Testing Decisions

- Test tool *selection* at the agent seam with the model mocked: given a query and a mocked model that emits a `tool_use` for `search_github_issues`, assert that handler runs and its result is fed back as an observation.
- Test `search_github_issues` against a mocked GitHub API boundary: correct query construction and result parsing, without hitting the network in tests.
- Test loop termination: a mocked model that keeps calling tools hits the max-step bound and exits cleanly.
- Test that `search_docs` still satisfies its Phase 2 contract after being moved under agent selection (regression).
- For the LangGraph version, test the same external behaviors (selection, termination, observation feedback) — not graph internals.
- Stretch `run_python`: test that disallowed operations are blocked and that a valid snippet's output is captured; isolation is the behavior under test.
- Only external behavior is tested — tool choice, termination, result feedback, sandbox isolation — never internal graph node names or private state shapes.

## Out of Scope

- New domains or corpora beyond Phase 2's.
- Deployment, Docker Compose for the full stack, CI/CD, production guardrails (Phase 4).
- Long-term agent memory across sessions, multi-agent systems.
- Anything beyond the three named tools; the point is the design *accommodates* more, not that more ship now.

## Further Notes

The elegant payoff of the whole architecture lands here: RAG becomes *one tool among several* that the agent chooses to use. The strongest interview story is the build-by-hand-then-adopt-LangGraph arc — keep both versions visible. `run_python` is high-wow but is correctly a stretch; do not let sandboxing eat the committed scope.


---

# Phase 4 — Productionize, Deploy & Package for Hiring

**Project:** Augur
**Demoable outcome:** A publicly deployed, Dockerized, CI/CD-backed assistant with guardrails, strong READMEs + architecture diagrams, and a resume rewritten around the work.

---

## Problem Statement

As a job-seeking AI engineer, a great app that only runs on my laptop is invisible to hiring managers. I need it deployed at a public URL, reproducible via Docker, continuously tested via CI, hardened with basic guardrails, and packaged so a reviewer grasps it in under a minute — and I need my resume to lead with it.

## Solution

Containerize the full stack (FastAPI + pgvector + Next.js) with Docker Compose, add GitHub Actions CI running the test suites from Phases 1–3, deploy to a public host, add input/output guardrails and rate limiting, write per-phase READMEs with architecture diagrams and the Phase 2 eval table, and rewrite the resume around the project with measurable outcomes.

## User Stories

1. As a hiring manager, I want a live URL I can try immediately, so that I see a working product, not just code.
2. As a hiring manager, I want a README with an architecture diagram, so that I understand the system in under a minute.
3. As a hiring manager, I want to see the eval results table, so that I can tell the candidate measures their work.
4. As a developer, I want the whole stack to come up with one `docker compose up`, so that the project is reproducible by anyone.
5. As a developer, I want CI running my tests on every push, so that the green badge signals engineering discipline.
6. As a developer, I want input validation and output moderation, so that the public demo can't be trivially abused.
7. As a developer, I want rate limiting on the endpoint, so that my API costs are bounded when the link is shared.
8. As a developer, I want secrets handled via environment/host config, so that nothing sensitive is in the repo.
9. As a developer, I want hallucination guardrails (enforced citation/abstention from Phase 2) reflected in the deployed behavior, so that the public demo stays honest.
10. As a developer, I want a clear cost story (the Phase 1 token logging) summarized, so that I can speak to running it economically.
11. As a job-seeker, I want my resume to lead with this project — stack, what I built, measurable outcomes — so that it reads as real engineering.
12. As a job-seeker, I want a short portfolio page linking the live demo, repo, and a brief writeup, so that applications point to one strong artifact.

## Implementation Decisions

- **Containerization:** Docker Compose orchestrating FastAPI backend, Postgres+pgvector, and the Next.js frontend. One-command bring-up. The Phase 2 Postgres container graduates into the Compose stack.
- **CI/CD:** GitHub Actions running the Phase 1–3 test suites on push/PR; build the images in CI. Surface a status badge in the README.
- **Deployment:** a public host with a free/low tier (e.g. Railway, Render, Fly.io, or a cloud free tier). Backend + DB hosted; frontend deployed and pointed at the backend URL.
- **Guardrails:** input validation (length/shape), output moderation, and the citation/abstention behavior from Phase 2 enforced in production. Rate limiting on `/chat` to bound cost.
- **Secrets:** environment variables on the host; none committed.
- **Documentation:** a top-level README (what it is, live URL, architecture diagram, quickstart, eval table) plus per-phase notes telling the build story (hand-built loop → RAG+eval → agent+LangGraph). Architecture diagram shows backend, vector store, tool layer, frontend, and external tools.
- **Resume + portfolio:** resume bullet(s) leading with the project — stack named, what was built, and measurable outcomes (e.g. retrieval accuracy from the eval table, multi-tool agent, deployed at URL). A lightweight portfolio page links demo + repo + writeup.

## Testing Decisions

- CI is the headline testing artifact for this phase: the Phase 1–3 suites must pass in a clean CI environment, proving the project builds and tests reproducibly off a fresh checkout.
- Smoke-test the deployed environment: the live `/chat` endpoint streams and the agent can call tools end-to-end against the real (hosted) Postgres.
- Test guardrail behavior: oversized/malformed input is rejected; rate limiting triggers as configured.
- Test only external behavior — deployment health, guardrail enforcement, CI green — not implementation details.

## Out of Scope

- New features or tools beyond Phases 1–3.
- Autoscaling, multi-region, advanced infra.
- User accounts / auth beyond what's needed to keep the public demo safe.
- Paid infrastructure beyond free/low tiers.

## Further Notes

Depth over breadth is the through-line: one deployed, documented, tested, measured project beats a pile of half-finished ones. Be able to explain every decision — interviewers drill into whatever is on the resume. Lean on the existing SWE background; much of this phase (Docker, CI/CD, env config, READMEs) is solid backend engineering, which is the candidate's real advantage.


---

