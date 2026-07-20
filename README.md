# Augur

A developer-docs AI assistant built over the **FastAPI documentation** corpus — a portfolio project demonstrating production-shaped LLM engineering: streaming chat, retrieval-augmented generation with a real evaluation story, and an agentic tool loop.

> Built one phase at a time, each phase independently demoable, so the project is resume-worthy even if stopped early.

## Architecture

| Layer | Choice |
|---|---|
| Backend | FastAPI, fully async, SSE streaming, server-side history + tool orchestration |
| Frontend | Next.js + TypeScript, thin client over the SSE endpoint |
| LLM provider | Anthropic (Claude) |
| Vector store | pgvector (Postgres, Dockerized) |
| Corpus | FastAPI docs — Markdown from GitHub (MIT-licensed) |
| Orchestration | Hand-built RAG + agent loop through Phase 2; LangGraph added deliberately in Phase 3 |
| Evaluation | Golden set + retrieval metrics + LLM-as-judge, starting Phase 2 |

## Roadmap

- **Phase 1 — Streaming Chat Backbone & Tool-Calling Spine.** A deployed streaming chatbot with a polished Next.js UI that already routes through a tool-calling layer — the durable spine every later phase extends.
- **Phase 2 — RAG Over the Docs Corpus + Eval Harness.** "Chat with the FastAPI docs" with citations, backed by pgvector, plus a golden-set eval harness producing before/after retrieval and answer-quality numbers.
- **Phase 3 — Agentic Tool Loop (LangGraph).** The model *chooses* when to retrieve and which tools to call (`search_docs`, `search_github_issues`; `run_python` sandbox as a stretch).
- **Phase 4 — Hardening.** Docker, CI/CD, persistent history, guardrails.

Full specifications for every phase live in [`Augur-all-phases.md`](./Augur-all-phases.md).

## Local development

**Prerequisites:** Python 3.12+ with [uv](https://docs.astral.sh/uv/), and Node 20+ with npm.

### Backend (FastAPI)

```bash
cd backend
cp .env.example .env          # then set ANTHROPIC_API_KEY in .env
uv sync                       # install dependencies
uv run uvicorn app.main:app --reload
```

The API serves a health check at `http://localhost:8000/health`. Run the tests with:

```bash
uv run pytest
```

### Frontend (Next.js)

```bash
cd frontend
cp .env.example .env.local    # AUGUR_API_URL defaults to http://localhost:8000
npm install
npm run dev                   # http://localhost:3000
```

The browser talks only to the Next.js route at `/api/chat`, which proxies the SSE stream
through to the FastAPI backend — so the backend needs no CORS policy and its URL stays
server-side.

Run the frontend checks with:

```bash
npm test          # Vitest
npm run typecheck # tsc --noEmit
npm run lint
```

## Status

🚧 Phase 1 in progress.

## License

Code is released under the MIT License. The FastAPI documentation corpus is © its authors and used under its MIT license.
