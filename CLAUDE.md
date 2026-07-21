# Augur

## Agent skills

### Issue tracker

Issues are tracked in the `heichan2000/Augur` GitHub repo via the `gh` CLI; external PRs are not a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary — `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Doc-drift check

Non-optional before any PR that touches code: grep the docs and docstrings for claims the change invalidates, and fix or file every hit. Same checkpoint as running the tests. See `docs/agents/doc-drift-check.md`.
