# Configurable CORS allowlist, closed by default

Design for [issue #22](https://github.com/heichan2000/Augur/issues/22). Date: 2026-07-21.

## Problem

The backend sets no CORS policy, so no browser origin other than the API's own can call
it. The Phase 1 frontend works only because it proxies every request through a Next.js
route. That proxy stays the documented default — this slice makes direct browser access
*possible*, not mandatory, and clears a blocker from the Phase 4 deployment checklist.

Policy was largely settled in [#21](https://github.com/heichan2000/Augur/issues/21):
allowlist from environment configuration, empty by default, credentials off, methods and
headers scoped rather than wildcarded, applied as middleware so it covers every route. The
open questions were seams and failure modes, and they are what this document decides.

## Decisions

### `create_app` takes an optional `Settings`

Middleware must be added at construction, so `create_app` needs the origin list. Today it
takes no arguments and `get_settings()` is called only in `lifespan` and as a per-route
`Depends`.

`create_app(settings: Settings | None = None)`, defaulting to `get_settings()`. The
default keeps the module-level `app = create_app()` and both existing endpoint tests
working unchanged, while giving CORS tests a seam that constructs a `Settings` object
directly instead of monkeypatching env and clearing an `lru_cache`.

The parameter is a testing seam, not a way to defer validation. Resolving the
`None` default is eager, so the module-level `app = create_app()` calls
`get_settings()` at import either way — an earlier draft of this document
claimed otherwise and was wrong. Implementation added `backend/tests/conftest.py`
to set a dummy `ANTHROPIC_API_KEY`, without which importing `app.main` raises and
`tests/test_health.py` and `tests/test_chat_endpoint.py` fail to collect. Failing
at import with a clear message is more diagnostic than failing at lifespan, and
the app cannot serve without the key regardless.

### Origins parse with `NoDecode`, not as a plain `list[str]`

```python
cors_allowed_origins: Annotated[list[str], NoDecode] = []
```

with a `mode="before"` validator that splits on commas. `NoDecode` is required:
pydantic-settings JSON-decodes complex types from the environment, so a bare
`list[str]` field raises on a comma-separated value. Verified against pydantic-settings
2.14.2 — unset, empty string, a list with stray whitespace and a trailing comma, and
direct list construction (which the injected-`Settings` seam depends on) all behave.

Rejected: a `str` field plus a separate list-valued property. Two names for one concept.

### Configured origins are canonicalized; invalid ones fail loudly

`starlette/middleware/cors.py:105` is `return origin in self.allow_origins` — bare exact
string membership, no normalization whatsoever. Every one of these natural config values
therefore silently fails to match a browser's `https://app.example`:

| Config value | Matches today |
| --- | --- |
| `https://app.example/` (trailing slash) | no |
| `https://App.Example` (mixed-case host) | no |
| `HTTPS://app.example` (mixed-case scheme) | no |
| `https://app.example:443` (default port) | no |
| `app.example` (no scheme) | no |

A module-level `canonicalize_origin(raw: str) -> str` in `app/config.py`, applied
per-entry by the validator, fixes the first four: lowercase scheme and host, strip a
trailing slash, drop the default port (`:443` https, `:80` http). Results are deduplicated
with order preserved.

The fifth is not mechanically fixable — `app.example` is ambiguous between http and https
— so it raises rather than guessing. So does anything else that is not a bare
`scheme://host[:port]`: non-http(s) scheme, missing host, embedded credentials,
path/query/fragment, unparseable port, and `null`.

**`*` is rejected too.** `cors.py:35` is `allow_all_origins = "*" in allow_origins`, so a
one-character environment value silently opens the API to every origin. Issue #21's user
story 14 wants a careless deploy unable to do that; rejecting `*` makes closed-by-default
a property of the system rather than of the default value alone. An operator who genuinely
wants an open API changes code, which is reviewable.

Failing loudly beats dropping-and-warning here: a silently skipped origin surfaces much
later as a browser CORS error with no obvious cause, and startup warnings are routinely
missed. That is precisely the bug canonicalization exists to prevent.

`get_settings()` already wraps `ValidationError` in a `RuntimeError` that tells the user to
set `ANTHROPIC_API_KEY`. That message widens to name whichever variable actually failed,
so a bad origin does not report itself as a missing API key.

### Middleware is added unconditionally

Including when the list is empty — one code path, and an empty list already means "permit
nothing". The observable cost: a preflight to `/chat` returns `400 Disallowed CORS origin`
instead of today's `405 Method Not Allowed`. Both refuse; the 400 is the more diagnostic.

`allow_credentials=False` (the API has no cookie or session auth),
`allow_methods=["GET", "POST"]`, `allow_headers=["Content-Type"]`.

`GET` is included alongside `POST` because the middleware also covers `/health`, and a
browser-based status page hitting it cross-origin is a plausible real use. This is still
far from the wildcard the acceptance criteria rule out.

### A rejected cross-origin request still returns 200

This constrains what the tests can assert, so it is recorded as a decision rather than
left for implementation to rediscover.

Starlette does **not** reject a simple (non-preflight) request from an unlisted origin. It
omits `access-control-allow-origin` and lets the response through; the *browser* is what
blocks the read. The criterion "a request from an unlisted origin is not permitted" can
only be asserted as **absence of the header**. A test written as
`assert response.status_code == 403` would fail against correct behaviour.

Preflight is different, and is where the closed default becomes directly visible: an
unlisted origin — or an empty allowlist — gets `400`.

Requests carrying no `Origin` header are untouched in every configuration, so the
"non-browser clients behave identically" criterion holds without special handling.

## Backend

- `app/config.py` — `canonicalize_origin`, the `cors_allowed_origins` field and its
  validator, and a widened `get_settings()` error message.
- `app/main.py` — `create_app` gains the optional `settings` parameter and adds
  `CORSMiddleware`.
- `backend/.env.example` — `CORS_ALLOWED_ORIGINS=`, with a comment stating that empty
  grants no cross-origin browser access and when to set it.

## Testing

Split by what is being asserted: pure normalization rules are a unit concern, observable
HTTP behaviour belongs at the endpoint seam with `TestClient` against a constructed
application, matching the existing endpoint tests.

**`tests/test_config.py`** — the canonicalization table above; every rejection case
(`app.example`, `*`, `not a url`, `https://`, `ftp://app.example`,
`https://app.example/foo`, `https://u:p@app.example`, `null`, `https://app.example:abc`);
deduplication with order preserved; unset and empty string both yielding `[]`; a
multi-entry comma string with stray whitespace and a trailing comma.

**`tests/test_cors.py`** — `TestClient(create_app(Settings(...)))`:

| Case | Assertion |
| --- | --- |
| Empty list, cross-origin simple request | `200`, **no** `access-control-allow-origin` |
| Empty list, preflight `/chat` | `400` |
| Configured origin, simple request | `200`, header echoes that origin |
| Unlisted origin, simple request | `200`, **no** header — *not* a status assertion |
| Two origins configured, each in turn | each echoes **its own** origin, never the other, never both |
| Configured origin, preflight `/chat` | `200`, `allow-origin` echoes it |
| ↳ same response | `allow-methods == "GET, POST"` |
| ↳ same response | `"content-type" in allow-headers.lower()` |
| Unlisted origin, preflight | `400` |
| Preflight requesting `DELETE` | `400` — method scoping is real |
| Preflight requesting an unlisted header | `400` — header scoping is real |
| No `Origin` header, every configuration | identical to today |
| Any configuration | no `access-control-allow-credentials` |
| Non-canonical config (`https://App.Example/`) | matches a browser's `https://app.example` |

Three of these rows encode findings that would otherwise produce wrong assertions:

- **`allow-headers` is a substring check, not equality.** `cors.py:59` unions in the four
  CORS-safelisted headers (`Accept`, `Accept-Language`, `Content-Language`,
  `Content-Type`) regardless of configuration, so `== "Content-Type"` fails. The substring
  form also survives a future change to Starlette's safelist.
- **The wildcard guard is behavioural, not string-based.** Under `allow_headers=["*"]`
  Starlette does not echo a literal `*`; `cors.py:36,60` make it mirror back whatever the
  request asked for. So `allow-headers != "*"` can never fire, and `"content-type" in
  allow-headers` *passes* under wildcard. Only the unlisted-header preflight
  distinguishes them: `400` when scoped, `200` if someone widens to `["*"]`. The `DELETE`
  row guards `allow_methods` the same way.
- **The non-canonical row** ties canonicalization to observable behaviour rather than
  leaving it a unit-test-only concern.

`/health` is exercised alongside `/chat` so "middleware, not per-route" is asserted rather
than assumed.

Behaviours above were verified empirically against the installed versions — starlette
1.3.1, fastapi 0.138.1, pydantic-settings 2.14.2. They are Starlette implementation
details, not spec guarantees; if a dependency bump changes them, the tests are where it
will surface.

## Out of scope

- Any change to the SSE event contract, `stream_chat`, or the frontend. The Next.js proxy
  remains the documented default path; no frontend change is required or implied.
- Authentication, rate limiting, and input guardrails — Phase 4.
- `allow_origin_regex` and wildcard subdomain patterns. An explicit list covers the known
  need; regex matching is a wider blast radius than this slice justifies.
- `expose_headers`. SSE responses carry everything the client needs in the body.
