# Configurable CORS Allowlist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a deployment grant specific browser origins direct access to the backend API through environment configuration, with an empty, closed default and loud rejection of unusable or dangerous values.

**Architecture:** A `cors_allowed_origins` field on the existing `Settings` object parses a comma-separated environment variable, canonicalizing each entry (lowercase scheme and host, no trailing slash, no default port) and raising on anything ambiguous or dangerous — including `*`. `create_app` gains an optional `Settings` parameter so it can add Starlette's `CORSMiddleware` at construction time, covering every route rather than being wired per-route.

**Tech Stack:** Python 3, FastAPI 0.138.1, Starlette 1.3.1, pydantic 2.13.4, pydantic-settings 2.14.2, pytest 8.

**Source spec:** `docs/superpowers/specs/2026-07-21-cors-allowlist-design.md` (issue [#22](https://github.com/heichan2000/Augur/issues/22)).

## Global Constraints

Every task's requirements implicitly include this section.

- **Middleware configuration is exactly:** `allow_credentials=False`, `allow_methods=["GET", "POST"]`, `allow_headers=["Content-Type"]`. Never a wildcard for any of the three.
- **Middleware is added unconditionally**, including when the allowlist is empty. One code path; an empty list already means "permit nothing".
- **The default is `[]`** — an unconfigured deployment grants no cross-origin browser access.
- **`*` is rejected as a configured origin.** Closed-by-default must be a property of the system, not merely of the default value.
- **Invalid origins raise; they are never dropped-with-a-warning.** A silently skipped origin resurfaces much later as an unexplained browser CORS error.
- **Canonicalization rules:** lowercase the scheme and host, strip a trailing slash, drop the scheme's default port (`:443` for https, `:80` for http). Deduplicate with **order preserved**.
- **`create_app(settings: Settings | None = None)`**, defaulting to `get_settings()`.
- **Never assert a status code for a rejected *simple* cross-origin request.** Starlette returns `200` and omits the header; the browser is what blocks the read. The only correct assertion is **absence of `access-control-allow-origin`**. A `403` assertion would fail against correct behaviour.
- **`allow-headers` must be asserted as a substring check, never equality.** Starlette unions in the four CORS-safelisted headers regardless of configuration, so the response reads `Accept, Accept-Language, Content-Language, Content-Type`.
- **Out of scope:** any change to the SSE event contract, `stream_chat`, or the frontend. Also `allow_origin_regex`, wildcard subdomain patterns, `expose_headers`, auth, and rate limiting.
- **Working directory for all commands is `backend/`.**

### Verified behaviour (do not re-derive; do not "fix" tests that rely on these)

These were confirmed empirically against the installed versions during planning:

| Situation | Actual result |
| --- | --- |
| Empty allowlist, simple cross-origin request | `200`, no `access-control-allow-origin` |
| Empty allowlist, preflight `/chat` | `400`, body `Disallowed CORS origin` |
| Unlisted origin, simple request | `200`, **no** header |
| Preflight with a disallowed method | `400`, body `Disallowed CORS method` |
| Preflight with an unlisted header | `400`, body `Disallowed CORS headers` |
| Preflight, listed origin | `allow-methods` is exactly `GET, POST` |
| Preflight, listed origin | `allow-headers` is `Accept, Accept-Language, Content-Language, Content-Type` |
| Any configuration | no `access-control-allow-credentials` header |
| Request with no `Origin` header | untouched in every configuration |
| Under `allow_headers=["*"]`, preflight for `x-custom` | `200`, echoes `x-custom` — this is why the unlisted-header preflight is the only real wildcard guard |
| Today, before this change: preflight `/chat` | `405` (becomes `400` after this change — both refuse) |

### Decision that overrides the spec

The spec's "Rejected: calling `get_settings()` inline" paragraph argues the chosen design avoids moving `ANTHROPIC_API_KEY` validation to import time. **That is incorrect** — resolving the `None` default inside `create_app` is eager, so `app = create_app()` at `app/main.py:57` validates at import either way. Confirmed: there is no `backend/.env`, `ANTHROPIC_API_KEY` is unset, `import app.main` succeeds today, and `get_settings()` raises `RuntimeError` without the key. `tests/test_health.py` and `tests/test_chat_endpoint.py` import `app.main` at module scope with no `conftest.py` to set a key.

**Resolution (decided by the human, 2026-07-21):** accept import-time validation and add `tests/conftest.py` to set a dummy key for collection. Task 3 implements this; Task 4 corrects the spec's paragraph.

## File Structure

| File | Responsibility |
| --- | --- |
| `backend/app/config.py` (modify) | `canonicalize_origin`, the `cors_allowed_origins` field and its validator, widened `get_settings()` error |
| `backend/app/main.py` (modify) | `create_app` settings parameter; adds `CORSMiddleware` |
| `backend/tests/conftest.py` (create) | Sets a dummy `ANTHROPIC_API_KEY` so module-scope imports of `app.main` survive collection |
| `backend/tests/test_config.py` (modify) | Canonicalization and parsing unit tests |
| `backend/tests/test_cors.py` (create) | Observable HTTP behaviour through `TestClient` |
| `backend/.env.example` (modify) | Documents `CORS_ALLOWED_ORIGINS` and the closed default |

---

### Task 1: `canonicalize_origin`

A pure function. No settings, no middleware, no HTTP.

**Files:**
- Modify: `backend/app/config.py`
- Test: `backend/tests/test_config.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `canonicalize_origin(raw: str) -> str` at module scope in `app.config`. Returns the canonical `scheme://host[:port]` form. Raises `ValueError` with an explanatory message for any input it cannot canonicalize unambiguously. Task 2's validator calls this per entry.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_config.py` already has `import pytest` and
`from app.config import get_settings` at the top — widen the second to
`from app.config import canonicalize_origin, get_settings` rather than adding a
duplicate import line. Then append:

```python
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://app.example", "https://app.example"),
        ("https://app.example/", "https://app.example"),
        ("https://App.Example", "https://app.example"),
        ("HTTPS://app.example", "https://app.example"),
        ("https://app.example:443", "https://app.example"),
        ("http://app.example:80", "http://app.example"),
        ("  https://app.example  ", "https://app.example"),
        ("http://localhost:3000", "http://localhost:3000"),
        ("https://app.example:8443", "https://app.example:8443"),
    ],
)
def test_canonicalize_normalizes(raw, expected):
    assert canonicalize_origin(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "app.example",              # ambiguous scheme — cannot be guessed
        "not a url",
        "https://",                 # no host
        "ftp://app.example",        # non-http(s) scheme
        "https://app.example/foo",  # path
        "https://app.example?q=1",  # query
        "https://app.example#frag", # fragment
        "https://u:p@app.example",  # embedded credentials
        "null",
        "https://app.example:abc",  # unparseable port
        "",
        "   ",
    ],
)
def test_canonicalize_rejects(raw):
    with pytest.raises(ValueError):
        canonicalize_origin(raw)


def test_canonicalize_rejects_wildcard_with_a_pointed_message():
    """'*' must not be usable as a configured origin.

    starlette's `allow_all_origins = "*" in allow_origins` means a single
    stray character would open the API to every origin on the web. The
    message is asserted because the generic "must start with http://"
    wording would badly misdescribe what someone typing '*' intended.
    """
    with pytest.raises(ValueError) as exc_info:
        canonicalize_origin("*")
    assert "*" in str(exc_info.value)
    assert "every origin" in str(exc_info.value)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config.py -q`

Expected: FAIL at collection — `ImportError: cannot import name 'canonicalize_origin' from 'app.config'`.

- [ ] **Step 3: Write the implementation**

In `backend/app/config.py`, add `from urllib.parse import urlsplit` to the imports, then add this above the `Settings` class:

```python
# Starlette compares the browser's Origin header byte-for-byte against the
# allowlist — starlette/middleware/cors.py:105 is
# ``return origin in self.allow_origins``, with no normalization at all.
#
# So every natural way to write an origin in configuration silently fails
# to match: a trailing slash, a capitalized host, an explicit :443. The
# failure surfaces far from its cause, as an unexplained CORS error in
# someone's browser console, which is exactly what this function exists to
# prevent.
#
# What cannot be fixed unambiguously raises instead. ``app.example`` could
# mean either scheme, and guessing wrong reproduces the silent failure.
_DEFAULT_PORTS = {"http": 80, "https": 443}


def canonicalize_origin(raw: str) -> str:
    """Return the canonical ``scheme://host[:port]`` form of one configured origin.

    Lowercases the scheme and host, strips a trailing slash, and drops the
    scheme's default port. Raises ``ValueError`` — never returns a guess —
    for anything that is not a bare ``scheme://host[:port]``.
    """
    value = raw.strip()

    if value == "*":
        raise ValueError(
            "'*' is not usable as a configured origin: starlette reads a literal "
            "'*' as permission for every origin on the web, so one stray "
            "character would open this API to all of them. List each origin "
            "explicitly. An API that is genuinely meant to be open should say so "
            "in code, where the choice is reviewable."
        )
    if not value:
        raise ValueError("origin is empty")

    parts = urlsplit(value)
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS:
        raise ValueError(
            f"{raw!r}: an origin must start with http:// or https://. "
            "A bare host is ambiguous between the two, so it cannot be assumed."
        )
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise ValueError(
            f"{raw!r}: an origin is only a scheme, host, and optional port — "
            "it must not carry a path, query, or fragment."
        )
    if parts.username is not None or parts.password is not None:
        raise ValueError(f"{raw!r}: an origin must not embed credentials.")

    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError(f"{raw!r}: the port is not a number.") from exc

    host = parts.hostname
    if not host:
        raise ValueError(f"{raw!r}: no host.")

    if port is None or port == _DEFAULT_PORTS[scheme]:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config.py -q`

Expected: PASS — 3 existing tests plus 22 new parametrized cases.

- [ ] **Step 5: Run the full suite to confirm nothing regressed**

Run: `uv run pytest -q`

Expected: `100 passed` grows to `122 passed`, no failures, no new warnings.

- [ ] **Step 6: Commit**

```bash
git add backend/app/config.py backend/tests/test_config.py
git commit -m "feat(config): canonicalize configured CORS origins"
```

---

### Task 2: The `cors_allowed_origins` setting

**Files:**
- Modify: `backend/app/config.py`
- Test: `backend/tests/test_config.py`

**Interfaces:**
- Consumes: `canonicalize_origin(raw: str) -> str` from Task 1.
- Produces: `Settings.cors_allowed_origins: list[str]`, already canonicalized and deduplicated, defaulting to `[]`. Read from the `CORS_ALLOWED_ORIGINS` environment variable as a comma-separated string. Task 4 passes this straight to `CORSMiddleware(allow_origins=...)`.
- Produces: `get_settings()` raises `RuntimeError` naming **whichever** environment variables failed validation, not always `ANTHROPIC_API_KEY`.

**Why `NoDecode` is required:** pydantic-settings JSON-decodes complex types from the environment, so a bare `list[str]` field raises on a comma-separated value. `NoDecode` hands the raw string to the validator instead. Verified against pydantic-settings 2.14.2.

- [ ] **Step 1: Write the failing tests**

Widen the config import at the top of `backend/tests/test_config.py` again, to
`from app.config import Settings, canonicalize_origin, get_settings`. Then append:

```python
def test_origins_default_to_empty_when_unset(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-abc")
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    get_settings.cache_clear()
    assert get_settings().cors_allowed_origins == []


def test_empty_string_yields_empty_list(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-abc")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "")
    get_settings.cache_clear()
    assert get_settings().cors_allowed_origins == []


def test_comma_string_splits_and_canonicalizes(monkeypatch):
    """Stray whitespace and a trailing comma are what a real .env looks like."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-abc")
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        " https://App.Example/ , http://localhost:3000 ,",
    )
    get_settings.cache_clear()
    assert get_settings().cors_allowed_origins == [
        "https://app.example",
        "http://localhost:3000",
    ]


def test_duplicates_collapse_preserving_order(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-abc")
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        "https://b.example,https://a.example,https://B.Example/",
    )
    get_settings.cache_clear()
    assert get_settings().cors_allowed_origins == [
        "https://b.example",
        "https://a.example",
    ]


def test_settings_accepts_a_direct_list(monkeypatch):
    """The seam Task 4's tests depend on: construct Settings without env."""
    settings = Settings(
        anthropic_api_key="k", cors_allowed_origins=["https://App.Example/"]
    )
    assert settings.cors_allowed_origins == ["https://app.example"]


def test_invalid_origin_names_itself_not_the_api_key(monkeypatch):
    """A bad origin must not report itself as a missing ANTHROPIC_API_KEY."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-abc")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "app.example")
    get_settings.cache_clear()
    with pytest.raises(RuntimeError) as exc_info:
        get_settings()
    message = str(exc_info.value)
    assert "CORS_ALLOWED_ORIGINS" in message
    assert "ANTHROPIC_API_KEY" not in message


def test_wildcard_origin_is_rejected_at_settings_level(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-abc")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "*")
    get_settings.cache_clear()
    with pytest.raises(RuntimeError) as exc_info:
        get_settings()
    assert "CORS_ALLOWED_ORIGINS" in str(exc_info.value)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config.py -q`

Expected: FAIL. `test_origins_default_to_empty_when_unset` fails with `AttributeError: 'Settings' object has no attribute 'cors_allowed_origins'`; the two error-message tests fail because the message still hardcodes `ANTHROPIC_API_KEY`.

- [ ] **Step 3: Write the implementation**

In `backend/app/config.py`, extend the imports:

```python
import functools
from typing import Annotated
from urllib.parse import urlsplit

from pydantic import ValidationError, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
```

Replace the `Settings` class and `get_settings` with:

```python
class Settings(BaseSettings):
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-4-6"

    # Browser origins permitted to call this API directly, as a
    # comma-separated list (``CORS_ALLOWED_ORIGINS``). Empty by default:
    # an unconfigured deployment grants no cross-origin browser access at
    # all, and the Next.js proxy remains the documented default path.
    #
    # ``NoDecode`` is required. pydantic-settings JSON-decodes complex
    # types from the environment, so a bare ``list[str]`` would raise on a
    # comma-separated value; ``NoDecode`` hands the raw string to the
    # validator below instead.
    cors_allowed_origins: Annotated[list[str], NoDecode] = []

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def _canonicalize_origins(cls, value):
        # Also runs for a directly-constructed list, which is the seam the
        # CORS tests use to build a Settings without touching the
        # environment — so a list gets canonicalized exactly like a string.
        if isinstance(value, str):
            entries = [part.strip() for part in value.split(",")]
        else:
            entries = list(value)

        origins: list[str] = []
        for entry in entries:
            if not entry.strip():
                continue  # tolerate a trailing comma and blank padding
            origin = canonicalize_origin(entry)
            if origin not in origins:
                origins.append(origin)  # dedupe, preserving order
        return origins


@functools.lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        # Name whichever variables actually failed. This used to say
        # ANTHROPIC_API_KEY unconditionally, which would misreport a bad
        # CORS_ALLOWED_ORIGINS as a missing API key and send the reader
        # looking in the wrong place.
        names = sorted(
            {str(error["loc"][0]).upper() for error in exc.errors() if error["loc"]}
        )
        raise RuntimeError(
            f"Invalid backend configuration: {', '.join(names)}. "
            "Copy backend/.env.example to backend/.env and check these variables.\n"
            f"{exc}"
        ) from exc
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config.py -q`

Expected: PASS, including the pre-existing `test_missing_key_raises_clear_error` — with the key unset, `loc` is `anthropic_api_key`, so the widened message still contains `ANTHROPIC_API_KEY`.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`

Expected: `129 passed`, no failures.

- [ ] **Step 6: Commit**

```bash
git add backend/app/config.py backend/tests/test_config.py
git commit -m "feat(config): add the CORS_ALLOWED_ORIGINS setting"
```

---

### Task 3: `create_app` takes an optional `Settings`

The seam only. **Do not add `CORSMiddleware` in this task** — that is Task 4.

**Files:**
- Modify: `backend/app/main.py`
- Create: `backend/tests/conftest.py`
- Test: `backend/tests/test_health.py`

**Interfaces:**
- Consumes: `Settings` and `get_settings` from `app.config` (Task 2).
- Produces: `create_app(settings: Settings | None = None) -> FastAPI`. When `settings` is `None` it calls `get_settings()`. Task 4 reads `settings.cors_allowed_origins` inside this function.

**Read this before starting.** Resolving the `None` default is eager, so the module-level `app = create_app()` at the bottom of `app/main.py` now calls `get_settings()` **at import time**. Without a key in the environment that raises `RuntimeError`, which breaks collection of `tests/test_health.py` and `tests/test_chat_endpoint.py` — both import `app.main` at module scope. There is no `backend/.env` in this repo and no existing `conftest.py`. The new `tests/conftest.py` is what keeps collection working; it is not optional and not incidental cleanup. This trade was decided deliberately: failing at import with a clear message is more diagnostic than failing at lifespan, and the app cannot serve without the key regardless.

- [ ] **Step 1: Write the failing test**

In `backend/tests/test_health.py`, widen the existing config import to
`from app.config import Settings, get_settings`, then append:

```python
def test_create_app_uses_injected_settings_without_calling_get_settings(monkeypatch):
    """The seam the CORS tests need: build an app from a constructed Settings.

    get_settings is replaced with a landmine so the test actually discriminates.
    Without it, a create_app that ignored its parameter and called get_settings()
    anyway would still return a working app and the test would pass.

    Note there is no `with` here on purpose: entering the TestClient context
    runs `lifespan`, which calls get_settings() by design and would trip the
    landmine. This test is about create_app, not startup.
    """
    import app.main as main_module

    def _boom():
        raise AssertionError("create_app must not call get_settings() when given one")

    monkeypatch.setattr(main_module, "get_settings", _boom)

    app = create_app(Settings(anthropic_api_key="injected-key"))
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_health.py -q`

Expected: FAIL — `TypeError: create_app() takes 0 positional arguments but 1 was given`.

- [ ] **Step 3: Create `backend/tests/conftest.py`**

```python
"""Test-collection environment.

``app/main.py`` ends with a module-level ``app = create_app()``, and
``create_app`` now resolves ``get_settings()`` when no Settings is passed.
That runs at *import* time, so importing ``app.main`` raises RuntimeError
unless ANTHROPIC_API_KEY is set — and ``tests/test_health.py`` and
``tests/test_chat_endpoint.py`` both import it at module scope.

pytest imports conftest before collecting test modules, so setting a dummy
key here is what keeps collection working. ``setdefault`` means a real key
in the environment still wins, and tests that need a specific value (or its
absence) go on setting or deleting it themselves via monkeypatch.
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-collection")
```

- [ ] **Step 4: Add the parameter to `create_app`**

In `backend/app/main.py`, change the import line and the function signature:

```python
from app.config import Settings, get_settings
```

```python
def create_app(settings: Settings | None = None) -> FastAPI:
    configure_logging()

    # Resolved eagerly, so the module-level `app = create_app()` below
    # validates configuration at import. The parameter exists so tests can
    # construct a Settings directly instead of setting environment
    # variables and clearing an lru_cache.
    if settings is None:
        settings = get_settings()

    application = FastAPI(lifespan=lifespan)
```

Leave the rest of the function, the route bodies, and `app = create_app()` unchanged.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_health.py -q`

Expected: PASS, 2 tests.

- [ ] **Step 6: Prove the conftest is load-bearing, not decoration**

Temporarily move the conftest aside and confirm collection breaks without it:

```bash
mv tests/conftest.py tests/conftest.py.off
env -u ANTHROPIC_API_KEY uv run pytest tests/test_health.py -q
mv tests/conftest.py.off tests/conftest.py
env -u ANTHROPIC_API_KEY uv run pytest tests/test_health.py -q
```

Expected: without the conftest, collection **errors** with `RuntimeError: Invalid backend configuration: ANTHROPIC_API_KEY`. With it restored, tests pass. Report both outputs — this is the evidence that the import-time change was handled rather than papered over.

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest -q`

Expected: `130 passed`, no failures.

- [ ] **Step 8: Commit**

```bash
git add backend/app/main.py backend/tests/conftest.py backend/tests/test_health.py
git commit -m "refactor(main): let create_app take an explicit Settings"
```

---

### Task 4: Add `CORSMiddleware` and its behavioural tests

**Files:**
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_cors.py`
- Modify: `backend/.env.example`
- Modify: `docs/superpowers/specs/2026-07-21-cors-allowlist-design.md`

**Interfaces:**
- Consumes: `create_app(settings: Settings | None = None)` (Task 3) and `Settings.cors_allowed_origins` (Task 2).
- Produces: no new Python interface. The deliverable is observable HTTP behaviour.

**Do not assert a status code for a rejected simple request** — see Global Constraints. **Do not assert `allow-headers == "Content-Type"`** — it comes back as `Accept, Accept-Language, Content-Language, Content-Type`.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_cors.py`:

```python
"""Observable CORS behaviour, asserted through the real middleware stack.

Every test builds an app from a directly-constructed Settings, so no
environment variable or lru_cache clearing is involved.

Two assertions here are deliberately shaped and must not be "tidied":

1. A rejected *simple* request is asserted as ABSENCE OF THE HEADER, never
   as a status code. Starlette lets the response through with 200 and omits
   `access-control-allow-origin`; the browser is what blocks the read. An
   `assert response.status_code == 403` would fail against correct behaviour.

2. `allow-headers` is a substring check, never equality. starlette unions in
   the four CORS-safelisted headers regardless of configuration, so the
   response reads "Accept, Accept-Language, Content-Language, Content-Type".
"""
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

ALLOWED = "https://app.example"
SECOND = "https://other.example"
UNLISTED = "https://evil.example"


def build(origins):
    return create_app(
        Settings(anthropic_api_key="test-key-cors", cors_allowed_origins=origins)
    )


def preflight(client, path, origin, method="POST", headers="content-type"):
    request_headers = {"Origin": origin, "Access-Control-Request-Method": method}
    if headers is not None:
        request_headers["Access-Control-Request-Headers"] = headers
    return client.options(path, headers=request_headers)


# --- closed by default ----------------------------------------------------


def test_empty_allowlist_simple_request_gets_no_allow_origin_header():
    with TestClient(build([])) as client:
        response = client.get("/health", headers={"Origin": ALLOWED})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_empty_allowlist_rejects_preflight():
    with TestClient(build([])) as client:
        response = preflight(client, "/chat", ALLOWED)
    assert response.status_code == 400


# --- a configured origin --------------------------------------------------


def test_configured_origin_is_echoed_on_a_simple_request():
    with TestClient(build([ALLOWED])) as client:
        response = client.get("/health", headers={"Origin": ALLOWED})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED


def test_unlisted_origin_gets_no_header_but_still_returns_200():
    with TestClient(build([ALLOWED])) as client:
        response = client.get("/health", headers={"Origin": UNLISTED})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_each_configured_origin_is_echoed_alone():
    """Never the other origin, never both — the header is per-request."""
    with TestClient(build([ALLOWED, SECOND])) as client:
        first = client.get("/health", headers={"Origin": ALLOWED})
        second = client.get("/health", headers={"Origin": SECOND})
    assert first.headers["access-control-allow-origin"] == ALLOWED
    assert second.headers["access-control-allow-origin"] == SECOND


def test_configured_origin_passes_preflight_with_scoped_methods_and_headers():
    with TestClient(build([ALLOWED])) as client:
        response = preflight(client, "/chat", ALLOWED)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED
    assert response.headers["access-control-allow-methods"] == "GET, POST"
    # Substring, not equality — see this module's docstring.
    assert "content-type" in response.headers["access-control-allow-headers"].lower()


def test_unlisted_origin_fails_preflight():
    with TestClient(build([ALLOWED])) as client:
        response = preflight(client, "/chat", UNLISTED)
    assert response.status_code == 400


# --- scoping is real, not nominal ----------------------------------------


def test_preflight_for_an_unlisted_method_is_refused():
    """Guards allow_methods. Under a wildcard this would return 200."""
    with TestClient(build([ALLOWED])) as client:
        response = preflight(client, "/chat", ALLOWED, method="DELETE")
    assert response.status_code == 400


def test_preflight_for_an_unlisted_header_is_refused():
    """The ONLY assertion that catches a widening of allow_headers to ["*"].

    Under `allow_headers=["*"]` starlette does not echo a literal "*" — it
    mirrors back whatever the request asked for. So a check like
    `allow-headers != "*"` can never fire, and the substring check above
    passes under a wildcard too. This 400 is what distinguishes them.
    """
    with TestClient(build([ALLOWED])) as client:
        response = preflight(client, "/chat", ALLOWED, headers="x-custom")
    assert response.status_code == 400


# --- properties that must hold everywhere --------------------------------


@pytest.mark.parametrize("origins", [[], [ALLOWED]])
def test_requests_without_an_origin_header_are_untouched(origins):
    """Non-browser clients behave identically in every configuration."""
    with TestClient(build(origins)) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize("origins", [[], [ALLOWED]])
def test_credentials_are_never_allowed(origins):
    with TestClient(build(origins)) as client:
        simple = client.get("/health", headers={"Origin": ALLOWED})
        pre = preflight(client, "/chat", ALLOWED)
    assert "access-control-allow-credentials" not in simple.headers
    assert "access-control-allow-credentials" not in pre.headers


def test_middleware_covers_every_route_not_just_chat():
    """/health is exercised so "middleware, not per-route" is asserted."""
    with TestClient(build([ALLOWED])) as client:
        response = preflight(client, "/health", ALLOWED, method="GET")
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED


def test_non_canonical_configuration_matches_a_real_browser_origin():
    """Ties canonicalization to behaviour rather than leaving it a unit concern."""
    with TestClient(build(["https://App.Example/"])) as client:
        response = client.get("/health", headers={"Origin": ALLOWED})
    assert response.headers["access-control-allow-origin"] == ALLOWED
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cors.py -q`

Expected: FAIL. `test_configured_origin_is_echoed_on_a_simple_request` fails with `KeyError: 'access-control-allow-origin'`, and the preflight tests report `405` instead of `200`/`400` — there is no middleware yet.

- [ ] **Step 3: Add the middleware**

In `backend/app/main.py`, add the import:

```python
from fastapi.middleware.cors import CORSMiddleware
```

Then, in `create_app`, immediately after `application = FastAPI(lifespan=lifespan)`:

```python
    # Added unconditionally, including when the allowlist is empty — one
    # code path, and an empty list already means "permit nothing".
    #
    # The visible cost of that choice: a preflight to /chat now returns
    # 400 "Disallowed CORS origin" where it used to return 405. Both
    # refuse; the 400 says why.
    #
    # Nothing here is a wildcard. Credentials are off because the API has
    # no cookie or session auth. GET is allowed alongside POST because this
    # middleware also covers /health, and a browser status page polling it
    # cross-origin is a plausible real use.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cors.py -q`

Expected: PASS, 15 tests (13 functions, two of them parametrized with 2 cases each).

- [ ] **Step 5: Update `backend/.env.example`**

```
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-4-6

# Browser origins allowed to call this API directly, comma-separated.
# Example: CORS_ALLOWED_ORIGINS=https://app.example,http://localhost:3000
#
# Empty (the default) grants NO cross-origin browser access. Leave it empty
# unless a browser calls this API directly from another origin — the Next.js
# frontend proxies its requests, so it does not need this set.
#
# Each entry must be a full scheme://host[:port]. A bare host, a path, or "*"
# is rejected at startup rather than silently ignored.
CORS_ALLOWED_ORIGINS=
```

- [ ] **Step 6: Correct the spec's rejected-alternative paragraph**

In `docs/superpowers/specs/2026-07-21-cors-allowlist-design.md`, replace the paragraph beginning "Rejected: calling `get_settings()` inline" with:

```markdown
The parameter is a testing seam, not a way to defer validation. Resolving the
`None` default is eager, so the module-level `app = create_app()` calls
`get_settings()` at import either way — an earlier draft of this document
claimed otherwise and was wrong. Implementation added `backend/tests/conftest.py`
to set a dummy `ANTHROPIC_API_KEY`, without which importing `app.main` raises and
`tests/test_health.py` and `tests/test_chat_endpoint.py` fail to collect. Failing
at import with a clear message is more diagnostic than failing at lifespan, and
the app cannot serve without the key regardless.
```

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest -q`

Expected: `145 passed`, no failures, no new warnings.

- [ ] **Step 8: Commit**

```bash
git add backend/app/main.py backend/tests/test_cors.py backend/.env.example \
        docs/superpowers/specs/2026-07-21-cors-allowlist-design.md
git commit -m "feat(main): apply a configurable CORS allowlist as middleware"
```

---

## Acceptance Criteria Coverage

| Spec requirement | Where |
| --- | --- |
| Allowlist from environment configuration | Task 2 |
| Empty by default; no cross-origin access when unset | Task 2, Task 4 (`test_empty_allowlist_*`) |
| Credentials off | Task 4 (`test_credentials_are_never_allowed`) |
| Methods and headers scoped, not wildcarded | Task 4 (`DELETE` and `x-custom` preflight tests) |
| Applied as middleware, covering every route | Task 4 (`test_middleware_covers_every_route_not_just_chat`) |
| Origins canonicalized | Task 1; Task 4 (`test_non_canonical_configuration_matches_a_real_browser_origin`) |
| Invalid origins fail loudly | Task 1, Task 2 |
| `*` rejected | Task 1, Task 2 |
| Error message names the failing variable | Task 2 |
| `create_app` takes an optional `Settings` | Task 3 |
| Non-browser clients behave identically | Task 4 (`test_requests_without_an_origin_header_are_untouched`) |
| Rejected simple request asserted as header absence | Task 4, enforced by module docstring |

## Out of Scope

- Any change to the SSE event contract, `stream_chat`, or the frontend. The Next.js proxy remains the documented default path.
- Authentication, rate limiting, input guardrails — Phase 4.
- `allow_origin_regex` and wildcard subdomain patterns.
- `expose_headers`.
