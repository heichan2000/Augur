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
