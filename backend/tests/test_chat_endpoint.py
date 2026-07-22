"""Tests for the POST /chat FastAPI route — TDD: written BEFORE the route exists.

Uses TestClient with app.dependency_overrides to inject a FAKE provider
(mirrors tests/test_chat.py's FakeProvider), a fresh InMemoryConversationStore
per test, and (for the tool test) a ToolRegistry holding a throwaway test tool.
"""
import json

from fastapi.testclient import TestClient

from app.config import get_settings
from app.conversation import InMemoryConversationStore, get_conversation_store
from app.main import create_app
from app.provider import TextDelta, ToolUseRequested, TurnComplete, get_provider
from app.tools import Tool, ToolRegistry, get_registry


# ---------------------------------------------------------------------------
# Fake provider (mirrors tests/test_chat.py's FakeProvider)
# ---------------------------------------------------------------------------


class FakeProvider:
    """Pops one scripted event-list (or raises) per stream_turn() call.

    Records the kwargs of every call for assertion.
    """

    def __init__(self, calls: list) -> None:
        self._calls = list(calls)
        self.received_calls: list[dict] = []

    async def stream_turn(self, *, messages, system=None, tools=None, max_tokens=2048):
        self.received_calls.append(
            {
                "messages": [dict(m) for m in messages],  # snapshot at call time
                "system": system,
                "tools": tools,
                "max_tokens": max_tokens,
            }
        )
        events = self._calls.pop(0)
        if isinstance(events, Exception):
            raise events
        for event in events:
            yield event


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _echo_handler(tool_input: dict) -> str:
    return f"echo:{tool_input}"


def _make_registry(handler=_echo_handler) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="echo",
            description="Echoes the input back.",
            input_schema={"type": "object", "properties": {}},
            handler=handler,
        )
    )
    return registry


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse a raw SSE response body into (event-name, data-dict) pairs."""
    parsed = []
    for chunk in body.split("\n\n"):
        if not chunk.strip():
            continue
        lines = chunk.split("\n")
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        event_name = lines[0][len("event: "):]
        data = json.loads(lines[1][len("data: "):])
        parsed.append((event_name, data))
    return parsed


def _make_app(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-chat-endpoint")
    get_settings.cache_clear()
    return create_app()


def _make_app_with_default_overrides(monkeypatch):
    """Build the app AND apply the standard fake overrides.

    FastAPI resolves the full dependency tree — including Depends(get_provider)
    — before Pydantic body validation can reject a malformed request. Every
    test that posts to /chat, even ones expecting a 422, MUST override
    get_provider (plus get_registry/get_conversation_store) so the real
    Anthropic client in app.provider.build_provider() is never constructed.
    """
    app = _make_app(monkeypatch)
    app.dependency_overrides[get_provider] = lambda: FakeProvider([])
    app.dependency_overrides[get_registry] = lambda: ToolRegistry()
    app.dependency_overrides[get_conversation_store] = lambda: InMemoryConversationStore()
    return app


# ---------------------------------------------------------------------------
# Behavior 1: Well-formed stream
# ---------------------------------------------------------------------------


def test_post_chat_returns_well_formed_sse_stream(monkeypatch):
    app = _make_app(monkeypatch)
    provider = FakeProvider(
        [[TextDelta("Hi"), TurnComplete(stop_reason="end_turn", input_tokens=1, output_tokens=1)]]
    )
    app.dependency_overrides[get_provider] = lambda: provider
    app.dependency_overrides[get_registry] = lambda: ToolRegistry()
    app.dependency_overrides[get_conversation_store] = lambda: InMemoryConversationStore()

    with TestClient(app) as client:
        response = client.post("/chat", json={"session_id": "s1", "message": "hello"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    assert events == [("token", {"text": "Hi"}), ("done", {"stop_reason": "end_turn"})]


# ---------------------------------------------------------------------------
# Behavior 2: Validation
# ---------------------------------------------------------------------------


def test_post_chat_missing_message_returns_422(monkeypatch):
    app = _make_app_with_default_overrides(monkeypatch)

    with TestClient(app) as client:
        response = client.post("/chat", json={"session_id": "s1"})

    assert response.status_code == 422


def test_post_chat_missing_session_id_returns_422(monkeypatch):
    app = _make_app_with_default_overrides(monkeypatch)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 422


def test_post_chat_validation_failure_never_builds_real_provider(monkeypatch):
    """Regression guard: body-validation failures must short-circuit before
    the real Anthropic client is ever constructed.

    get_provider is overridden (as in every other test) so the real client
    can't leak through the normal dependency path. On top of that, we spy on
    app.provider.build_provider directly — if anything in the request
    pipeline ever called it (e.g. a future regression that resolves
    dependencies in a different order, or calls build_provider() outside of
    get_provider), this test catches it even though the override would mask
    it from a plain status-code assertion alone.
    """
    import app.provider as provider_module

    app = _make_app_with_default_overrides(monkeypatch)

    build_calls: list[None] = []

    def _spy_build_provider():
        build_calls.append(None)
        raise AssertionError("build_provider() must not be called on a 422")

    monkeypatch.setattr(provider_module, "build_provider", _spy_build_provider)

    with TestClient(app) as client:
        response = client.post("/chat", json={"session_id": "s1"})

    assert response.status_code == 422
    assert len(build_calls) == 0


# ---------------------------------------------------------------------------
# Behavior 3: Tool path over HTTP
# ---------------------------------------------------------------------------


def test_post_chat_tool_path_yields_tool_use_then_token_then_done(monkeypatch):
    app = _make_app(monkeypatch)

    async def handler(tool_input: dict) -> str:
        return "echo-result"

    registry = _make_registry(handler=handler)
    provider = FakeProvider(
        [
            [
                ToolUseRequested(id="t1", name="echo", input={"v": 1}),
                TurnComplete(stop_reason="tool_use", input_tokens=1, output_tokens=1),
            ],
            [TextDelta("ok"), TurnComplete(stop_reason="end_turn", input_tokens=1, output_tokens=1)],
        ]
    )
    app.dependency_overrides[get_provider] = lambda: provider
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_conversation_store] = lambda: InMemoryConversationStore()

    with TestClient(app) as client:
        response = client.post("/chat", json={"session_id": "s1", "message": "use the tool"})

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert events == [
        ("tool_use", {"id": "t1", "name": "echo", "input": {"v": 1}}),
        ("token", {"text": "ok"}),
        ("done", {"stop_reason": "end_turn"}),
    ]


# ---------------------------------------------------------------------------
# Behavior 4: Multi-turn over HTTP shares the store dependency
# ---------------------------------------------------------------------------


def test_post_chat_multi_turn_shares_history_via_store_dependency(monkeypatch):
    app = _make_app(monkeypatch)
    store = InMemoryConversationStore()
    app.dependency_overrides[get_registry] = lambda: ToolRegistry()
    app.dependency_overrides[get_conversation_store] = lambda: store

    provider1 = FakeProvider(
        [[TextDelta("Hi"), TurnComplete(stop_reason="end_turn", input_tokens=1, output_tokens=1)]]
    )
    app.dependency_overrides[get_provider] = lambda: provider1

    with TestClient(app) as client:
        first = client.post("/chat", json={"session_id": "s1", "message": "hello"})
        assert first.status_code == 200

        provider2 = FakeProvider(
            [[TextDelta("again"), TurnComplete(stop_reason="end_turn", input_tokens=1, output_tokens=1)]]
        )
        app.dependency_overrides[get_provider] = lambda: provider2

        second = client.post("/chat", json={"session_id": "s1", "message": "second message"})
        assert second.status_code == 200

    second_call_messages = provider2.received_calls[0]["messages"]
    assert second_call_messages[:2] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
    ]
