from fastapi.testclient import TestClient
from app.config import Settings, get_settings
from app.main import create_app


def test_health_returns_ok(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-health")
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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
