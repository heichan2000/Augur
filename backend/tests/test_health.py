from fastapi.testclient import TestClient
from app.config import get_settings
from app.main import create_app


def test_health_returns_ok(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-health")
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
