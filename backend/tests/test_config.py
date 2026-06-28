import pytest
from app.config import get_settings


@pytest.fixture(autouse=True)
def isolate_from_dotenv(monkeypatch, tmp_path):
    """Change cwd to a temp dir so pydantic-settings cannot find backend/.env."""
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_loads_key_and_model_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-abc")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-5")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.anthropic_api_key == "test-key-abc"
    assert settings.anthropic_model == "claude-opus-4-5"


def test_model_defaults_when_unset(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-abc")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.anthropic_model == "claude-sonnet-4-6"


def test_missing_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    with pytest.raises(RuntimeError) as exc_info:
        get_settings()
    assert "ANTHROPIC_API_KEY" in str(exc_info.value)
