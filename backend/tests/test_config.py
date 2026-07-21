import pytest
from app.config import canonicalize_origin, get_settings


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
