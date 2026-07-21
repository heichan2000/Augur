import pytest
from app.config import Settings, canonicalize_origin, get_settings


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
        ("https://my-app.example", "https://my-app.example"),
        ("https://[::1]:8080", "https://[::1]:8080"),
        ("https://[2001:db8::1]", "https://[2001:db8::1]"),
        ("https://[::1]:443", "https://[::1]"),
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
        "https://*",                 # bare host is "*" — never matches, and inert
        "https://a b.example",       # embedded whitespace
        "https://app.example..",     # trailing empty label
        "https://.app.example",      # leading empty label
        "https://[not-an-ipv6]",     # malformed IPv6 literal
    ],
)
def test_canonicalize_rejects(raw):
    with pytest.raises(ValueError):
        canonicalize_origin(raw)


def test_canonicalize_rejects_non_ascii_host_with_punycode_suggestion():
    with pytest.raises(ValueError) as exc_info:
        canonicalize_origin("https://Bücher.example")
    assert "xn--" in str(exc_info.value)


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


def test_blank_entries_are_tolerated_but_malformed_ones_are_not(monkeypatch):
    """Pins the distinction: a trailing comma is tolerated, a typo is not.

    Nothing should ever collapse "tolerate a trailing comma" into "swallow
    a typo" — that would be the exact silent failure this module exists
    to prevent.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-abc")

    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://a.example,notaurl,")
    get_settings.cache_clear()
    with pytest.raises(RuntimeError):
        get_settings()

    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://a.example,,")
    get_settings.cache_clear()
    assert get_settings().cors_allowed_origins == ["https://a.example"]
