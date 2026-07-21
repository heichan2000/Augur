import functools
from urllib.parse import urlsplit
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# How many provider round trips one turn may make before the agent loop
# gives up (see ``app.agent.run_turn``).
#
# The bound exists so a turn always terminates: every round is a paid API
# call, and a model that keeps requesting tools — repeating one call, or
# two tools feeding each other work — would otherwise loop until the
# client disconnects.
#
# Why 8: this value was picked by judgement, not measurement. It wants to
# sit above any legitimate turn (real answers resolve in one or two
# rounds, a handful at the outside) while still cutting off a runaway
# loop quickly. If a genuine turn ever needs more than 8 rounds, raise it
# — hitting this bound truncates the turn, so the number failing low is
# worse than it failing high.
#
# Import this rather than writing the number inline: tests script exactly
# this many rounds to exercise the bound, and a second copy would drift.
AGENT_MAX_STEPS = 8

# How long a stopped turn's persistence may take before we give up on it
# (see ``app.chat``, the client-disconnect handler).
#
# The write runs inside ``asyncio.shield`` because it happens while a
# cancellation is unwinding; the shield is what stops the write itself
# from being cancelled, and this bound is what stops the unwind from
# hanging on a store that never returns.
#
# Why 5 seconds: judgement, not measurement. The Phase-1 in-memory store
# never suspends, so this is dead weight today and exists for the Phase-2
# persistent store. It wants to sit well above a healthy round trip to a
# local Postgres (single-digit milliseconds) while still bounding the
# unwind at something a shutting-down worker can absorb. Revisit with a
# real store and real latency numbers.
PERSIST_ON_STOP_TIMEOUT_SECONDS = 5.0

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


class Settings(BaseSettings):
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-4-6"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@functools.lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Copy backend/.env.example to backend/.env and set it."
        ) from exc
