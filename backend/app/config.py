import functools
import ipaddress
import re
from typing import Annotated
from urllib.parse import urlsplit

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

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

# A validated ASCII hostname's dot-separated labels, each non-empty and
# drawn only from letters, digits, and hyphens. ``^[a-z0-9.-]+$`` alone
# would accept "app.example.." — consecutive/leading/trailing dots produce
# empty labels, so labels must be checked individually.
_LABEL_RE = re.compile(r"^[a-z0-9-]+$")


def canonicalize_origin(raw: str) -> str:
    """Return the canonical ``scheme://host[:port]`` form of one configured origin.

    Lowercases the scheme and host, strips a trailing slash, and drops the
    scheme's default port. Raises ``ValueError`` — never returns a guess —
    for anything that is not a bare ``scheme://host[:port]``, including a
    host that Starlette's byte-for-byte comparison could never match: a
    malformed or unbracketed IPv6 literal, a non-hostname string like
    ``*`` or one containing whitespace, or a non-ASCII host a browser
    would send encoded.
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

    if ":" in host:
        # urlsplit strips the brackets from an IPv6 literal (and never
        # puts them back), so an unadorned host containing ':' is always
        # one. Validate it — a malformed literal must raise, not be
        # echoed back — and re-bracket it for the returned origin.
        try:
            ipaddress.ip_address(host)
        except ValueError as exc:
            raise ValueError(f"{raw!r}: {host!r} is not a valid IPv6 address.") from exc
        host = f"[{host}]"
    elif not host.isascii():
        # A browser only ever sends the punycode form in its Origin
        # header, so a Unicode host here would silently never match.
        message = (
            "non-ASCII hosts must be written in punycode — browsers send "
            "the encoded form."
        )
        try:
            suggestion = host.encode("idna").decode("ascii")
        except UnicodeError:
            raise ValueError(f"{raw!r}: {message}") from None
        raise ValueError(f"{raw!r}: {message} Try {scheme}://{suggestion}.")
    else:
        labels = host.split(".")
        if not all(label and _LABEL_RE.match(label) for label in labels):
            raise ValueError(
                f"{raw!r}: {host!r} is not a valid hostname — each "
                "dot-separated label must be non-empty and contain only "
                "letters, digits, and hyphens."
            )

    if port is None or port == _DEFAULT_PORTS[scheme]:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


class Settings(BaseSettings):
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-4-6"
    # Ceiling on a single provider response. A turn that hits it comes back
    # with stop_reason "max_tokens" and is surfaced to the user as truncated
    # (see app.provider.TRUNCATION_STOP_REASONS) rather than passed off as a
    # finished answer.
    anthropic_max_tokens: int = Field(default=2048, ge=1)

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
