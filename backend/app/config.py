import functools
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
