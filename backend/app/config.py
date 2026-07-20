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
