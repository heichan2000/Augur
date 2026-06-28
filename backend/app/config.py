import functools
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


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
