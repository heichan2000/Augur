from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings()  # fail fast if ANTHROPIC_API_KEY is missing
    yield


def create_app() -> FastAPI:
    application = FastAPI(lifespan=lifespan)

    @application.get("/health")
    async def health():
        return {"status": "ok"}

    return application


app = create_app()
