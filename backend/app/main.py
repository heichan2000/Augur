from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.chat import stream_chat
from app.config import Settings, get_settings
from app.conversation import get_conversation_store
from app.observability import configure_logging
from app.provider import get_provider
from app.tools import get_registry


class ChatRequest(BaseModel):
    session_id: str
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings()  # fail fast if ANTHROPIC_API_KEY is missing
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    configure_logging()

    # Resolved eagerly, so the module-level `app = create_app()` below
    # validates configuration at import. The parameter exists so tests can
    # construct a Settings directly instead of setting environment
    # variables and clearing an lru_cache.
    if settings is None:
        settings = get_settings()

    application = FastAPI(lifespan=lifespan)

    @application.get("/health")
    async def health():
        return {"status": "ok"}

    @application.post("/chat")
    async def chat(
        req: ChatRequest,
        provider=Depends(get_provider),
        registry=Depends(get_registry),
        store=Depends(get_conversation_store),
        settings=Depends(get_settings),
    ):
        return StreamingResponse(
            stream_chat(
                provider=provider,
                registry=registry,
                store=store,
                session_id=req.session_id,
                message=req.message,
                model=settings.anthropic_model,
            ),
            media_type="text/event-stream",
        )

    return application


app = create_app()
