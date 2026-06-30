from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.chat import stream_chat
from app.config import get_settings
from app.conversation import get_conversation_store
from app.provider import get_provider
from app.tools import get_registry


class ChatRequest(BaseModel):
    session_id: str
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings()  # fail fast if ANTHROPIC_API_KEY is missing
    yield


def create_app() -> FastAPI:
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
    ):
        return StreamingResponse(
            stream_chat(
                provider=provider,
                registry=registry,
                store=store,
                session_id=req.session_id,
                message=req.message,
            ),
            media_type="text/event-stream",
        )

    return application


app = create_app()
