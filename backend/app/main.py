from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
    # Cached, so for the module-level `app` this is a no-op — import
    # already resolved (and validated) settings below. It still runs here
    # because `lifespan` always calls `get_settings()` directly, so for an
    # app built with an injected `Settings` this validates the
    # *environment's* configuration, not the one that app is actually
    # using (see the note in `create_app`).
    get_settings()
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    configure_logging()

    # Resolved eagerly, so the module-level `app = create_app()` below
    # validates configuration at import. The parameter exists so tests can
    # construct a Settings directly instead of setting environment
    # variables and clearing an lru_cache.
    #
    # This is a partial seam, not a full override: the `settings` passed
    # here configures only the CORS middleware built below. `lifespan`
    # above and the `/chat` route's `Depends(get_settings)` both still
    # resolve settings from the environment via `get_settings()`
    # regardless of what is passed in here. A `create_app(Settings(...))`
    # call with a non-default `anthropic_model`, for instance, would serve
    # `/chat` using the environment's model, not this one.
    if settings is None:
        settings = get_settings()

    application = FastAPI(lifespan=lifespan)

    # Added unconditionally, including when the allowlist is empty — one
    # code path, and an empty list already means "permit nothing".
    #
    # The visible cost of that choice: a preflight to /chat now returns
    # 400 "Disallowed CORS origin" where it used to return 405. Both
    # refuse; the 400 says why.
    #
    # Nothing here is a wildcard. Credentials are off because the API has
    # no cookie or session auth. GET is allowed alongside POST because this
    # middleware also covers /health, and a browser status page polling it
    # cross-origin is a plausible real use.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

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
