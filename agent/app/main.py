"""Production AI Agent entrypoint."""
import json
import logging
from pathlib import Path
import signal
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from app.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    hash_password,
    require_admin,
    verify_password,
)
from app.config import settings
from app.cost_guard import budget_status, check_and_record_cost
from app.database import Base, engine, get_db
from app.models import User
from app.multi_agent import get_dependency_checks, get_safety_metrics, stream_multi_agent, warmup_regulation_store
from app.rate_limiter import check_rate_limit, redis_health_check

from utils.mock_llm import ask as llm_ask

# ─────────────────────────────────────────────────────────
# Logging — JSON structured
# ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format='{"ts":"%(asctime)s","lvl":"%(levelname)s","msg":"%(message)s"}',
)
logger = logging.getLogger(__name__)

START_TIME = time.time()
_is_ready = False
_request_count = 0
_error_count = 0
FRONTEND_DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

# ─────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _is_ready
    logger.info(json.dumps({
        "event": "startup",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }))
    time.sleep(0.1)  # simulate init
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    user_columns = {column["name"] for column in inspector.get_columns("users")} if inspector.has_table("users") else set()
    if "role" not in user_columns:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE users "
                    "ADD COLUMN role VARCHAR(20) NOT NULL DEFAULT 'user'"
                )
            )
    with Session(engine) as db:
        existing_admins = db.query(User).filter(User.role == "admin").all()
        if len(existing_admins) > 1:
            raise ValueError("Only one admin account is allowed")
        if not existing_admins:
            admin = User(
                email=settings.admin_email,
                full_name="System Administrator",
                role="admin",
                hashed_password=hash_password(settings.admin_password),
            )
            db.add(admin)
            db.commit()
    warmup_regulation_store()
    _is_ready = True
    logger.info(json.dumps({"event": "ready", "rate_limit": settings.rate_limit_per_minute}))

    yield

    _is_ready = False
    logger.info(json.dumps({"event": "shutdown"}))

# ─────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

if FRONTEND_DIST_DIR.exists():
    app.mount("/frontend-static", StaticFiles(directory=str(FRONTEND_DIST_DIR)), name="frontend-static")

@app.middleware("http")
async def request_middleware(request: Request, call_next):
    global _request_count, _error_count
    start = time.time()
    _request_count += 1
    try:
        response: Response = await call_next(request)
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        if "server" in response.headers:
            del response.headers["server"]
        duration = round((time.time() - start) * 1000, 1)
        logger.info(json.dumps({
            "event": "request",
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "ms": duration,
        }))
        return response
    except Exception as e:
        _error_count += 1
        raise

# ─────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          description="Your question for the agent")

class AskResponse(BaseModel):
    question: str
    answer: str
    model: str
    timestamp: str


class AskMultiAgentRequest(BaseModel):
    session_id: str = Field(
        default="",
        description="Session ID for tracking conversation history."
    )
    question: str = Field(..., min_length=1, max_length=2000)
    human_approved: bool = Field(
        default=False,
        description="HITL approval. Set true to release final answer.",
    )
    human_feedback: str = Field(
        default="",
        max_length=500,
        description="Optional reviewer feedback applied after approval.",
    )


class AskMultiAgentResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    requires_human_approval: bool
    human_approved: bool
    blocked_by_guardrails: bool
    blocked_by: str
    judge_passed: bool
    judge_scores: dict
    redactions: list[str]
    audit_id: str
    latency_ms: int
    alerts: list[str]
    trace: list[str]
    timestamp: str


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=255)
    full_name: str = Field(..., min_length=2, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)


class AuthResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserProfileResponse(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    created_at: str


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=10)


class AdminUserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    created_at: str

# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "endpoints": {
            "ask": "POST /ask (requires Bearer JWT)",
            "ask_multi_agent_stream": "POST /ask-multi-agent/stream — HR chatbot (requires Bearer JWT)",
            "register": "POST /auth/register",
            "login": "POST /auth/login",
            "refresh": "POST /auth/refresh",
            "me": "GET /auth/me (requires Bearer JWT)",
            "admin_users": "GET /admin/users (admin only)",
            "ui": "GET /ui",
            "health": "GET /health",
            "ready": "GET /ready",
        },
    }


@app.get("/ui", tags=["UI"])
def ui_page():
    """TypeScript UX demo for the multi-agent pipeline."""
    frontend_index = FRONTEND_DIST_DIR / "index.html"
    if frontend_index.exists():
        return FileResponse(frontend_index)
    raise HTTPException(status_code=404, detail="UI file not found")


@app.post("/auth/register", response_model=AuthResponse, tags=["Auth"], status_code=status.HTTP_201_CREATED)
def register(
    body: RegisterRequest,
    db: Session = Depends(get_db),
):
    existing = db.query(User).filter(User.email == body.email.lower()).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=body.email.lower(),
        full_name=body.full_name.strip(),
        role="user",
        hashed_password=hash_password(body.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return AuthResponse(
        access_token=create_access_token(user),
        refresh_token=create_refresh_token(user),
    )


@app.post("/auth/login", response_model=AuthResponse, tags=["Auth"])
def login(
    body: LoginRequest,
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.email == body.email.lower()).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return AuthResponse(
        access_token=create_access_token(user),
        refresh_token=create_refresh_token(user),
    )


@app.post("/auth/refresh", response_model=AuthResponse, tags=["Auth"])
def refresh_token(body: RefreshRequest, db: Session = Depends(get_db)):
    payload = decode_token(body.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    try:
        user_id = int(payload.get("sub", "0"))
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid refresh token subject")
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return AuthResponse(
        access_token=create_access_token(user),
        refresh_token=create_refresh_token(user),
    )


@app.get("/auth/me", response_model=UserProfileResponse, tags=["Auth"])
def me(current_user: User = Depends(get_current_user)):
    return UserProfileResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        created_at=current_user.created_at.isoformat(),
    )


@app.get("/admin/users", response_model=list[AdminUserResponse], tags=["Admin"])
def admin_list_users(
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [
        AdminUserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            created_at=user.created_at.isoformat(),
        )
        for user in users
    ]


@app.delete("/admin/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Admin"])
def admin_delete_user(
    user_id: int,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.role == "admin":
        raise HTTPException(status_code=400, detail="Admin account cannot be deleted")
    db.delete(user)
    db.commit()


@app.post("/ask", response_model=AskResponse, tags=["Agent"])
async def ask_agent(
    body: AskRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """
    Send a question to the AI agent.

    **Authentication:** Include header `Authorization: Bearer <access-token>`
    """
    check_rate_limit(f"user-{current_user.id}")

    # Budget check
    input_tokens = len(body.question.split()) * 2
    check_and_record_cost(input_tokens, 0)

    logger.info(json.dumps({
        "event": "agent_call",
        "q_len": len(body.question),
        "client": str(request.client.host) if request.client else "unknown",
    }))

    answer = llm_ask(body.question)

    output_tokens = len(answer.split()) * 2
    check_and_record_cost(0, output_tokens)

    return AskResponse(
        question=body.question,
        answer=answer,
        model=settings.llm_model,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/ask-multi-agent/stream", tags=["Agent"])
async def ask_multi_agent_stream(
    body: AskMultiAgentRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """LangGraph multi-agent flow over SSE (Server-Sent Events)."""
    check_rate_limit(f"ma-user-{current_user.id}")

    input_tokens = len(body.question.split()) * 2
    check_and_record_cost(input_tokens, 0)

    logger.info(json.dumps({
        "event": "multi_agent_stream_call",
        "q_len": len(body.question),
        "client": str(request.client.host) if request.client else "unknown",
        "human_approved": body.human_approved,
    }))

    async def generate():
        async for chunk in stream_multi_agent(
            session_id=body.session_id,
            question=body.question,
            guardrails_enabled=settings.enable_nemo_guardrails,
            human_approved=body.human_approved,
            human_feedback=body.human_feedback,
            user_key=f"ma-user-{current_user.id}",
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health", tags=["Operations"])
def health():
    """Liveness probe. Platform restarts container if this fails."""
    status = "ok"
    llm_check = {"status": "mock" if not settings.openai_api_key else "openai"}
    dependency_checks = {
        "redis": redis_health_check(),
        **get_dependency_checks(),
    }
    if any(item.get("status") != "ok" for item in dependency_checks.values()):
        status = "degraded"
    checks = {
        "llm": llm_check,
        "dependencies": dependency_checks,
    }
    return {
        "status": status,
        "version": settings.app_version,
        "environment": settings.environment,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "total_requests": _request_count,
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/ready", tags=["Operations"])
def ready():
    """Readiness probe. Load balancer stops routing here if not ready."""
    if not _is_ready:
        raise HTTPException(503, "Not ready")
    redis_check = redis_health_check()
    chromadb_check = get_dependency_checks().get("chromadb", {"status": "degraded", "detail": "missing"})
    if redis_check.get("status") != "ok" or chromadb_check.get("status") != "ok":
        raise HTTPException(
            503,
            detail={
                "ready": False,
                "dependencies": {
                    "redis": redis_check,
                    "chromadb": chromadb_check,
                },
            },
        )
    return {"ready": True}


@app.get("/metrics", tags=["Operations"])
def metrics(_admin: User = Depends(require_admin)):
    """Basic metrics (protected)."""
    budget = budget_status()
    safety = get_safety_metrics()
    return {
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "total_requests": _request_count,
        "error_count": _error_count,
        **budget,
        **safety,
    }


# ─────────────────────────────────────────────────────────
# Graceful Shutdown
# ─────────────────────────────────────────────────────────
def _handle_signal(signum, _frame):
    logger.info(json.dumps({"event": "signal", "signum": signum}))

signal.signal(signal.SIGTERM, _handle_signal)


if __name__ == "__main__":
    logger.info(f"Starting {settings.app_name} on {settings.host}:{settings.port}")
    logger.info("JWT authentication enabled")
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        timeout_graceful_shutdown=30,
    )
