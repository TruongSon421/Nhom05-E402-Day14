"""Simple redis-backed rate limiter (10 req/min/user by default)."""
import logging
import time
from collections import defaultdict, deque

import redis
from fastapi import HTTPException

from app.config import settings

_fallback_windows: dict[str, deque] = defaultdict(deque)
logger = logging.getLogger(__name__)
_redis_client = None
try:
    _redis_client = redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=0.2)
except Exception:
    _redis_client = None


def _check_with_memory(user_key: str) -> None:
    now = time.time()
    window = _fallback_windows[user_key]
    while window and window[0] < now - 60:
        window.popleft()
    if len(window) >= settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=429,
            detail=f"rate_limit exceeded: {settings.rate_limit_per_minute} req/min",
            headers={"Retry-After": "60"},
        )
    window.append(now)


def check_rate_limit(user_key: str) -> None:
    """Raise 429 when request rate exceeds configured per-minute limit."""
    if not user_key:
        user_key = "anonymous"

    if _redis_client is None:
        _check_with_memory(user_key)
        return

    key = f"rl:{user_key}"
    try:
        current = _redis_client.incr(key)
        if current == 1:
            _redis_client.expire(key, 60)
        if current > settings.rate_limit_per_minute:
            raise HTTPException(
                status_code=429,
                detail=f"rate_limit exceeded: {settings.rate_limit_per_minute} req/min",
                headers={"Retry-After": "60"},
            )
    except HTTPException:
        raise
    except Exception:
        _check_with_memory(user_key)


def redis_health_check() -> dict:
    """Return redis connectivity status for ops endpoints."""
    if _redis_client is None:
        return {"status": "degraded", "detail": "redis_client_not_initialized"}
    try:
        pong = _redis_client.ping()
        return {"status": "ok" if pong else "degraded", "detail": "pong" if pong else "no_pong"}
    except Exception as exc:
        logger.warning("Redis health check failed: %s", exc)
        return {"status": "degraded", "detail": str(exc)}
