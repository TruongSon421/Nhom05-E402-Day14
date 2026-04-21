"""Daily budget guard; returns HTTP 402 when exceeded."""
import time
from threading import Lock

from fastapi import HTTPException

from app.config import settings

_lock = Lock()
_daily_cost = 0.0
_cost_reset_day = time.strftime("%Y-%m-%d")


def _rotate_day_if_needed() -> None:
    global _daily_cost, _cost_reset_day
    today = time.strftime("%Y-%m-%d")
    if today != _cost_reset_day:
        _daily_cost = 0.0
        _cost_reset_day = today


def check_and_record_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate usage cost and enforce daily budget.

    Pricing model:
      - input:  $0.00015 / 1k tokens
      - output: $0.00060 / 1k tokens
    """
    global _daily_cost
    with _lock:
        _rotate_day_if_needed()
        if _daily_cost >= settings.daily_budget_usd:
            raise HTTPException(status_code=402, detail="Daily budget exceeded")

        cost = (input_tokens / 1000) * 0.00015 + (output_tokens / 1000) * 0.0006
        _daily_cost += cost
        if _daily_cost > settings.daily_budget_usd:
            raise HTTPException(status_code=402, detail="Daily budget exceeded")
        return cost


def budget_status() -> dict:
    with _lock:
        _rotate_day_if_needed()
        used_pct = round((_daily_cost / settings.daily_budget_usd) * 100, 1)
        return {
            "daily_cost_usd": round(_daily_cost, 6),
            "daily_budget_usd": settings.daily_budget_usd,
            "budget_used_pct": used_pct,
        }
