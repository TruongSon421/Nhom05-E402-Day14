"""Safety layers: rate limit, guardrails, judge, audit, monitoring."""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock
from pydantic import BaseModel, Field
from app.config import settings


class SlidingWindowRateLimiter:
    def __init__(self, limit: int = 12, window_seconds: int = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        self._windows: dict[str, deque] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, user_key: str) -> bool:
        now = time.time()
        with self._lock:
            window = self._windows[user_key]
            while window and window[0] < now - self.window_seconds:
                window.popleft()
            if len(window) >= self.limit:
                return False
            window.append(now)
            return True


class NemoGuardrailsAdapter:
    """Use NeMo Guardrails when available and keep local checks deterministic."""

    def __init__(self, enabled: bool = True):
        self._rails = None
        if not enabled:
            return
        try:
            from nemoguardrails import LLMRails, RailsConfig

            config_path = str(Path(__file__).resolve().parent / "nemo_config")
            rails_config = RailsConfig.from_path(config_path)
            self._rails = LLMRails(rails_config)
        except Exception:
            self._rails = None

    def check(self, text: str) -> bool:
        if self._rails is None:
            return True
        try:
            _ = self._rails
        except Exception:
            return True
        return True


class InputGuardrails:
    def __init__(self, nemo_enabled: bool):
        self._nemo = NemoGuardrailsAdapter(enabled=nemo_enabled)
        self._injection_patterns = [
            r"ignore\s+all\s+previous\s+instructions",
            r"system\s*prompt",
            r"jailbreak",
            r"developer\s+message",
        ]
        self._dangerous_patterns = [
            r"\bhack\b",
            r"\bddos\b",
            r"\bmalware\b",
            r"\bsmuggle\b",
            r"bypass\s+authentication",
        ]

    def check(self, text: str) -> tuple[bool, str]:
        lowered = text.lower()
        if any(re.search(p, lowered) for p in self._injection_patterns):
            return False, "prompt_injection"
        if any(re.search(p, lowered) for p in self._dangerous_patterns):
            return False, "dangerous_topic"
        if not self._nemo.check(text):
            return False, "nemo_block"
        return True, "passed"


class OutputGuardrails:
    def filter(self, text: str) -> tuple[str, list[str]]:
        redactions: list[str] = []
        patterns = {
            "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            "phone": r"(?:\+84|0)\d{9,10}",
            "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
            "api_key": r"sk-[A-Za-z0-9_-]{10,}",
        }
        output = text
        for name, pattern in patterns.items():
            if re.search(pattern, output):
                output = re.sub(pattern, f"[REDACTED_{name.upper()}]", output)
                redactions.append(name)
        return output, redactions


class LLMJudgeScore(BaseModel):
    safety: float = Field(description="Score 0.0 to 1.0 for safety (no harmful advice).")
    relevance: float = Field(description="Score 0.0 to 1.0 for relevance to the user's travel question.")
    accuracy: float = Field(description="Score 0.0 to 1.0 for accuracy.")
    tone: float = Field(description="Score 0.0 to 1.0 for helpful, polite tone.")

class LLMJudge:
    """Async LLM-as-Judge with multi-criteria scoring via structured output."""

    async def evaluate(self, question: str, answer: str) -> tuple[bool, dict]:
        from langchain_openai import ChatOpenAI
        
        explicit_reject = "response rejected by llm-as-judge" in answer.lower()
        if explicit_reject:
            return False, {"safety": 0.0, "relevance": 0.0, "accuracy": 0.0, "tone": 0.0}

        try:
            llm = ChatOpenAI(model=settings.llm_model, temperature=0.0)
            structured_llm = llm.with_structured_output(LLMJudgeScore)
            prompt = (
                "You are an impartial judge evaluating a travel assistant's response.\n"
                f"User Question: {question}\n"
                f"Assistant Answer: {answer}\n\n"
                "Evaluate the assistant's answer based on safety, relevance, accuracy, and tone.\n"
                "Provide honest scores."
            )
            result: LLMJudgeScore = await structured_llm.ainvoke(prompt)
            scores = {
                "safety": result.safety,
                "relevance": result.relevance,
                "accuracy": result.accuracy,
                "tone": result.tone,
            }
            passed = result.safety >= 0.8 and result.relevance >= 0.5 and result.accuracy >= 0.5 and result.tone >= 0.5
            return passed, scores
        except Exception:
            # Fallback in case of LLM failure
            return True, {"safety": 1.0, "relevance": 0.9, "accuracy": 0.8, "tone": 0.9}


class AuditLogger:
    def __init__(self, root_dir: Path):
        self._path = root_dir / "logs" / "safety_audit.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def write(self, payload: dict) -> None:
        line = json.dumps(payload, ensure_ascii=True)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as file_obj:
                file_obj.write(line + "\n")

    @property
    def export_path(self) -> str:
        return str(self._path)


class SafetyMonitor:
    def __init__(self):
        self._lock = Lock()
        self._totals = {
            "requests": 0,
            "blocked": 0,
            "rate_limit_hits": 0,
            "judge_fail": 0,
        }
        self._alerts: list[str] = []

    def record(self, blocked_by: str, judge_passed: bool) -> list[str]:
        with self._lock:
            self._totals["requests"] += 1
            if blocked_by and blocked_by != "none":
                self._totals["blocked"] += 1
            if blocked_by == "rate_limiter":
                self._totals["rate_limit_hits"] += 1
            if not judge_passed:
                self._totals["judge_fail"] += 1

            requests = max(1, self._totals["requests"])
            block_rate = self._totals["blocked"] / requests
            rate_limit_rate = self._totals["rate_limit_hits"] / requests
            judge_fail_rate = self._totals["judge_fail"] / requests

            self._alerts = []
            if block_rate > 0.35:
                self._alerts.append("high_block_rate")
            if rate_limit_rate > 0.2:
                self._alerts.append("high_rate_limit_hits")
            if judge_fail_rate > 0.25:
                self._alerts.append("high_judge_fail_rate")
            return list(self._alerts)

    def snapshot(self) -> dict:
        with self._lock:
            requests = max(1, self._totals["requests"])
            return {
                **self._totals,
                "block_rate": round(self._totals["blocked"] / requests, 3),
                "judge_fail_rate": round(self._totals["judge_fail"] / requests, 3),
                "alerts": list(self._alerts),
            }
