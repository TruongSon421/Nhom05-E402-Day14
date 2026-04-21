"""Telemetry layer: audit logging and monitoring integration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.agents.safety import AuditLogger, SafetyMonitor


class TelemetryService:
    def __init__(self):
        root_dir = Path(__file__).resolve().parent.parent.parent
        self._audit = AuditLogger(root_dir=root_dir)
        self._monitor = SafetyMonitor()

    def get_metrics(self) -> dict:
        return {"monitoring": self._monitor.snapshot(), "audit_log_path": self._audit.export_path}

    def create_audit_id(self) -> str:
        return str(uuid4())

    def record_interaction(
        self,
        *,
        audit_id: str,
        user_key: str,
        question: str,
        answer: str,
        blocked_by: str,
        judge_passed: bool,
        judge_scores: dict,
        redactions: list[str],
        latency_ms: int,
        trace: list[str],
    ) -> list[str]:
        alerts = self._monitor.record(blocked_by=blocked_by, judge_passed=judge_passed)
        payload = {
            "audit_id": audit_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_key": user_key,
            "input": question,
            "output": answer,
            "blocked_by": blocked_by,
            "judge_passed": judge_passed,
            "judge_scores": judge_scores,
            "redactions": redactions,
            "latency_ms": latency_ms,
            "trace": trace,
            "alerts": alerts,
        }
        self._audit.write(payload)
        return alerts
