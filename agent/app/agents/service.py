"""Orchestration service for safety pipeline and travel graph."""

from __future__ import annotations

import json
import logging
import time

from app.agents.hr_rag import hr_rag
from app.agents.audit_monitoring import TelemetryService
from app.agents.pipeline import PipelineService
from app.agents.types import AgentState

logger = logging.getLogger(__name__)


class AgentService:
    def __init__(self):
        self._pipeline = PipelineService()
        self._telemetry = TelemetryService()

    def get_safety_metrics(self) -> dict:
        return self._telemetry.get_metrics()

    def warmup_regulation_store(self) -> None:
        try:
            hr_rag.initialize()
            logger.info("Warmup HR knowledge base completed")
        except Exception as exc:
            logger.warning("Warmup HR knowledge base failed: %s", exc)

    def get_dependency_checks(self) -> dict:
        return {
            "chromadb": hr_rag.health_check(),
        }

    async def stream_run(
        self,
        session_id: str,
        question: str,
        guardrails_enabled: bool = True,
        human_approved: bool = False,
        human_feedback: str = "",
        user_key: str = "anonymous",
    ):
        audit_id = self._telemetry.create_audit_id()
        # yield initial event
        yield json.dumps({"event": "metadata", "data": {"status": "Đang xử lý câu hỏi HR của bạn..."}}) + "\n\n"
        
        final_payload = None
        alerts = []
        started = time.time()

        async for chunk in self._pipeline.stream_execute(
            session_id=session_id,
            question=question,
            guardrails_enabled=guardrails_enabled,
            human_approved=human_approved,
            human_feedback=human_feedback,
            user_key=user_key,
        ):
            if '"event": "message_end"' in chunk:
                try:
                    payload_str = chunk.split('"data": ')[1].strip()[:-2]
                    final_payload = json.loads(payload_str)
                except Exception:
                    pass
            yield chunk

        if final_payload:
            trace = list(final_payload.get("trace", []))
            trace.append("audit_monitoring:recorded")
            alerts = self._telemetry.record_interaction(
                audit_id=audit_id,
                user_key=user_key,
                question=question,
                answer=final_payload.get("answer", ""),
                blocked_by=final_payload.get("blocked_by", "none"),
                judge_passed=final_payload.get("judge_passed", True),
                judge_scores=final_payload.get("judge_scores", {}),
                redactions=final_payload.get("redactions", []),
                latency_ms=final_payload.get("latency_ms", 0),
                trace=trace,
            )
            final_payload["audit_id"] = audit_id
            final_payload["alerts"] = alerts
            final_payload["requires_human_approval"] = True
            
            # send the wrapper final event
            yield json.dumps({"event": "complete", "data": final_payload}) + "\n\n"

