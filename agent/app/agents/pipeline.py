"""Pipeline control flow for multi-layer safety and agent execution."""

from __future__ import annotations

import logging
import time
import json
import redis
from langchain_core.messages import HumanMessage, messages_from_dict, messages_to_dict
from app.config import settings

# Langfuse observability (optional — disabled if keys not set)
_langfuse_enabled = bool(settings.langfuse_public_key and settings.langfuse_secret_key)
if _langfuse_enabled:
    from langfuse.callback import CallbackHandler as LangfuseCallbackHandler

from app.agents.safety import (
    InputGuardrails,
    LLMJudge,
    OutputGuardrails,
    SlidingWindowRateLimiter,
)
from app.agents.hr_graph import build_multi_agent_graph
from app.agents.types import AgentState

logger = logging.getLogger(__name__)


class PipelineService:
    def __init__(self):
        self._graph = build_multi_agent_graph()
        self._pipeline_rate_limiter = SlidingWindowRateLimiter(limit=10, window_seconds=60)
        self._input_guardrails = InputGuardrails(nemo_enabled=True)
        self._output_guardrails = OutputGuardrails()
        self._judge = LLMJudge()
        self._redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)

    def _get_history(self, session_id: str) -> list:
        if not session_id:
            return []
        data = self._redis.get(f"chat_history:{session_id}")
        if data:
            return messages_from_dict(json.loads(data))
        return []

    def _save_history(self, session_id: str, messages: list):
        if not session_id or not messages:
            return
        data = json.dumps(messages_to_dict(messages))
        self._redis.setex(f"chat_history:{session_id}", 3600, data) # 1 hour TTL

    async def stream_execute(
        self,
        *,
        session_id: str = "",
        question: str,
        guardrails_enabled: bool = True,
        human_approved: bool = False,
        human_feedback: str = "",
        user_key: str = "anonymous",
    ):
        start = time.time()
        trace = [f"pipeline:start:user_key={user_key}"]
        blocked_by = "none"
        judge_passed = True
        judge_scores = {"safety": 0.0, "relevance": 0.0, "accuracy": 0.0, "tone": 0.0}
        redactions: list[str] = []
        answer = ""

        if not self._pipeline_rate_limiter.allow(user_key):
            blocked_by = "rate_limiter"
            trace.append("rate_limiter:blocked")
            answer = "Too many requests. Please retry in 1 minute."
            yield json.dumps({"event": "error", "data": answer}) + "\n\n"
            return

        trace.append("rate_limiter:passed")
        
        checker = self._input_guardrails if guardrails_enabled else InputGuardrails(nemo_enabled=False)
        allowed, reason = checker.check(question)
        if not allowed:
            blocked_by = f"input_guardrails:{reason}"
            trace.append(f"input_guardrails:blocked:{reason}")
            answer = "Request blocked by input guardrails."
            yield json.dumps({"event": "error", "data": answer}) + "\n\n"
            return
            
        trace.append("input_guardrails:passed")

        state: AgentState = {
            "session_id": session_id,
            "question": question,
            "human_approved": human_approved,
            "human_feedback": human_feedback,
            "trace": trace,
            "messages": self._get_history(session_id) + [HumanMessage(content=question)]
        }
        
        final_messages = []
        try:
            # Build run config với Langfuse callback nếu keys được set
            run_config: dict = {}
            if _langfuse_enabled:
                langfuse_handler = LangfuseCallbackHandler(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                    session_id=session_id or None,
                    user_id=user_key,
                    trace_name="multi-agent-pipeline",
                    metadata={
                        "guardrails_enabled": guardrails_enabled,
                        "human_approved": human_approved,
                    },
                )
                run_config = {"callbacks": [langfuse_handler]}

            async for event in self._graph.astream_events(state, version="v2", config=run_config):
                kind = event["event"]
                name = event.get("name", "")
                
                if kind == "on_chat_model_start":
                    if name == "ChatOpenAI":
                        answer = "" # reset for each LLM turnaround so we only catch the final output
                elif kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if getattr(chunk, "content", None) and isinstance(chunk.content, str):
                        answer += chunk.content
                        yield json.dumps({"event": "token", "data": chunk.content}) + "\n\n"
                elif kind == "on_tool_start":
                    yield json.dumps({"event": "metadata", "data": {"status": f"Đang tra cứu tài liệu HR: {name}..."}}) + "\n\n"
                elif kind == "on_chain_end":
                    if event["name"] == "agent" and "messages" in event["data"]["output"]:
                        # Capture updated messages from agent node
                        final_messages = event["data"]["output"]["messages"]
                        
            if final_messages:
                self._save_history(session_id, final_messages)
                
        except Exception as exc:
            blocked_by = "pipeline_graph_error"
            answer = "Pipeline execution failed. Please retry."
            trace.append(f"graph:failed:{type(exc).__name__}")
            logger.exception("pipeline_graph_invoke_failed user_key=%s", user_key)
            yield json.dumps({"event": "error", "data": answer}) + "\n\n"
            return

        filtered_answer, redactions = self._output_guardrails.filter(answer)
        answer = filtered_answer
        trace.append("output_guardrails:passed")

        yield json.dumps({"event": "metadata", "data": {"status": "Đang đánh giá đáp án (LLM Judge)..."}}) + "\n\n"
        judge_passed, judge_scores = await self._judge.evaluate(question, answer)
        if not judge_passed:
            blocked_by = "llm_as_judge"
            trace.append("llm_as_judge:blocked")
        else:
            trace.append("llm_as_judge:passed")

        latency_ms = int((time.time() - start) * 1000)
        
        final_payload = {
            "answer": answer,
            "blocked_by": blocked_by,
            "judge_passed": judge_passed,
            "judge_scores": judge_scores,
            "latency_ms": latency_ms,
            "trace": trace
        }
        yield json.dumps({"event": "message_end", "data": final_payload}) + "\n\n"
