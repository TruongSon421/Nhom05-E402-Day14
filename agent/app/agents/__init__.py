"""Public API for travel multi-agent package."""

from app.agents.service import AgentService

_service = AgentService()


async def stream_multi_agent(
    session_id: str,
    question: str,
    guardrails_enabled: bool = True,
    human_approved: bool = False,
    human_feedback: str = "",
    user_key: str = "anonymous",
):
    async for chunk in _service.stream_run(
        session_id=session_id,
        question=question,
        guardrails_enabled=guardrails_enabled,
        human_approved=human_approved,
        human_feedback=human_feedback,
        user_key=user_key,
    ):
        yield chunk


def get_safety_metrics() -> dict:
    return _service.get_safety_metrics()


def get_dependency_checks() -> dict:
    return _service.get_dependency_checks()


def warmup_regulation_store() -> None:
    _service.warmup_regulation_store()
