"""Shared types for travel multi-agent pipeline."""

from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    session_id: str
    question: str
    intent: str
    destination: str
    nights: int
    budget_vnd: int
    flight_result: dict
    hotel_result: dict
    itinerary_result: list[str]
    budget_note: str
    regulation_context: list[str]
    draft_answer: str
    answer: str
    requires_human_approval: bool
    human_approved: bool
    human_feedback: str
    blocked_by_guardrails: bool
    blocked_by: str
    judge_passed: bool
    judge_scores: dict
    redactions: list[str]
    trace: list[str]
    audit_id: str
    latency_ms: int
    alerts: list[str]
