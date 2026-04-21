"""Travel LangGraph ReAct agent builder."""

from __future__ import annotations
import unicodedata

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.config import settings
from app.agents.air_rules_rag import regulation_rag
from app.agents.mock_data import MOCK_FLIGHTS, MOCK_HOTELS, MOCK_ITINERARY
from app.agents.types import AgentState


def normalize_location(loc: str) -> str:
    loc = loc.lower().strip()
    loc = unicodedata.normalize('NFKD', loc).encode('ASCII', 'ignore').decode('utf-8')
    if "ho chi minh" in loc or "hcm" in loc or "sai gon" in loc:
        return "hcm"
    if "ha noi" in loc or "hn" in loc or "hanoi" in loc:
        return "hanoi"
    if "da nang" in loc or "dn" in loc or "danang" in loc:
        return "danang"
    return loc.replace(" ", "")

@tool
def search_flight(origin: str, destination: str) -> dict:
    """Search for flights based on origin and destination. Ex: ho chi minh to ha noi."""
    route = f"{normalize_location(origin)}-{normalize_location(destination)}"
    # Mock fallback for reverse route if not found (optional, but keep it simple)
    options = [f for f in MOCK_FLIGHTS if f["route"] == route]
    if not options:
        reverse_route = f"{normalize_location(destination)}-{normalize_location(origin)}"
        options = [f for f in MOCK_FLIGHTS if f["route"] == reverse_route]
    return min(options, key=lambda i: i["price_vnd"]) if options else {"error": "No flights found."}

@tool
def search_hotel(city: str) -> dict:
    """Search for hotels in a specific city."""
    norm_city = normalize_location(city)
    options = [h for h in MOCK_HOTELS if h["city"] == norm_city]
    return min(options, key=lambda i: i["price_vnd_per_night"]) if options else {"error": "No hotels found."}

@tool
def search_itinerary(city: str, nights: int) -> list[str]:
    """Get a suggested travel itinerary for the city."""
    norm_city = normalize_location(city)
    itinerary = MOCK_ITINERARY.get(norm_city, ["Day 1: Explore the city"])
    return itinerary[: nights + 1]

@tool
def search_regulations(query: str) -> list[str]:
    """Search airline regulation knowledge base (luggage, ID card, rules)."""
    return regulation_rag.search(query, limit=3)

tools = [search_flight, search_hotel, search_itinerary, search_regulations]

sys_prompt = """You are a helpful and polite travel planning and airline regulation assistant.
Use the provided tools to fetch information about flights, hotels, itineraries, and airline regulations.
When mock data tools return an 'error' or empty result, politely inform the user that you couldn't find matches.
Always try to calculate the estimated budget if flight and hotel prices are retrieved.
DO NOT fabricate information. If a tool doesn't provide it, state that you don't know or don't have that data.

LANGUAGE POLICY:
- Only support two output languages: Vietnamese and English.
- If the user writes in Vietnamese, respond in Vietnamese.
- If the user writes in English, respond in English.
- If the user's language is mixed or unclear, default to Vietnamese.
- Do not switch to a third language.

CRITICAL INSTRUCTION: Format your final response into a clean, consistent Markdown structure showing Travel Itinerary, Flights, Hotels and total estimated budget if applicable. If it is about regulations, format it as a distinct list of rules. Keep the language natural and helpful.
"""

async def agent_node(state: AgentState):
    llm = ChatOpenAI(model=settings.llm_model, temperature=0.2)
    bound_llm = llm.bind_tools(tools)
    messages = state.get("messages", [])
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=sys_prompt)] + messages
    
    response = await bound_llm.ainvoke(messages)
    trace = list(state.get("trace", []))
    trace.append("llm:agent_step")
    return {"messages": [response], "trace": trace}

def should_continue(state: AgentState) -> str:
    messages = state.get("messages", [])
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "hitl"

async def hitl_node(state: AgentState) -> AgentState:
    approved = state.get("human_approved", False)
    feedback = state.get("human_feedback", "").strip()
    trace = list(state.get("trace", []))
    if not approved:
        trace.append("hitl:waiting_approval")
        return {"requires_human_approval": True, "trace": trace}
    
    # In standard ReAct, the final message is the text from the LLM. 
    messages = state.get("messages", [])
    last_message = messages[-1]
    answer = getattr(last_message, "content", "") if messages else ""

    if feedback:
        answer = f"{answer}\n\n[Human feedback applied] {feedback}"
        
    trace.append("hitl:approved")
    return {"requires_human_approval": True, "answer": answer, "trace": trace}

def build_multi_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("hitl", hitl_node)
    
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "hitl": "hitl"})
    graph.add_edge("tools", "agent")
    graph.add_edge("hitl", END)
    
    return graph.compile()
