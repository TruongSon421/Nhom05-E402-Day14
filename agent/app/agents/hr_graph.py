"""HR LangGraph ReAct agent builder."""

from __future__ import annotations

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.config import settings
from app.agents.hr_rag import hr_rag
from app.agents.types import AgentState


# ─────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────

@tool
def search_hr_policy(query: str) -> list[str]:
    """Search HR policies: benefits, leave, working hours, company rules, employee handbook."""
    return hr_rag.search(query, limit=4)


@tool
def search_recruitment_process(query: str) -> list[str]:
    """Search recruitment & onboarding information: application process, interview rounds, probation period."""
    return hr_rag.search(query, limit=4)


@tool
def search_faq(query: str) -> list[str]:
    """Search HR FAQ: common questions about payroll, insurance, leave, training, etc."""
    return hr_rag.search(query, limit=4)


@tool
def search_performance_rewards(query: str) -> list[str]:
    """Search performance evaluation and rewards: KPI, OKR, bonus, salary review, promotion criteria."""
    return hr_rag.search(query, limit=4)


tools = [search_hr_policy, search_recruitment_process, search_faq, search_performance_rewards]

# ─────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────

sys_prompt = """Bạn là trợ lý HR thông minh và thân thiện của công ty, hỗ trợ nhân viên và ứng viên tra cứu thông tin nhân sự.

Bạn có khả năng tìm kiếm thông tin từ các tài liệu HR nội bộ:
- **Cẩm nang nhân viên**: Chính sách, quy định, phúc lợi, văn hóa công ty
- **Tuyển dụng & Onboarding**: Quy trình ứng tuyển, phỏng vấn, thử việc, hội nhập
- **FAQ**: Câu hỏi thường gặp về lương, bảo hiểm, nghỉ phép, đào tạo
- **Đánh giá & Khen thưởng**: KPI, OKR, đánh giá hiệu suất, tăng lương, thăng chức

NGUYÊN TẮC TRẢ LỜI:
1. Luôn sử dụng tools để tra cứu thông tin trước khi trả lời
2. Trả lời chính xác dựa trên thông tin từ cơ sở dữ liệu HR
3. Nếu không tìm thấy thông tin, hướng dẫn nhân viên liên hệ HR trực tiếp
4. KHÔNG bịa đặt thông tin không có trong tài liệu

CHÍNH SÁCH NGÔN NGỮ:
- Nếu người dùng viết tiếng Việt → trả lời tiếng Việt
- Nếu người dùng viết tiếng Anh → trả lời tiếng Anh
- Mặc định dùng tiếng Việt nếu không rõ

ĐỊNH DẠNG KẾT QUẢ:
- Trả lời rõ ràng, có cấu trúc với Markdown khi cần thiết
- Dùng danh sách, tiêu đề để dễ đọc
- Cuối câu trả lời, thêm gợi ý câu hỏi liên quan nếu phù hợp
"""

# ─────────────────────────────────────────────────────────
# Graph nodes
# ─────────────────────────────────────────────────────────

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
