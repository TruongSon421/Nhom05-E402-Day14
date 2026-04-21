"""
Adapter benchmark — wrapper gọi thẳng production LangGraph agent.

Không dùng PipelineService vì PipelineService cần Redis (chat history)
và NeMo Guardrails — những thứ không cần thiết khi benchmark.
Thay vào đó, gọi trực tiếp build_multi_agent_graph() (không có Redis/guardrails)
để lấy câu trả lời và contexts thật từ ChromaDB.

Interface query() KHÔNG thay đổi để engine/runner.py tiếp tục hoạt động.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Đưa agent/ vào sys.path để có thể import `app.*` (app nằm trong agent/)
_AGENT_DIR = Path(__file__).resolve().parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from langchain_core.messages import HumanMessage          # noqa: E402
from app.agents.hr_graph import build_multi_agent_graph   # noqa: E402
from app.agents.hr_rag import hr_rag                      # noqa: E402
from app.agents.types import AgentState                   # noqa: E402

logger = logging.getLogger(__name__)


class MainAgent:
    """
    Wrapper benchmark cho production multi-agent (LangGraph + ChromaDB RAG).
    Gọi LangGraph graph trực tiếp, bỏ qua Redis / guardrails / LLM-judge
    để benchmark nhanh và không phụ thuộc infrastructure.
    """

    def __init__(self):
        self.name = "SupportAgent-v1"
        self._graph = build_multi_agent_graph()
        # Warmup ChromaDB (lazy — chỉ thực sự kết nối khi query đầu tiên)
        try:
            hr_rag.initialize()
        except Exception as exc:
            logger.warning("[MainAgent] ChromaDB warmup thất bại: %s", exc)

    async def query(self, question: str) -> Dict:
        """
        Chạy production LangGraph agent, collect toàn bộ stream rồi trả về dict.

        Returns:
            {
                "answer":   str,        # BẮT BUỘC — engine/runner.py đọc field này
                "contexts": list[str],  # Các đoạn tài liệu HR đã retrieve thật
                "metadata": dict,
            }
        """
        state: AgentState = {
            "session_id": "",
            "question": question,
            "human_approved": False,
            "human_feedback": "",
            "trace": ["benchmark:start"],
            "messages": [HumanMessage(content=question)],
        }

        answer_parts: List[str] = []
        contexts: List[str] = []

        try:
            async for event in self._graph.astream_events(state, version="v2"):
                kind = event["event"]

                # Thu thập token từ LLM
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if getattr(chunk, "content", None) and isinstance(chunk.content, str):
                        answer_parts.append(chunk.content)

                # Thu thập kết quả tool (RAG contexts)
                elif kind == "on_tool_end":
                    output = event["data"].get("output")
                    if isinstance(output, list):
                        contexts.extend(str(c) for c in output if c)
                    elif isinstance(output, str) and output:
                        contexts.append(output)

        except Exception as exc:
            logger.error("[MainAgent] LangGraph execution thất bại: %s", exc)
            return {
                "answer": f"Lỗi xử lý: {exc}",
                "contexts": [],
                "metadata": {"error": str(exc)},
            }

        answer = "".join(answer_parts).strip()
        if not answer:
            answer = "Không tìm thấy câu trả lời phù hợp trong tài liệu HR."

        return {
            "answer": answer,
            "contexts": contexts,
            "metadata": {
                "agent": "multi-agent-langgraph",
                "num_contexts": len(contexts),
            },
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def test():
        agent = MainAgent()
        resp = await agent.query("Chính sách nghỉ phép hàng năm của công ty như thế nào?")
        print("Answer:", resp["answer"][:300])
        print("Contexts:", len(resp["contexts"]))
        print("Metadata:", resp["metadata"])

    asyncio.run(test())
