"""
RAG Agent V1 (simple) va V2 (query rewriting + top-K lon hon + prompt tot hon).
Dung chung ChromaDB collection 'hr_chunks' voi engine/retrieval_eval.py.
"""
import asyncio
import os
import time
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# Pricing gpt-4o-mini (per 1M tokens)
COST_PER_1M_INPUT = 0.15
COST_PER_1M_OUTPUT = 0.60

V1_SYSTEM_PROMPT = """Ban la tro ly HR. Tra loi cau hoi dua tren context duoc cung cap.
Neu khong tim thay thong tin, noi ro ban khong biet.
Tra loi bang tieng Viet, ngan gon."""

V2_SYSTEM_PROMPT = """Ban la tro ly HR chuyen nghiep va chi tiet. Tra loi cau hoi dua tren context duoc cung cap.

Yeu cau:
- Trich dan cu the tu tai lieu khi tra loi (vi du: "Theo muc 3.2 cua Cam nang Nhan vien...")
- Neu context co nhieu thong tin lien quan, tong hop day du
- Neu khong tim thay thong tin trong context, noi ro ban khong co thong tin ve van de nay
- Tra loi bang tieng Viet, ro rang, chuyen nghiep"""


class _ChromaStore:
    """Shared ChromaDB access — same collection as retrieval_eval.py."""

    def __init__(self, persist_dir: str = ".chromadb"):
        api_key = os.getenv("OPENAI_API_KEY", "")
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small",
        )
        client = chromadb.PersistentClient(path=persist_dir)
        self.collection = client.get_or_create_collection(
            name="hr_chunks",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    def search(self, query: str, top_k: int = 3):
        """Returns (chunk_ids, texts, titles)."""
        if self.collection.count() == 0:
            return [], [], []
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
        )
        ids = results["ids"][0] if results["ids"] else []
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        titles = [m.get("title", "") for m in metas]
        return ids, docs, titles


class BaseAgent:
    def __init__(self, store: _ChromaStore = None, model: str = "gpt-4o-mini"):
        self.store = store or _ChromaStore()
        self.llm = AsyncOpenAI()
        self.model = model
        self.name = "BaseAgent"
        self.version = "base"
        self.system_prompt = V1_SYSTEM_PROMPT

    def _calc_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * COST_PER_1M_INPUT + completion_tokens * COST_PER_1M_OUTPUT) / 1_000_000

    async def _generate(self, question: str, contexts: List[str]) -> tuple:
        context_text = "\n---\n".join(contexts)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nCau hoi: {question}"},
        ]
        resp = await self.llm.chat.completions.create(model=self.model, messages=messages)
        answer = resp.choices[0].message.content
        return answer, resp.usage.prompt_tokens, resp.usage.completion_tokens

    async def query(self, question: str) -> Dict:
        raise NotImplementedError


class AgentV1(BaseAgent):
    """Simple RAG: query -> search(top-3) -> generate."""

    def __init__(self, store: _ChromaStore = None, model: str = "gpt-4o-mini"):
        super().__init__(store, model)
        self.name = "SupportAgent-v1"
        self.version = "v1"
        self.system_prompt = V1_SYSTEM_PROMPT

    async def query(self, question: str) -> Dict:
        start = time.perf_counter()

        ids, docs, titles = self.store.search(question, top_k=3)
        sources = list(set(titles))

        answer, p_tok, c_tok = await self._generate(question, docs)

        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "answer": answer,
            "contexts": docs,
            "retrieved_ids": ids,
            "metadata": {
                "model": self.model,
                "tokens_used": p_tok + c_tok,
                "cost_usd": self._calc_cost(p_tok, c_tok),
                "latency_ms": round(latency_ms, 2),
                "sources": sources,
                "version": self.version,
            },
        }


class AgentV2(BaseAgent):
    """Enhanced RAG: query -> rewrite -> search(top-5) -> generate (better prompt)."""

    def __init__(self, store: _ChromaStore = None, model: str = "gpt-4o-mini"):
        super().__init__(store, model)
        self.name = "SupportAgent-v2"
        self.version = "v2"
        self.system_prompt = V2_SYSTEM_PROMPT

    async def _rewrite_query(self, question: str) -> tuple:
        messages = [
            {"role": "system", "content": "Viet lai cau hoi sau cho ro rang va cu the hon de tim kiem trong tai lieu HR. Chi tra ve cau hoi da viet lai, khong giai thich."},
            {"role": "user", "content": question},
        ]
        resp = await self.llm.chat.completions.create(model=self.model, messages=messages)
        return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens

    async def query(self, question: str) -> Dict:
        start = time.perf_counter()

        # Rewrite query
        rewritten, rw_p, rw_c = await self._rewrite_query(question)

        # Retrieve with rewritten query, top_k=5
        ids, docs, titles = self.store.search(rewritten, top_k=5)
        sources = list(set(titles))

        # Generate with original question + retrieved contexts
        answer, gen_p, gen_c = await self._generate(question, docs)

        total_p = rw_p + gen_p
        total_c = rw_c + gen_c
        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "answer": answer,
            "contexts": docs,
            "retrieved_ids": ids,
            "metadata": {
                "model": self.model,
                "tokens_used": total_p + total_c,
                "cost_usd": self._calc_cost(total_p, total_c),
                "latency_ms": round(latency_ms, 2),
                "sources": sources,
                "version": self.version,
                "rewritten_query": rewritten,
            },
        }


# Backward compat
MainAgent = AgentV1

if __name__ == "__main__":
    agent = AgentV1()
    async def test():
        resp = await agent.query("Thoi gian thu viec la bao lau?")
        print(f"Answer: {resp['answer'][:100]}")
        print(f"Retrieved IDs: {resp['retrieved_ids']}")
        print(f"Cost: ${resp['metadata']['cost_usd']}")
        print(f"Latency: {resp['metadata']['latency_ms']}ms")
    asyncio.run(test())
