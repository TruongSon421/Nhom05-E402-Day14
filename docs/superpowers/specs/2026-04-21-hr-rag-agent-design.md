# HR RAG Agent - Design Spec

**Goal:** Xay dung he thong RAG Agent cho domain HR, ho tro 2 version (V1 simple, V2 query rewriting) de phuc vu benchmark va regression testing.

**Domain:** HR - cam nang nhan vien, quy trinh tuyen dung & onboarding, FAQ.

**Tech Stack:** Python async, ChromaDB, OpenAI (gpt-4o-mini + text-embedding-3-small), LangChain (chunking only), pypdf.

---

## 1. Kien truc tong the

```
Documents (PDF/TXT/MD)
    |
    v
DocumentLoader --> Chunker (LangChain) --> ChromaDB (text-embedding-3-small)
                                                |
User Query --[V2: Query Rewriter]--> Embed --> Search (top-5)
                                                |
                                                v
                                    gpt-4o-mini + Contexts --> Answer + metadata
```

- **V1 (AgentV1):** query -> search -> generate
- **V2 (AgentV2):** query -> rewrite_query -> search -> generate

V2 them 1 LLM call de rewrite query, cai thien Hit Rate/MRR nhung ton them cost va latency.

---

## 2. Files

| File | Chuc nang |
|---|---|
| `agent/loader.py` | Load PDF (pypdf) + TXT/MD, tra ve list Document(text, metadata) |
| `agent/chunker.py` | RecursiveCharacterTextSplitter(chunk_size=500, overlap=50). Gan ID: `{filename}_{chunk_index}` |
| `agent/vector_store.py` | Wrap ChromaDB persistent mode (`data/chroma_db/`). add_documents(), search(query, top_k=5), reset() |
| `agent/main_agent.py` | BaseAgent + AgentV1 + AgentV2. Track tokens_used, cost_usd, latency_ms |
| `agent/ingest.py` | Script chay 1 lan: load -> chunk -> embed -> store. `python agent/ingest.py` |

---

## 3. Interface

Agent.query() tra ve format tuong thich voi engine/runner.py:

```python
async def query(self, question: str) -> Dict:
    return {
        "answer": str,
        "contexts": List[str],          # text chunks da retrieve
        "retrieved_ids": List[str],      # chunk IDs cho Hit Rate/MRR
        "metadata": {
            "model": "gpt-4o-mini",
            "tokens_used": int,          # prompt + completion tokens
            "cost_usd": float,           # tinh tu token usage
            "latency_ms": float,         # wall-clock time
            "sources": List[str],        # filename goc
            "version": "v1" | "v2"
        }
    }
```

---

## 4. Tich hop voi Eval Engine

### 4.1 Golden Dataset format (data/golden_set.jsonl)

```json
{
    "question": "Thoi gian thu viec la bao lau?",
    "expected_answer": "Thoi gian thu viec la 2 thang theo quy dinh cong ty.",
    "context": "Theo cam nang nhan vien, muc 3.2...",
    "ground_truth_ids": ["cam_nang_nv_12", "cam_nang_nv_13"],
    "metadata": {"difficulty": "easy", "type": "fact-check", "category": "onboarding"}
}
```

### 4.2 Cap nhat engine/retrieval_eval.py

evaluate_batch() nhan (dataset, results), so sanh ground_truth_ids vs retrieved_ids, tinh avg Hit Rate va MRR that.

### 4.3 Cap nhat main.py

- V1: BenchmarkRunner(AgentV1(), evaluator, judge)
- V2: BenchmarkRunner(AgentV2(), evaluator, judge)
- Regression gate so sanh 3 chieu: Quality (avg_score), Cost (avg_cost_usd), Performance (avg_latency_ms)

### 4.4 Khong thay doi

- engine/runner.py: giu nguyen, chi truyen agent khac
- engine/llm_judge.py: nhom khac implement

---

## 5. Du lieu

- Thu muc `data/documents/`: dat tai lieu HR (PDF, TXT, MD)
- ChromaDB luu tai `data/chroma_db/` (gitignore)
- Golden set: `data/golden_set.jsonl`

---

## 6. Dependencies them

```
chromadb>=0.4.0
langchain-text-splitters>=0.0.1
```

Them vao requirements.txt.

---

## 7. V1 vs V2 Regression

| Chieu | V1 | V2 | Ky vong |
|---|---|---|---|
| Quality (Hit Rate, MRR, Judge Score) | Baseline | Cao hon | Delta > 0 |
| Cost (USD/query) | Thap | Cao hon (~+30%) | Chap nhan |
| Performance (latency) | Nhanh | Cham hon (~+200ms) | Chap nhan |

Release Gate: Approve neu Quality tang va Cost/Latency trong nguong cho phep.
