"""
Retrieval Evaluator - Lab Day 14
Người 2: Hit Rate & MRR với Vector DB thực tế (ChromaDB + OpenAI Embeddings)
"""
import json
import os
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# ─── Hằng số ────────────────────────────────────────────────────────────────
COLLECTION_NAME = "hr_chunks"
DATASET_PATH    = "data/documents/hr_rag_dataset.json"
TOP_K_DEFAULT   = 3


# ─── Vector DB helpers ──────────────────────────────────────────────────────

def _load_chunks_from_dataset() -> List[Dict]:
    """Đọc tất cả subsection + FAQ chunks từ hr_rag_dataset.json."""
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)

    chunks = []
    for doc in data["documents"]:
        for section in doc.get("sections", []):
            for sub in section.get("subsections", []):
                chunks.append({
                    "chunk_id":   sub["subsection_id"],
                    "text":       sub.get("chunk_text") or sub.get("content", ""),
                    "doc_id":     doc["doc_id"],
                    "title":      sub.get("title", ""),
                })
            for faq in section.get("faqs", []):
                chunks.append({
                    "chunk_id":   faq["faq_id"],
                    "text":       faq.get("answer", ""),
                    "doc_id":     doc["doc_id"],
                    "title":      faq.get("question", ""),
                })
    return chunks


def build_vector_db(persist_dir: str = ".chromadb") -> chromadb.Collection:
    """
    Xây dựng (hoặc nạp lại) ChromaDB collection từ hr_rag_dataset.json.
    Dùng OpenAI text-embedding-3-small để nhúng văn bản.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError("Thiếu OPENAI_API_KEY trong .env")

    # Embedding function
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name="text-embedding-3-small",
    )

    # Client lưu dữ liệu trên đĩa để không phải index lại mỗi lần chạy
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Nếu collection rỗng → index lần đầu
    if collection.count() == 0:
        print("[VectorDB] Lần đầu chạy – đang index chunks vào ChromaDB…")
        chunks = _load_chunks_from_dataset()

        ids       = [c["chunk_id"] for c in chunks]
        documents = [c["text"]     for c in chunks]
        metadatas = [{"doc_id": c["doc_id"], "title": c["title"]} for c in chunks]

        # Upsert theo batch để tránh vượt giới hạn API
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            collection.upsert(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )
        print(f"[VectorDB] Đã index {len(ids)} chunks.")
    else:
        print(f"[VectorDB] Dùng lại collection '{COLLECTION_NAME}' ({collection.count()} chunks).")

    return collection


def query_vector_db(collection: chromadb.Collection, question: str, top_k: int = TOP_K_DEFAULT) -> List[str]:
    """
    Truy vấn ChromaDB, trả về danh sách chunk_id theo thứ tự độ tương đồng giảm dần.
    """
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
    )
    # results["ids"] là list of lists (1 query → 1 inner list)
    return results["ids"][0] if results["ids"] else []


# ─── RetrievalEvaluator ─────────────────────────────────────────────────────

class RetrievalEvaluator:
    """
    Đánh giá chất lượng retrieval stage bằng hai chỉ số:
      • Hit Rate  – ít nhất 1 chunk đúng có trong top-K kết quả
      • MRR       – Mean Reciprocal Rank (vị trí trung bình của kết quả đúng đầu tiên)
    """

    def __init__(self, persist_dir: str = ".chromadb"):
        self._collection = None
        self._persist_dir = persist_dir

    @property
    def collection(self) -> chromadb.Collection:
        """Lazy-load: chỉ kết nối VectorDB khi cần."""
        if self._collection is None:
            self._collection = build_vector_db(self._persist_dir)
        return self._collection

    # ── Metric primitives ────────────────────────────────────────────────────

    def calculate_hit_rate(
        self,
        expected_ids:  List[str],
        retrieved_ids: List[str],
        top_k:         int = TOP_K_DEFAULT,
    ) -> float:
        """
        Hit Rate@K: 1.0 nếu ít nhất 1 trong expected_ids xuất hiện trong top_k
        của retrieved_ids, ngược lại 0.0.
        """
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(
        self,
        expected_ids:  List[str],
        retrieved_ids: List[str],
    ) -> float:
        """
        Reciprocal Rank: 1 / rank của expected_id đầu tiên tìm thấy trong
        retrieved_ids (1-indexed).  Trả về 0.0 nếu không tìm thấy.
        """
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in expected_ids:
                return 1.0 / rank
        return 0.0

    # ── Batch evaluation ─────────────────────────────────────────────────────

    async def evaluate_batch(
        self,
        dataset: List[Dict],
        top_k:   int = TOP_K_DEFAULT,
    ) -> Dict:
        """
        Chạy đánh giá retrieval cho toàn bộ golden dataset.

        Mỗi phần tử trong dataset cần có:
          • "question"          – câu hỏi gửi vào retrieval
          • "ground_truth_ids"  – list chunk_id đúng (từ golden_set.jsonl)

        Trả về:
          {
            "avg_hit_rate":   float,   # Hit Rate@K trung bình
            "avg_mrr":        float,   # MRR trung bình
            "total_cases":    int,
            "hits":           int,     # số case có hit
            "top_k":          int,
            "per_case":       list,    # chi tiết từng case
          }
        """
        hit_rates: List[float] = []
        mrrs:      List[float] = []
        per_case:  List[Dict]  = []

        for case in dataset:
            question         = case.get("question", "")
            ground_truth_ids = case.get("ground_truth_ids", [])

            # Out-of-scope cases: không có ground truth → bỏ qua retrieval metrics
            if not ground_truth_ids:
                per_case.append({
                    "question":         question,
                    "ground_truth_ids": ground_truth_ids,
                    "retrieved_ids":    [],
                    "hit_rate":         None,
                    "mrr":              None,
                    "skipped":          True,
                })
                continue

            # ── Truy vấn Vector DB ──────────────────────────────────────────
            retrieved_ids = query_vector_db(self.collection, question, top_k=top_k)

            # ── Tính metric ─────────────────────────────────────────────────
            hr  = self.calculate_hit_rate(ground_truth_ids, retrieved_ids, top_k)
            mrr = self.calculate_mrr(ground_truth_ids, retrieved_ids)

            hit_rates.append(hr)
            mrrs.append(mrr)
            per_case.append({
                "question":         question,
                "ground_truth_ids": ground_truth_ids,
                "retrieved_ids":    retrieved_ids,
                "hit_rate":         hr,
                "mrr":              mrr,
                "skipped":          False,
            })

        avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
        avg_mrr      = sum(mrrs)      / len(mrrs)      if mrrs      else 0.0

        print(
            f"[Retrieval] Top-{top_k}  |  "
            f"Hit Rate: {avg_hit_rate:.3f}  |  "
            f"MRR: {avg_mrr:.3f}  |  "
            f"Cases: {len(hit_rates)} (skipped {len(per_case) - len(hit_rates)})"
        )

        return {
            "avg_hit_rate": round(avg_hit_rate, 4),
            "avg_mrr":      round(avg_mrr,      4),
            "total_cases":  len(hit_rates),
            "hits":         int(sum(hit_rates)),
            "top_k":        top_k,
            "per_case":     per_case,
        }


# ─── CLI Demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    GOLDEN_SET_PATH = "data/golden_set.jsonl"

    if not os.path.exists(GOLDEN_SET_PATH):
        print(f"[ERROR] Không tìm thấy {GOLDEN_SET_PATH}. Hãy chạy `python data/synthetic_gen.py` trước.")
        exit(1)

    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    evaluator = RetrievalEvaluator()

    async def demo():
        results = await evaluator.evaluate_batch(dataset, top_k=3)
        print("\n=== Kết quả Retrieval Evaluation ===")
        print(f"  Hit Rate@3 : {results['avg_hit_rate']:.4f}  ({results['hits']}/{results['total_cases']} cases)")
        print(f"  MRR        : {results['avg_mrr']:.4f}")
        print(f"  Top-K      : {results['top_k']}")

        # In 5 case đầu tiên để kiểm tra
        print("\n--- 5 case mẫu ---")
        for case in results["per_case"][:5]:
            if case["skipped"]:
                continue
            icon = "✓" if case["hit_rate"] == 1.0 else "✗"
            print(f"  {icon} HR={case['hit_rate']} MRR={case['mrr']:.3f}")
            print(f"    Q: {case['question'][:80]}")
            print(f"    Expected : {case['ground_truth_ids']}")
            print(f"    Retrieved: {case['retrieved_ids']}\n")

    asyncio.run(demo())
