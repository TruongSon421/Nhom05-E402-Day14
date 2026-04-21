"""RAG helper for HR knowledge base — ChromaDB + OpenAI embeddings.

Chia sẻ cùng collection 'hr_chunks' với engine/retrieval_eval.py để
hai thành phần đọc/ghi nhất quán vào một vector store duy nhất.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from app.config import settings

# ── Đường dẫn — ưu tiên env var (set trong Docker), fallback local ─────────
_THIS_FILE = Path(__file__).resolve()
HR_DATASET_PATH    = Path(os.getenv("DATA_DIR", str(_THIS_FILE.parents[2] / "data"))) / "documents" / "hr_rag_dataset.json"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(_THIS_FILE.parents[2] / ".chromadb"))
COLLECTION_NAME = "hr_chunks"   # phải trùng với retrieval_eval.py
BATCH_SIZE      = 100

logger = logging.getLogger(__name__)

# ── Fallback nếu dataset không tìm thấy ────────────────────────────────────
FALLBACK_DOCS = [
    {
        "chunk_id": "fallback-001",
        "text":     "Công ty áp dụng các chính sách nhân sự minh bạch, công bằng, lấy nhân viên làm trung tâm.",
        "doc_id":   "fallback",
        "title":    "Chính sách nhân sự",
    },
]


def _load_chunks_from_dataset() -> list[dict]:
    """Flatten tất cả subsection + FAQ thành danh sách chunk.
    Logic giống hệt engine/retrieval_eval.py._load_chunks_from_dataset().
    """
    if not HR_DATASET_PATH.exists():
        logger.warning("HR dataset không tìm thấy tại %s — dùng fallback", HR_DATASET_PATH)
        return FALLBACK_DOCS

    try:
        with HR_DATASET_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.error("Lỗi đọc HR dataset: %s — dùng fallback", exc)
        return FALLBACK_DOCS

    chunks: list[dict] = []
    for doc in data.get("documents", []):
        for section in doc.get("sections", []):
            for sub in section.get("subsections", []):
                text = sub.get("chunk_text") or sub.get("content", "")
                if not text:
                    continue
                chunks.append({
                    "chunk_id": sub["subsection_id"],
                    "text":     text.strip(),
                    "doc_id":   doc["doc_id"],
                    "title":    sub.get("title", ""),
                })
            for faq in section.get("faqs", []):
                answer = faq.get("answer", "")
                if not answer:
                    continue
                chunks.append({
                    "chunk_id": faq["faq_id"],
                    "text":     answer.strip(),
                    "doc_id":   doc["doc_id"],
                    "title":    faq.get("question", ""),
                })

    if not chunks:
        logger.warning("Không extract được chunk nào từ HR dataset — dùng fallback")
        return FALLBACK_DOCS

    logger.info("Đã load %d HR chunks từ %s", len(chunks), HR_DATASET_PATH)
    return chunks


@dataclass
class HRDocumentRAG:
    """ChromaDB-backed RAG cho HR knowledge base."""

    persist_dir: str = CHROMA_PERSIST_DIR
    _collection: chromadb.Collection | None = field(default=None, init=False, repr=False)

    # ── Lazy init ────────────────────────────────────────────────────────────

    @property
    def collection(self) -> chromadb.Collection:
        """Kết nối ChromaDB khi cần, index nếu collection rỗng."""
        if self._collection is None:
            self._collection = self._build_collection()
        return self._collection

    def _build_collection(self) -> chromadb.Collection:
        api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY là bắt buộc để dùng HR RAG embeddings.")

        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=settings.embedding_model,   # text-embedding-3-small theo config
        )

        client = chromadb.PersistentClient(path=self.persist_dir)
        col = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

        if col.count() == 0:
            logger.info("[VectorDB] Lần đầu chạy — đang index HR chunks vào ChromaDB…")
            chunks = _load_chunks_from_dataset()

            ids       = [c["chunk_id"] for c in chunks]
            documents = [c["text"]     for c in chunks]
            metadatas = [{"doc_id": c["doc_id"], "title": c["title"]} for c in chunks]

            for i in range(0, len(ids), BATCH_SIZE):
                col.upsert(
                    ids=ids[i : i + BATCH_SIZE],
                    documents=documents[i : i + BATCH_SIZE],
                    metadatas=metadatas[i : i + BATCH_SIZE],
                )
            logger.info("[VectorDB] Đã index %d HR chunks.", len(ids))
        else:
            logger.info(
                "[VectorDB] Dùng lại collection '%s' (%d chunks).",
                COLLECTION_NAME, col.count(),
            )

        return col

    # ── Public API ───────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Khởi động trước (warmup) — gọi khi app start."""
        _ = self.collection  # trigger lazy init

    def search(self, query: str, limit: int = 4) -> list[str]:
        """Truy vấn ChromaDB, trả về list snippet '[title] text'."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
            )
        except Exception as exc:
            logger.warning("HR RAG search thất bại: %s", exc)
            return []

        snippets: list[str] = []
        ids       = (results.get("ids")       or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        for chunk_id, text, meta in zip(ids, documents, metadatas):
            title = (meta or {}).get("title") or chunk_id
            if text:
                snippets.append(f"[{title}] {text}")
        return snippets

    def health_check(self) -> dict:
        """Trả về trạng thái ChromaDB cho ops endpoints."""
        try:
            _ = self.collection
            return {"status": "ok", "detail": f"chromadb connected ({self.collection.count()} chunks)"}
        except Exception as exc:
            logger.warning("ChromaDB health check thất bại: %s", exc)
            return {"status": "degraded", "detail": str(exc)}


hr_rag = HRDocumentRAG()
