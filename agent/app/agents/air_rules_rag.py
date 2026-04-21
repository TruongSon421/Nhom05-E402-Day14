"""RAG helper for airline regulations backed by text files + OpenAI embeddings + Qdrant."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings

DATA_POLICY_DIR = Path(__file__).resolve().parents[2] / "data_policy"
logger = logging.getLogger(__name__)

FALLBACK_DOCS = [
    {
        "title": "Hanh ly xach tay",
        "content": "Hanh khach duoc mang toi da 7kg hanh ly xach tay va chat long khong vuot qua 100ml moi chai.",
    },
    {
        "title": "Giay to tuy than",
        "content": "Can CCCD hoac ho chieu khi bay noi dia, tre em dung giay khai sinh theo quy dinh.",
    },
]


def _load_policy_docs() -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    if DATA_POLICY_DIR.exists():
        for path in sorted(DATA_POLICY_DIR.glob("*.txt")):
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            title = path.stem.replace("_", " ").replace("-", " ").strip().title()
            docs.append({"title": title, "content": content})
    if docs:
        logger.info("Loaded %s policy docs from %s", len(docs), DATA_POLICY_DIR)
        return docs
    logger.warning("No txt policy docs found in %s, using fallback docs", DATA_POLICY_DIR)
    return FALLBACK_DOCS


def _chunk_text(title: str, content: str, chunk_size: int = 180, overlap: int = 40) -> list[str]:
    text = f"{title}. {content}".strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


@dataclass
class RegulationRAG:
    _client: QdrantClient | None = None
    _embed_client: OpenAI | None = None
    _embed_dim: int | None = None
    _initialized: bool = False

    def _client_or_create(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(url=settings.qdrant_url, timeout=4.0)
        return self._client

    def _embedder_or_create(self) -> OpenAI:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for policy RAG embeddings.")
        if self._embed_client is None:
            self._embed_client = OpenAI(api_key=settings.openai_api_key)
        return self._embed_client

    def _embed_text(self, text: str) -> list[float]:
        client = self._embedder_or_create()
        response = client.embeddings.create(model=settings.embedding_model, input=text)
        vector = list(response.data[0].embedding)
        if self._embed_dim is None:
            self._embed_dim = len(vector)
            logger.info("Embedding dimension detected: %s", self._embed_dim)
        return vector

    def initialize(self) -> None:
        if self._initialized:
            return
        client = self._client_or_create()
        collection = settings.qdrant_collection_air_rules
        docs = _load_policy_docs()
        chunks: list[tuple[str, str]] = []
        for doc in docs:
            for chunk in _chunk_text(doc["title"], doc["content"]):
                chunks.append((doc["title"], chunk))

        if not chunks:
            raise RuntimeError("No policy chunks available for indexing.")

        vectors = [self._embed_text(chunk) for _, chunk in chunks]
        embed_dim = self._embed_dim or len(vectors[0])

        existing = [item.name for item in client.get_collections().collections]
        if collection in existing:
            try:
                info = client.get_collection(collection_name=collection)
                current_dim = int(info.config.params.vectors.size)  # type: ignore[union-attr]
            except Exception:
                current_dim = embed_dim
            if current_dim != embed_dim:
                logger.warning(
                    "Recreating collection %s due to dim mismatch %s != %s",
                    collection,
                    current_dim,
                    embed_dim,
                )
                client.delete_collection(collection_name=collection)
                existing.remove(collection)

        if collection not in existing:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
            )

        points: list[PointStruct] = []
        for pid, ((title, chunk), vector) in enumerate(zip(chunks, vectors), start=1):
            points.append(
                PointStruct(
                    id=pid,
                    vector=vector,
                    payload={"title": title, "text": chunk},
                )
            )
        if points:
            client.upsert(collection_name=collection, points=points)
            logger.info("Upserted %s policy chunks to collection %s", len(points), collection)
        self._initialized = True

    def search(self, query: str, limit: int = 3) -> list[str]:
        try:
            self.initialize()
            client = self._client_or_create()
            query_vector = self._embed_text(query)
            if hasattr(client, "query_points"):
                response = client.query_points(
                    collection_name=settings.qdrant_collection_air_rules,
                    query=query_vector,
                    limit=limit,
                )
                result = list(getattr(response, "points", []) or [])
            else:
                result = client.search(
                    collection_name=settings.qdrant_collection_air_rules,
                    query_vector=query_vector,
                    limit=limit,
                )
        except Exception as exc:
            logger.warning("RAG search failed: %s", exc)
            return []

        snippets: list[str] = []
        for item in result:
            payload = item.payload or {}
            title = str(payload.get("title", "Quy dinh"))
            text = str(payload.get("text", "")).strip()
            if text:
                snippets.append(f"[{title}] {text}")
        return snippets

    def health_check(self) -> dict:
        """Return qdrant connectivity status for ops endpoints."""
        try:
            client = self._client_or_create()
            client.get_collections()
            return {"status": "ok", "detail": "connected"}
        except Exception as exc:
            logger.warning("Qdrant health check failed: %s", exc)
            return {"status": "degraded", "detail": str(exc)}


regulation_rag = RegulationRAG()
