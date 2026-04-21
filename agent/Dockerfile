# ============================================================
# Production Dockerfile — Multi-stage, non-root
# Build context: Nhom05-E402-Day14/  (thư mục cha của agent/)
# ============================================================

# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY agent/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

RUN groupadd -r agent && useradd -r -g agent -d /app agent

WORKDIR /app

# Copy packages từ builder
COPY --from=builder /root/.local /home/agent/.local

# Copy application code (từ agent/)
COPY agent/app/   ./app/
COPY agent/utils/ ./utils/

# Copy HR dataset (từ data/ ở cùng cấp với agent/)
COPY data/        ./data/

RUN chown -R agent:agent /app

USER agent

ENV PATH=/home/agent/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Trỏ hr_rag.py đến đúng thư mục trong container
ENV DATA_DIR=/app/data
ENV CHROMA_PERSIST_DIR=/app/.chromadb

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
