# ── Stage 1: build wheels ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="ml-trading-nasdaq100"
LABEL description="ML cross-sectional equity strategy – NASDAQ-100 universe"

# System deps for LightGBM + matplotlib
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app
COPY . .

# Pre-create output dirs
RUN mkdir -p data/raw data/interim data/processed outputs

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "-m", "scripts.run_all"]
CMD ["--config", "configs/base.yaml"]
