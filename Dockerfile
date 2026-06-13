# ── Production image ─────────────────────────────────────────────────────────
# Target: x86_64 Linux (VPS / cloud).  PaddleOCR runs fast here with MKL-DNN.
# To build locally on an Apple Silicon Mac add --platform=linux/amd64.
# ─────────────────────────────────────────────────────────────────────────────
FROM --platform=linux/amd64 python:3.10-slim

# ── Runtime env defaults (all overridable at container start with -e) ─────────
# OCR_ENGINE=paddle   → PaddleOCR primary, Tesseract fallback
# OCR_ENGINE=tesseract → Tesseract only (set this to skip Paddle)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/app \
    PORT=8000 \
    WORKERS=4 \
    CORS_ORIGINS="*" \
    OCR_ENGINE=paddle \
    OCR_LANG=devanagari \
    TESS_LANG=eng+hin+mar \
    OCR_PREPROCESS=1 \
    OCR_PREPROCESS_FAST=1 \
    OCR_MIN_CONF=45 \
    PADDLE_ENABLE_MKLDNN=1 \
    PADDLE_CPU_THREADS=4 \
    PADDLE_DET_LIMIT_SIDE_LEN=960

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# libgomp1      – required by PaddleOCR CPU runtime
# tesseract-*   – Tesseract fallback with Hindi + Marathi language packs
# poppler-utils – PDF page rendering (pdf2image / PyMuPDF backup)
# ─────────────────────────────────────────────────────────────────────────────
COPY requirements.txt /app/

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin tesseract-ocr-mar \
        poppler-utils curl ca-certificates \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# ── Application code ──────────────────────────────────────────────────────────
COPY . /app/

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd -m -d /home/appuser -s /bin/bash appuser \
    && chown -R appuser:appuser /app

# ── Pre-cache models at build time (avoids cold-start download on first request)
# InsightFace (face verification)
RUN python -c "\
from insightface.app import FaceAnalysis; \
a = FaceAnalysis(name='buffalo_l'); \
a.prepare(ctx_id=0, det_size=(640,640)); \
print('InsightFace models cached')" \
    || echo "InsightFace cache failed – will download at runtime"

# PaddleOCR – devanagari detection + recognition models
RUN python -c "\
from paddleocr import PaddleOCR; \
PaddleOCR(use_angle_cls=False, lang='devanagari', show_log=False, use_gpu=False); \
print('PaddleOCR devanagari models cached')" \
    || echo "PaddleOCR cache failed – will download at runtime"

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD uvicorn main:app --host 0.0.0.0 --port "$PORT" --proxy-headers --workers "$WORKERS"
