FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/app \
    PORT=8000 \
    WORKERS=4 \
    CORS_ORIGINS="*"

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt /app/

# Install system deps (including Tesseract and Poppler) and Python deps,
# then purge build tools to keep image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        tesseract-ocr \
        poppler-utils \
        curl \
        ca-certificates \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY . /app/

# Create an unprivileged user and ensure /app is owned by it
RUN useradd -m -d /home/appuser -s /bin/bash appuser \
    && chown -R appuser:appuser /app

# Cache InsightFace models at build time to avoid runtime cold-starts.
# If this step fails, the build will continue (models will be downloaded at runtime).
RUN python -c "from insightface.app import FaceAnalysis; a=FaceAnalysis(name='buffalo_l'); a.prepare(ctx_id=0, det_size=(640,640)); print('Models cached')" || echo "Model caching failed, continuing..."

# Ensure app files are owned by the runtime user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 8000

# Basic healthcheck against the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Run Uvicorn with environment variables
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--workers", "4"]
