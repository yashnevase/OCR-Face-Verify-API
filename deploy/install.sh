#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing $PYTHON_BIN. Install Python 3.10 before continuing." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  python3.10-venv python3.10-dev build-essential pkg-config \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
  tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin tesseract-ocr-mar \
  poppler-utils curl ca-certificates

"$PYTHON_BIN" -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

if [[ ! -f "$APP_DIR/.env" ]]; then
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
fi

"$APP_DIR/.venv/bin/python" "$APP_DIR/deploy/preload_models.py"
echo "Installation complete. Review $APP_DIR/.env, then start with deploy/start.sh."
