# Production Deployment

This deployment targets an x86_64 Ubuntu/Debian server. Python 3.10 is
recommended for the pinned PaddlePaddle, PaddleOCR, InsightFace, and ONNX
Runtime versions.

## Install

```bash
sudo mkdir -p /opt/OCR-Face-Verify-API
sudo chown "$USER":"$USER" /opt/OCR-Face-Verify-API
git clone YOUR_REPOSITORY_URL /opt/OCR-Face-Verify-API
cd /opt/OCR-Face-Verify-API
chmod +x deploy/install.sh deploy/start.sh
./deploy/install.sh
```

The installer creates `.venv`, installs system and Python dependencies,
creates `.env`, and downloads and validates PaddleOCR and InsightFace models.

Review `.env`, then test:

```bash
./deploy/start.sh
curl http://localhost:8000/health
```

The health response must show:

```json
{
  "ocr_ready": true,
  "ocr_engine": "paddle",
  "ocr_engine_info": {
    "paddle_required": true,
    "paddle_initialized": true
  }
}
```

With `OCR_REQUIRE_PADDLE=1`, the API refuses to start if PaddleOCR cannot
initialize. Tesseract remains available only as a per-request fallback if
Paddle inference unexpectedly fails.

## Run With systemd

Update the `User`, `Group`, and paths in `deploy/ocr-face-api.service` if the
installation differs from `/opt/OCR-Face-Verify-API`, then:

```bash
sudo cp deploy/ocr-face-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ocr-face-api
sudo systemctl status ocr-face-api
curl http://localhost:8000/health
```

Logs:

```bash
sudo journalctl -u ocr-face-api -f
```

Keep `WORKERS=1` unless the server has enough RAM for a separate PaddleOCR and
InsightFace model in every worker.
