# Face Verify API + OCR

Simple, free, self-hosted face verification and document OCR API. No third-party APIs, no data leaves your server.

## APIs (3 Endpoints)

### 1. `GET /health`
Check server status and OCR availability.

**Response:**
```json
{
  "status": "ok",
  "name": "Face Verify API",
  "version": "0.1.0",
  "ocr_ready": true
}
```

---

### 2. `POST /verify` - Face Verification (Multipart)
Compare two faces for attendance/matching.

**Input:** Form-data with two images
- `image1` (file): Enrolled/stored face photo
- `image2` (file): New selfie to verify
- `threshold` (float, optional): 0.55 default (0.55-0.65 recommended)

**Supported formats:** JPG, PNG, WEBP

**Example:**
```bash
curl -X POST http://localhost:8000/verify \
  -F "image1=@/path/to/enrolled.jpg" \
  -F "image2=@/path/to/selfie.jpg" \
  -F "threshold=0.6"
```

**Response:**
```json
{
  "match": true,
  "score": 0.73,
  "threshold": 0.6,
  "faces_detected": {"image1": 1, "image2": 1}
}
```

- `score`: Cosine similarity (0-1), higher = more similar
- `match`: true if score >= threshold

---

### 3. `POST /verify-json` - Face Verification (JSON)
Same as `/verify` but accepts base64 or URLs.

**Input:** JSON body
```json
{
  "image1": "base64string_or_url",
  "image2": "base64string_or_url",
  "threshold": 0.6
}
```

**Example with URLs:**
```bash
curl -X POST http://localhost:8000/verify-json \
  -H "Content-Type: application/json" \
  -d '{
    "image1": "https://example.com/enrolled.jpg",
    "image2": "https://example.com/selfie.jpg",
    "threshold": 0.6
  }'
```

**Example with base64:**
```bash
B64_1=$(base64 -i /path/to/enrolled.jpg)
B64_2=$(base64 -i /path/to/selfie.jpg)
curl -X POST http://localhost:8000/verify-json \
  -H "Content-Type: application/json" \
  -d "{\"image1\":\"$B64_1\",\"image2\":\"$B64_2\",\"threshold\":0.6}"
```

---

### 4. `POST /ocr` - Document OCR (Multipart)
Extract text from documents to verify they're genuine (not blank/fake).

**Input:** Form-data
- `image` (file): Document photo
- `min_chars` (int, optional): Minimum characters to qualify as document (default 20)
- `min_words` (int, optional): Minimum words (default 5)
- `return_text` (bool, optional): Include extracted text (default true)

**Example:**
```bash
curl -X POST http://localhost:8000/ocr \
  -F "image=@/path/to/id_card.jpg" \
  -F "min_chars=50" \
  -F "min_words=10"
```

**Response:**
```json
{
  "ok": true,
  "text_length": 245,
  "words": 32,
  "lines": 8,
  "alpha_ratio": 0.72,
  "digit_ratio": 0.15,
  "is_likely_document": true,
  "text": "NAME: JOHN DOE\nID: 12345678..."
}
```

- `is_likely_document`: true if text_length >= min_chars AND words >= min_words AND alpha_ratio > 0.3

---

### 5. `POST /ocr-json` - Document OCR (JSON)
Same as `/ocr` but accepts base64 or URL.

**Example:**
```bash
curl -X POST http://localhost:8000/ocr-json \
  -H "Content-Type: application/json" \
  -d '{
    "image": "https://example.com/document.jpg",
    "min_chars": 50,
    "min_words": 10,
    "return_text": true
  }'
```

---

## Quick Start (Local)

### 1. Install Python dependencies
```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

### 2. Install Tesseract OCR (optional, for OCR endpoints)
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr
```

### 3. Run server
```bash
venv/bin/python main.py
```

Server starts at `http://localhost:8000`

### 4. Test
```bash
curl http://localhost:8000/health
```

---

## Deploy with Docker

### Build and run locally
```bash
docker build -t face-verify-api .
docker run -p 8000:8000 face-verify-api
```

### Deploy to cloud (Render/Railway/Fly.io)
1. Push code to GitHub
2. Connect to Render/Railway/Fly
3. They auto-build from Dockerfile
4. Expose port 8000

---

## What We Built

### Face Verification (`/verify`, `/verify-json`)
**Purpose:** Verify that the person marking attendance is the actual enrolled user by comparing selfie with stored photo.

**How it works:**
1. Detect faces in both images using RetinaFace (buffalo_l model)
2. Extract 512-dimensional face embeddings using ArcFace model
3. Calculate cosine similarity between embeddings
4. Return match if similarity >= threshold

**Why ArcFace:**
- State-of-the-art accuracy (99.8%+ on LFW benchmark)
- Free, open-source
- Runs locally on CPU
- Industry standard for face verification

**Accuracy:**
- Very high accuracy for frontal faces in good lighting
- Handles some angle variation, glasses, minor age changes
- Struggles with: extreme angles, heavy occlusion, very low resolution

---

### Document OCR (`/ocr`, `/ocr-json`)
**Purpose:** Verify that uploaded documents contain actual text (not blank images or random photos).

**How it works:**
1. Convert image to RGB
2. Run Tesseract OCR to extract text
3. Calculate heuristics:
   - Text length (character count)
   - Word count
   - Alpha ratio (letters / total chars) - filters garbled OCR
   - Line count
4. Return `is_likely_document: true` if criteria met

**Why Tesseract:**
- Free, open-source OCR by Google
- Works offline
- Supports 100+ languages
- Battle-tested for decades

**Accuracy:**
- Good for clear, high-contrast documents
- Struggles with: blurry photos, handwriting, complex backgrounds
- Heuristics prevent false positives (face photos won't trigger as documents)

---

## Dependencies Used

| Package | Purpose | Free? |
|---------|---------|-------|
| **FastAPI** | Web framework for APIs | Yes |
| **Uvicorn** | ASGI server | Yes |
| **InsightFace** | Face detection + recognition models | Yes |
| **ONNX Runtime** | Run neural network models efficiently | Yes |
| **OpenCV** | Image processing | Yes |
| **NumPy** | Numerical operations | Yes |
| **Pillow** | Image loading for OCR | Yes |
| **Pytesseract** | Python wrapper for Tesseract OCR | Yes |
| **Tesseract** | OCR engine (system dependency) | Yes |

**All dependencies are completely free and open-source.**

---

## Safety & Privacy

### ✅ 100% Local Processing
- No API keys needed
- No data sent to third parties
- No cloud AI services
- Everything runs on your server

### ✅ Data Never Leaves
- Face images processed locally
- No storage of images (ephemeral processing)
- No logging to external services

### ✅ Self-Hosted
- You control the server
- You control the data
- Deploy on your own infrastructure

---

## Architecture

```
┌─────────────────┐
│  Your React App │
└────────┬────────┘
         │
         │ HTTP POST (images)
         ▼
┌─────────────────┐
│   Node Server   │
└────────┬────────┘
         │
         │ HTTP POST (images)
         ▼
┌─────────────────────────────┐
│   Face Verify API (Python) │
│   - FastAPI                │
│   - InsightFace (ArcFace)  │
│   - Tesseract OCR          │
└─────────────────────────────┘
```

**Flow for attendance:**
1. User opens React app, takes selfie
2. Node server receives: selfie + userId + location + IP
3. Node fetches enrolled photo from DB
4. Node sends both to Face Verify API
5. API returns: `match: true/false` + score
6. Node marks attendance if match=true

---

## Files Structure

```
face detection/
├── main.py           # API code (all 5 endpoints)
├── requirements.txt  # Python dependencies
├── Dockerfile      # Deployment config
└── venv/           # Virtual environment (not committed)
```

**No clutter. Only 3 essential files.**

---

## Performance

- **Face verification:** ~100-300ms per comparison (CPU)
- **OCR:** ~200-500ms per image (depends on text density)
- **Memory:** ~500MB-1GB RAM (mainly for face recognition models)

---

## Troubleshooting

### OCR not available locally
Install Tesseract:
```bash
brew install tesseract  # macOS
```

### Slow first request
Face model downloads on first run (~300MB). Subsequent runs use cached model.

### No face detected
Ensure:
- Face is clearly visible
- Good lighting
- Not extreme angle
- Minimum resolution 100x100 pixels

### Low match score
- Adjust threshold (0.5-0.7 range)
- Ensure same person in both photos
- Check lighting/angle similarity

---

## Cost: $0

- No API usage fees
- No subscription costs
- No per-request charges
- Only server hosting costs (if deploying to cloud)

**Completely free forever.**
