import base64
import io
import logging
import os
import re
import shutil
import zipfile
from contextlib import asynccontextmanager
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from insightface.app import FaceAnalysis
from pydantic import BaseModel
from PIL import Image
import pytesseract

from ocr_pipeline import (
    init_engine,
    get_engine,
    extract_marksheet_fields,
    normalize_for_search,
    OCR_MIN_CONF,
)

try:
    from pdf2image import convert_from_bytes
except ImportError:
    convert_from_bytes = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import docx
except ImportError:
    docx = None

# Auto-detect Tesseract path (cross-platform: Windows, Linux, macOS)
def _find_tesseract():
    # Try PATH first (works on Linux, Docker, macOS with Homebrew)
    if shutil.which("tesseract"):
        return shutil.which("tesseract")
    # Windows default location
    if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
        return r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # macOS Homebrew location
    if os.path.exists("/usr/local/bin/tesseract"):
        return "/usr/local/bin/tesseract"
    if os.path.exists("/opt/homebrew/bin/tesseract"):
        return "/opt/homebrew/bin/tesseract"
    return None

TESSERACT_PATH = _find_tesseract()
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Main external libraries used in this API:
# - FastAPI/Uvicorn: HTTP API server.
# - InsightFace + ONNX Runtime: local face detection/verification model.
# - PaddleOCR: primary free/self-hosted OCR engine for marksheets.
# - Tesseract + pytesseract: fallback OCR engine.
# - OpenCV/Pillow/PyMuPDF/pdf2image/python-docx: file decoding and preprocessing.

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("uvicorn.error")

# Configuration from environment variables
PORT = int(os.getenv("PORT", "8000"))
WORKERS = int(os.getenv("WORKERS", "4"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
if CORS_ORIGINS == ["*"]:
    CORS_ORIGINS = ["*"]
else:
    CORS_ORIGINS = [o.strip() for o in CORS_ORIGINS if o.strip()]

face_app: Optional[FaceAnalysis] = None
ocr_ready: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_app, ocr_ready
    try:
        # InsightFace downloads/uses the "buffalo_l" face model pack.
        # This is only for /verify and /verify-json; OCR does not use this model.
        face_app = FaceAnalysis(name="buffalo_l")
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("✓ InsightFace model initialized (buffalo_l)")
    except Exception as e:
        logger.error("✗ Failed to initialize InsightFace: %s", e)
        raise
    
    try:
        # OCR engine selection lives in ocr_pipeline.py:
        # primary = PaddleOCR, fallback = Tesseract if Paddle is unavailable/fails.
        active = init_engine()
        if active:
            ocr_ready = True
            logger.info("\u2713 OCR engine ready (active=%s)", active)
        else:
            ocr_ready = False
            logger.warning("\u2717 No OCR engine available. OCR endpoints will return 503.")
    except Exception as _e:
        ocr_ready = False
        logger.warning("\u2717 OCR engine init failed: %s. OCR endpoints will return 503.", _e)
    
    logger.info(f"Server starting on port {PORT} with {WORKERS} workers")
    yield
    logger.info("Server shutting down")


app = FastAPI(title="Face Verify API", version="0.1.0", lifespan=lifespan)

# CORS Configuration - allow all origins for now (can be restricted via CORS_ORIGINS env var)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class VerifyJSONRequest(BaseModel):
    image1: str  # base64 (optionally data URL) or http(s) URL
    image2: str  # base64 (optionally data URL) or http(s) URL
    threshold: float = 0.55  # cosine similarity threshold [0..1]


class OCRJSONRequest(BaseModel):
    image: str  # base64 (optionally data URL) or http(s) URL to an image, PDF, or DOCX file
    min_chars: int = 20
    min_words: int = 5
    return_text: bool = True


def _bytes_from_b64(s: str) -> bytes:
    if "," in s and s.strip().lower().startswith("data:"):
        s = s.split(",", 1)[1]
    return base64.b64decode(s)


def _read_bytes_from_url(url: str, timeout: int = 10) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def _try_decode_image_bytes(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img


def _looks_like_pdf(data: bytes) -> bool:
    return data[:4] == b"%PDF"


def _looks_like_docx(data: bytes) -> bool:
    return zipfile.is_zipfile(io.BytesIO(data)) and b"word/document.xml" in data[:4096]


def _read_pdf_from_bytes(data: bytes) -> np.ndarray:
    # Prefer PyMuPDF (pymupdf) which doesn't require external Poppler tools on PATH.
    if fitz is not None:
        try:
            doc = fitz.open(stream=data, filetype="pdf")
            if doc.page_count < 1:
                raise ValueError("Unable to read pages from PDF")
            page = doc.load_page(0)
            # render at higher resolution for better OCR
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            mode = "RGB" if pix.n < 4 else "RGBA"
            pil_img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            pil_img = pil_img.convert("RGB")
            arr = np.array(pil_img)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            # fallthrough to pdf2image if available
            pass

    if convert_from_bytes is None:
        raise ValueError("PDF support requires the pdf2image package or PyMuPDF (pymupdf)")
    images = convert_from_bytes(data, first_page=1, last_page=1)
    if not images:
        raise ValueError("Unable to convert PDF to image")
    pil_img = images[0].convert("RGB")
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _read_docx_text_from_bytes(data: bytes) -> str:
    if docx is None:
        raise ValueError("DOCX support requires the python-docx package")
    with io.BytesIO(data) as bio:
        document = docx.Document(bio)
    return "\n".join(p.text for p in document.paragraphs if p.text.strip())


def _is_blank_image(img: np.ndarray) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std = float(np.std(gray))
    if std < 8.0:
        return True
    white_ratio = float(np.mean(gray > 245))
    black_ratio = float(np.mean(gray < 10))
    return white_ratio > 0.98 or black_ratio > 0.98


def _extract_text_from_bytes(data: bytes, filename: Optional[str] = None, lang: str = "eng") -> tuple[str, float]:
    content_type = (filename or "").lower()
    if content_type.endswith(".docx") or _looks_like_docx(data):
        text = _read_docx_text_from_bytes(data)
        return text, 95.0
    if _looks_like_pdf(data):
        img = _read_pdf_from_bytes(data)
        return _extract_text_with_confidence(img, lang=lang)
    img = _try_decode_image_bytes(data)
    if img is None:
        if convert_from_bytes is not None and content_type.endswith(".pdf"):
            img = _read_pdf_from_bytes(data)
        else:
            raise ValueError("Unable to decode image or PDF from bytes")
    return _extract_text_with_confidence(img, lang=lang)


def _extract_best_face_embedding(img: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Returns: (embedding (L2-normalized), faces_count)
    Raises ValueError if no face or no embedding.
    """
    assert face_app is not None, "Model not initialized"
    faces = face_app.get(img)
    count = len(faces)
    if count == 0:
        raise ValueError("No face detected")

    # pick the face with largest area and highest score
    def face_score_area(f):
        box = getattr(f, "bbox", None)
        if box is None:
            return 0.0
        x1, y1, x2, y2 = box
        area = max(0, x2 - x1) * max(0, y2 - y1)
        score = getattr(f, "det_score", 0.0)
        return score * area

    faces_sorted = sorted(faces, key=face_score_area, reverse=True)
    best = faces_sorted[0]

    emb = getattr(best, "embedding", None)
    if emb is None:
        emb = getattr(best, "normed_embedding", None)
    if emb is None:
        raise ValueError("Face embedding not available")

    emb = np.asarray(emb, dtype=np.float32)
    norm = np.linalg.norm(emb) + 1e-9
    emb = emb / norm
    return emb, count


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))


def _extract_text_with_confidence(img: np.ndarray, lang: str = "eng") -> tuple[str, float]:
    if _is_blank_image(img):
        return "", 0.0
    engine = get_engine()
    if engine is None or not engine.ready():
        return "", 0.0
    text, avg_conf, _lines = engine.extract(img)
    return text or "", avg_conf


def _doc_heuristics(
    text: str,
    min_chars: int = 20,
    min_words: int = 5,
    avg_conf: float = 0.0,
) -> dict:
    s = text.strip()
    words = re.findall(r"[A-Za-z0-9]+", s)
    lines = [ln for ln in s.splitlines() if ln.strip()]
    letters = sum(1 for c in s if c.isalpha())
    digits = sum(1 for c in s if c.isdigit())
    total = len(s) if len(s) > 0 else 1
    alpha_ratio = letters / total
    digit_ratio = digits / total
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    meaningful_words = [w for w in alpha_words if len(re.sub(r"[^A-Za-z]", "", w)) >= 3]
    avg_word_length = (
        float(sum(len(w) for w in alpha_words)) / len(alpha_words)
        if alpha_words
        else 0.0
    )
    meaningful_ratio = float(len(meaningful_words)) / len(words) if words else 0.0
    has_document_structure = (
        len(lines) >= 2
        or (len(lines) == 1 and len(words) >= int(min_words * 1.5))
    )
    # line-level structure: prefer documents where many lines contain multiple meaningful words
    line_word_counts = [len(re.findall(r"[A-Za-z0-9]+", ln)) for ln in lines] if lines else []
    lines_with_3plus = sum(1 for c in line_word_counts if c >= 3)
    prop_lines_with_3plus = (lines_with_3plus / len(lines)) if lines else 0.0
    avg_line_length = float(sum(len(ln) for ln in lines)) / len(lines) if lines else 0.0
    meaningful_line_counts = [
        len([w for w in re.findall(r"[A-Za-z0-9]+", ln) if len(re.sub(r"[^A-Za-z]", "", w)) >= 3])
        for ln in lines
    ]
    prop_meaningful_lines = (
        sum(1 for c in meaningful_line_counts if c >= 2) / len(lines)
    ) if lines else 0.0

    lower = s.lower()
    doc_keywords = {
        "certificate",
        "exam",
        "marks",
        "subject",
        "percentage",
        "result",
        "candidate",
        "name",
        "address",
        "date",
        "grade",
        "score",
        "signature",
        "official",
        "issued",
        "department",
        "authority",
        "pass",
        "regd",
        "registration",
        "board",
        "school",
    }
    negative_keywords = {
        "request failed",
        "status code",
        "longitude",
        "latitude",
        "gmt",
        "translate",
        "google",
        "path.join",
        "fs.exists",
        "require(",
        "const ",
        "console.",
        "application.log",
        "process.env",
        "http",
        "error",
        "exception",
        "warning",
    }
    doc_keyword_count = sum(1 for kw in doc_keywords if kw in lower)
    negative_keyword_count = sum(1 for kw in negative_keywords if kw in lower)
    has_negative_signals = negative_keyword_count > 0

    is_confident = avg_conf >= 30.0
    doc_keyword_signal = doc_keyword_count >= 2
    has_repeated_line = any(
        lower.count(line.lower()) > 1
        for line in lines
        if len(line.strip()) >= 10
    )

    is_long_doc = (
        len(s) >= 200
        and len(lines) >= 5
        and len(words) >= 50
        and alpha_ratio >= 0.45
        and meaningful_ratio >= 0.45
        and avg_line_length >= 25
        and prop_lines_with_3plus >= 0.6
        and prop_meaningful_lines >= 0.6
        and not (has_negative_signals and not doc_keyword_signal)
    )
    is_likely = (
        (doc_keyword_signal and len(s) >= 100 and prop_meaningful_lines >= 0.4)
        or (
            len(s) >= int(min_chars)
            and len(words) >= int(min_words)
            and alpha_ratio > 0.3
            and avg_word_length >= 3.0
            and meaningful_ratio >= 0.4
            and len(meaningful_words) >= max(2, int(min_words * 0.5))
            and has_document_structure
            and (is_confident or is_long_doc)
            and not (has_negative_signals and not doc_keyword_signal)
        )
    )
    return {
        "text_length": len(s),
        "words": len(words),
        "lines": len(lines),
        "alpha_ratio": alpha_ratio,
        "digit_ratio": digit_ratio,
        "ocr_confidence": avg_conf,
        "avg_word_length": avg_word_length,
        "avg_line_length": avg_line_length,
        "lines_with_words_ratio": prop_lines_with_3plus,
        "meaningful_ratio": meaningful_ratio,
        "prop_meaningful_lines": prop_meaningful_lines,
        "doc_keyword_count": doc_keyword_count,
        "negative_keyword_count": negative_keyword_count,
        "has_repeated_line": has_repeated_line,
        "is_likely_document": bool(is_likely),
    }


@app.get("/health")
async def health():
    engine = get_engine()
    return {
        "status": "ok",
        "name": app.title,
        "version": app.version,
        "ocr_ready": bool(ocr_ready),
        "ocr_engine": (engine.active if engine else None),
        "ocr_engine_info": (engine.info() if engine else None),
    }


@app.post("/verify")
async def verify_form(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    threshold: float = Form(0.55),
):
    """
    Multipart form-data verification.
    Fields: image1 (file), image2 (file), threshold (float, optional)
    """
    try:
        img1_bytes = await image1.read()
        img2_bytes = await image2.read()
        img1 = _try_decode_image_bytes(img1_bytes)
        img2 = _try_decode_image_bytes(img2_bytes)
        if img1 is None or img2 is None:
            raise ValueError("Unable to decode one or both images")

        emb1, c1 = _extract_best_face_embedding(img1)
        emb2, c2 = _extract_best_face_embedding(img2)
        score = _cosine_similarity(emb1, emb2)
        is_match = score >= float(threshold)

        return {
            "match": bool(is_match),
            "score": score,
            "threshold": float(threshold),
            "faces_detected": {"image1": c1, "image2": c2},
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Unexpected error in /verify")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/ocr")
async def ocr_form(
    image: UploadFile = File(...),
    min_chars: int = Form(20),
    min_words: int = Form(5),
    return_text: bool = Form(True),
):
    if not ocr_ready:
        raise HTTPException(status_code=503, detail="OCR engine not available on this server")
    try:
        file_bytes = await image.read()
        text, avg_conf = _extract_text_from_bytes(file_bytes, filename=image.filename)
        heur = _doc_heuristics(
            text,
            min_chars=min_chars,
            min_words=min_words,
            avg_conf=avg_conf,
        )
        engine = get_engine()
        resp = {
            "ok": True,
            "engine": (engine.last_used if engine else None),
            "engine_info": (engine.info() if engine else None),
            "low_quality": bool(avg_conf < OCR_MIN_CONF),
            "fields": extract_marksheet_fields(text),
            "search_text": normalize_for_search(text),
            **heur,
        }
        if return_text:
            resp["text"] = text
        return resp
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        logger.exception("Unexpected error in /ocr")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/ocr-json")
async def ocr_json(payload: OCRJSONRequest):
    if not ocr_ready:
        raise HTTPException(status_code=503, detail="OCR engine not available on this server")
    try:
        s = payload.image.strip()
        if s.lower().startswith("http://") or s.lower().startswith("https://"):
            file_bytes = _read_bytes_from_url(payload.image)
            text, avg_conf = _extract_text_from_bytes(file_bytes)
        else:
            file_bytes = _bytes_from_b64(payload.image)
            text, avg_conf = _extract_text_from_bytes(file_bytes)
        heur = _doc_heuristics(
            text,
            min_chars=payload.min_chars,
            min_words=payload.min_words,
            avg_conf=avg_conf,
        )
        engine = get_engine()
        resp = {
            "ok": True,
            "engine": (engine.last_used if engine else None),
            "engine_info": (engine.info() if engine else None),
            "low_quality": bool(avg_conf < OCR_MIN_CONF),
            "fields": extract_marksheet_fields(text),
            "search_text": normalize_for_search(text),
            **heur,
        }
        if payload.return_text:
            resp["text"] = text
        return resp
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        logger.exception("Unexpected error in /ocr-json")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/verify-json")
async def verify_json(payload: VerifyJSONRequest):
    """
    JSON verification.
    Body: { image1: base64|url|dataURL, image2: base64|url|dataURL, threshold?: float }
    """
    try:
        def load_img(s: str) -> np.ndarray:
            sl = s.strip()
            if sl.lower().startswith("http://") or sl.lower().startswith("https://"):
                file_bytes = _read_bytes_from_url(s)
            else:
                file_bytes = _bytes_from_b64(s)
            img = _try_decode_image_bytes(file_bytes)
            if img is None:
                raise ValueError("Unable to decode image bytes for face verification")
            return img

        img1 = load_img(payload.image1)
        img2 = load_img(payload.image2)

        emb1, c1 = _extract_best_face_embedding(img1)
        emb2, c2 = _extract_best_face_embedding(img2)
        score = _cosine_similarity(emb1, emb2)
        is_match = score >= float(payload.threshold)

        return {
            "match": bool(is_match),
            "score": score,
            "threshold": float(payload.threshold),
            "faces_detected": {"image1": c1, "image2": c2},
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Unexpected error in /verify-json")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info"
    )
