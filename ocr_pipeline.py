"""
OCR pipeline — production-grade, Indian document extraction.

Engines  : PaddleOCR (primary, x86 Linux) → Tesseract (fallback / ARM Mac)
Languages: English, Hindi, Marathi (devanagari model + eng+hin+mar tess packs)
Formats  : JPEG, PNG, WebP, TIFF, BMP, PDF, DOCX — any quality / size
Fields   : name, seat/roll no, board, exam, year, marks, %, CGPA, result, subjects
"""
from __future__ import annotations

import logging
import os
import platform
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Optional heavy imports — service starts even if one is missing
# ---------------------------------------------------------------------------
try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception:
    PaddleOCR = None

try:
    import pytesseract  # type: ignore
    from PIL import Image as _PILImage  # type: ignore
except Exception:
    pytesseract = None
    _PILImage = None

# ---------------------------------------------------------------------------
# Configuration — all tunable via environment variables
# ---------------------------------------------------------------------------
OCR_ENGINE   = os.getenv("OCR_ENGINE",   "paddle").lower()   # paddle | tesseract
OCR_REQUIRE_PADDLE = os.getenv("OCR_REQUIRE_PADDLE", "1") == "1"
OCR_LANG     = os.getenv("OCR_LANG",     "devanagari")        # paddle lang model
TESS_LANG    = os.getenv("TESS_LANG",    "eng+hin+mar")       # tesseract lang packs
OCR_PREPROCESS      = os.getenv("OCR_PREPROCESS",      "1") == "1"
OCR_PREPROCESS_FAST = os.getenv("OCR_PREPROCESS_FAST", "1") == "1"
OCR_MIN_CONF = float(os.getenv("OCR_MIN_CONF", "45"))

_IS_X86 = platform.machine().lower() in ("x86_64", "amd64", "i386", "i686")
PADDLE_CPU_THREADS       = int(os.getenv("PADDLE_CPU_THREADS",       str(os.cpu_count() or 4)))
PADDLE_ENABLE_MKLDNN     = os.getenv("PADDLE_ENABLE_MKLDNN", "1" if _IS_X86 else "0") == "1"
PADDLE_DET_LIMIT_SIDE_LEN = int(os.getenv("PADDLE_DET_LIMIT_SIDE_LEN", "960"))


# ===========================================================================
# Image preprocessing — fast + quality adaptive
# ===========================================================================
def _deskew(img: np.ndarray) -> np.ndarray:
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thr  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thr > 0))
        if coords.shape[0] < 50:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle += 90
        if abs(angle) < 0.5 or abs(angle) > 30:
            return img
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return img


def preprocess_for_ocr(img: np.ndarray, fast: bool = True) -> np.ndarray:
    """Adaptive preprocessing:
    - Upscale small images to 1200 px longest side (helps low-res scans).
    - Deskew up to ±30°.
    - Fast mode  : mild unsharp mask (≈0.05 s).
    - Full mode  : NLM denoise + CLAHE (≈1–3 s, for very poor scans)."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest < 1200:
        scale = min(1200.0 / max(longest, 1), 3.0)
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    img  = _deskew(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if fast:
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    else:
        gray = cv2.fastNlMeansDenoising(gray, None, h=10,
                                        templateWindowSize=7, searchWindowSize=21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ===========================================================================
# PaddleOCR line reconstruction
# ===========================================================================
def _group_into_lines(
    items: List[Tuple[float, float, float, str, float]]
) -> List[str]:
    if not items:
        return []
    heights = [it[2] for it in items if it[2] > 0]
    med_h   = float(np.median(heights)) if heights else 12.0
    y_tol   = max(8.0, med_h * 0.6)
    items   = sorted(items, key=lambda t: (t[0], t[1]))
    lines:    List[str]               = []
    current:  List[Tuple[float, str]] = []
    cur_y:    Optional[float]         = None
    for y_c, x_min, _h, text, _conf in items:
        if cur_y is None or abs(y_c - cur_y) <= y_tol:
            current.append((x_min, text))
            cur_y = y_c if cur_y is None else (cur_y + y_c) / 2.0
        else:
            current.sort(key=lambda t: t[0])
            lines.append(" ".join(t for _, t in current).strip())
            current = [(x_min, text)]
            cur_y   = y_c
    if current:
        current.sort(key=lambda t: t[0])
        lines.append(" ".join(t for _, t in current).strip())
    return [ln for ln in lines if ln]


# ===========================================================================
# OCR engine wrapper
# ===========================================================================
class OCREngine:
    def __init__(self) -> None:
        self.paddle: object        = None
        self.tesseract_ok: bool    = False
        self.active: Optional[str] = None
        self.last_used: Optional[str]            = None
        self.last_fallback_reason: Optional[str] = None

    # ---- init helpers ------------------------------------------------------
    def _build_paddle(self, mkldnn: bool) -> Optional[object]:
        try:
            ocr = PaddleOCR(
                use_angle_cls=False,
                lang=OCR_LANG,
                show_log=False,
                use_gpu=False,
                enable_mkldnn=mkldnn,
                cpu_threads=PADDLE_CPU_THREADS,
                det_limit_side_len=PADDLE_DET_LIMIT_SIDE_LEN,
                det_limit_type="max",
            )
            logger.info("✓ PaddleOCR ready (lang=%s mkldnn=%s threads=%s det=%s)",
                        OCR_LANG, mkldnn, PADDLE_CPU_THREADS, PADDLE_DET_LIMIT_SIDE_LEN)
            return ocr
        except Exception as exc:
            logger.warning("PaddleOCR init failed (mkldnn=%s): %s", mkldnn, exc)
            return None

    def init(self) -> Optional[str]:
        if OCR_ENGINE in ("paddle", "auto") and PaddleOCR is not None:
            mkldnn = PADDLE_ENABLE_MKLDNN and _IS_X86  # never enable on ARM
            self.paddle = self._build_paddle(mkldnn)
            if self.paddle is None and mkldnn:
                logger.warning("Retrying PaddleOCR without MKL-DNN")
                self.paddle = self._build_paddle(False)
            if self.paddle is not None:
                self.active = "paddle"

        if pytesseract is not None:
            try:
                pytesseract.get_tesseract_version()
                self.tesseract_ok = True
            except Exception:
                self.tesseract_ok = False

        if self.active is None and self.tesseract_ok:
            self.active = "tesseract"
        if OCR_REQUIRE_PADDLE and self.active != "paddle":
            raise RuntimeError(
                "PaddleOCR is required but failed to initialize. "
                "Check the paddlepaddle/paddleocr installation and model download access."
            )
        if self.active is None:
            logger.error("No OCR engine available — check PaddleOCR / Tesseract install")
        else:
            logger.info("✓ OCR engine ready (active=%s)", self.active)
        return self.active

    def ready(self) -> bool:
        return self.active is not None

    def info(self) -> dict:
        return {
            "requested_engine":            OCR_ENGINE,
            "paddle_required":             OCR_REQUIRE_PADDLE,
            "active_engine":               self.active,
            "last_used_engine":            self.last_used,
            "fallback_engine":             "tesseract" if self.tesseract_ok else None,
            "last_fallback_reason":        self.last_fallback_reason,
            "paddle_available":            PaddleOCR is not None,
            "paddle_initialized":          self.paddle is not None,
            "paddle_lang":                 OCR_LANG,
            "paddle_mkldnn":               PADDLE_ENABLE_MKLDNN and _IS_X86,
            "paddle_cpu_threads":          PADDLE_CPU_THREADS,
            "paddle_det_limit_side_len":   PADDLE_DET_LIMIT_SIDE_LEN,
            "tesseract_available":         self.tesseract_ok,
            "tesseract_lang":              TESS_LANG,
            "preprocess_enabled":          OCR_PREPROCESS,
            "preprocess_fast":             OCR_PREPROCESS_FAST,
            "low_quality_threshold":       OCR_MIN_CONF,
        }

    # ---- engine runners ----------------------------------------------------
    def _run_paddle(self, img: np.ndarray) -> Tuple[str, float, List[str]]:
        result = self.paddle.ocr(img, cls=False)
        if not result or result[0] is None:
            return "", 0.0, []
        items: List[Tuple[float, float, float, str, float]] = []
        confs: List[float] = []
        for entry in result[0]:
            try:
                box, (text, conf) = entry[0], entry[1]
            except Exception:
                continue
            text = str(text).strip()
            if not text:
                continue
            ys = [float(p[1]) for p in box]
            xs = [float(p[0]) for p in box]
            items.append((sum(ys)/len(ys), min(xs), max(ys)-min(ys), text, conf))
            confs.append(float(conf))
        lines = _group_into_lines(items)
        avg   = (sum(confs) / len(confs) * 100.0) if confs else 0.0
        return "\n".join(lines).strip(), avg, lines

    def _run_tesseract(self, img: np.ndarray) -> Tuple[str, float, List[str]]:
        if pytesseract is None or _PILImage is None:
            return "", 0.0, []
        pil  = _PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        data = pytesseract.image_to_data(
            pil, lang=TESS_LANG,
            config="--oem 3 --psm 6",
            output_type=pytesseract.Output.DICT,
        )
        lines: List[str]  = []
        cur:   List[str]  = []
        cur_key = None
        n = len(data.get("text", []))
        for i in range(n):
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            if cur_key is None:
                cur_key = key
            if key != cur_key:
                if cur:
                    lines.append(" ".join(cur))
                cur, cur_key = [], key
            w = str(data["text"][i]).strip()
            if w:
                cur.append(w)
        if cur:
            lines.append(" ".join(cur))
        lines = [ln for ln in lines if ln.strip()]
        text  = "\n".join(lines).strip()
        confs = [float(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit() and float(c) >= 0]
        avg   = (sum(confs) / len(confs)) if confs else 0.0
        return text, avg, lines

    def _run(self, img: np.ndarray) -> Tuple[str, float, List[str]]:
        if self.active == "paddle" and self.paddle is not None:
            try:
                self.last_used = "paddle"
                self.last_fallback_reason = None
                return self._run_paddle(img)
            except Exception as exc:
                self.last_fallback_reason = str(exc)
                logger.warning("PaddleOCR inference failed, falling back: %s", exc)
        if self.tesseract_ok:
            self.last_used = "tesseract"
            return self._run_tesseract(img)
        return "", 0.0, []

    # ---- public API --------------------------------------------------------
    def extract(
        self, img: np.ndarray, preprocess: bool = True
    ) -> Tuple[str, float, List[str]]:
        """Single-pass OCR with optional preprocessing.
        Retries on the raw image only if the first result is weak."""
        primary:   np.ndarray          = img
        secondary: Optional[np.ndarray] = None

        if preprocess and OCR_PREPROCESS:
            try:
                primary   = preprocess_for_ocr(img, fast=OCR_PREPROCESS_FAST)
                secondary = img
            except Exception as exc:
                logger.debug("preprocess error: %s", exc)

        text, conf, lines = self._run(primary)
        if conf >= OCR_MIN_CONF and len(lines) >= 3:
            return text, conf, lines

        if secondary is not None:
            t2, c2, l2 = self._run(secondary)
            s1 = conf * (1.0 + min(len(lines), 40) * 0.02)
            s2 = c2   * (1.0 + min(len(l2),    40) * 0.02)
            if s2 > s1:
                return t2, c2, l2
        return text, conf, lines


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_ENGINE: Optional[OCREngine] = None


def init_engine() -> Optional[str]:
    global _ENGINE
    _ENGINE = OCREngine()
    return _ENGINE.init()


def get_engine() -> Optional[OCREngine]:
    return _ENGINE


def normalize_for_search(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


# ===========================================================================
# Marksheet / document field extraction
# ===========================================================================
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|"
    "october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)

_NAME_LABELS = [
    "candidate's full name", "candidate's name", "candidate name",
    "name of the candidate", "name of candidate",
    "student's name", "student name", "name of the student", "name of student",
    "full name",
    "उमेदवाराचे संपूर्ण नाव", "उमेदवाराचे नाव",
    "विद्यार्थ्याचे नाव", "नाव", "नाम",
    "name",
]
_ROLL_LABELS = [
    "exam seat no", "seat no.", "seat no", "seat number",
    "roll no.", "roll no", "roll number",
    "registration no.", "registration no", "registration number",
    "reg. no.", "reg. no", "reg no",
    "enrollment no", "enrolment no",
    "hall ticket no", "admit card no",
]
_BOARD_PATTERNS = [
    r"(maharashtra state board[^\n]*)",
    r"(central board of secondary education[^\n]*)",
    r"(council for the indian school certificate[^\n]*)",
    r"(board of (?:secondary|higher secondary)[^\n]*)",
    r"([a-z .]{3,40} board of [a-z .]+)",
    r"(university of [a-z .]+)",
    r"([a-z .]{3,50} university)",
    r"\b(c\.?\s*b\.?\s*s\.?\s*e\.?)\b",
    r"\b(i\.?\s*c\.?\s*s\.?\s*e\.?)\b",
    r"\b(n\.?\s*i\.?\s*o\.?\s*s\.?)\b",
]
_EXAM_PATTERNS = [
    r"(all india senior school certificate exam\w*)",
    r"(higher secondary(?: school)? certificate[^\n]{0,60})",
    r"(secondary school certificate[^\n]{0,60})",
    r"(senior school certificate[^\n]{0,60})",
    r"(bachelor of [a-z .]+)",
    r"(master of [a-z .]+)",
    r"(diploma in [a-z .]+)",
    r"\b(h\.?\s*s\.?\s*c\.?)\b",
    r"\b(s\.?\s*s\.?\s*c\.?)\b",
]
_RESULT_PATTERNS = [
    r"\b(first class with distinction)\b",
    r"\b(distinction)\b",
    r"\b(first (?:class|division))\b",
    r"\b(second (?:class|division))\b",
    r"\b(third (?:class|division))\b",
    r"\b(pass(?:ed)?)\b",
    r"\b(fail(?:ed)?)\b",
]

# Common Indian subject keywords for better subject detection
_SUBJECT_KEYWORDS = {
    "english", "hindi", "marathi", "sanskrit", "mathematics", "maths",
    "science", "physics", "chemistry", "biology", "history", "geography",
    "civics", "economics", "commerce", "accounts", "accountancy",
    "computer", "informatics", "evs", "social",
    "गणित", "विज्ञान", "इतिहास", "भूगोल", "हिंदी", "मराठी",
}


def _clean(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s or "").strip()


def _after_label(
    text: str, labels: List[str], also_next_line: bool = True
) -> Optional[str]:
    lines = text.splitlines()
    for lab in labels:
        lab_l = lab.lower()
        for i, line in enumerate(lines):
            idx = line.lower().find(lab_l)
            if idx == -1:
                continue
            rest = line[idx + len(lab):]
            rest = re.sub(r"^[\s:\-.=|/()]+", "", rest).strip()
            rest = re.split(r"\s{3,}", rest)[0].strip()
            if rest and len(rest) > 1:
                return rest
            if also_next_line:
                for j in range(i + 1, min(i + 4, len(lines))):
                    nxt = lines[j].strip()
                    if nxt and len(nxt) > 1:
                        return nxt
    return None


def _first_match(text: str, patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return _clean(m.group(1))
    return None


def _extract_name(text: str) -> Optional[str]:
    val = _after_label(text, _NAME_LABELS, also_next_line=True)
    if not val:
        return None
    # Strip non-name characters; keep Latin, Devanagari, spaces, dots
    val = re.sub(r"[^A-Za-z\u0900-\u097F .]", " ", val)
    val = _clean(val)
    noise = {"surname", "first", "mother", "father", "full", "name", "candidate",
              "of", "the", "student", "roll", "seat", "no"}
    words = [w for w in val.split() if w.lower() not in noise]
    val = " ".join(words).strip()
    if len(re.sub(r"[ .]", "", val)) < 2:
        return None
    return val.title() if val.isupper() else val


def _extract_roll(text: str) -> Optional[str]:
    # Try same-line value after label
    val = _after_label(text, _ROLL_LABELS, also_next_line=False)
    if val:
        m = re.search(r"\b([A-Z0-9][A-Z0-9\-/]{2,15})\b", val, flags=re.IGNORECASE)
        if m and not m.group(0).isalpha():
            return m.group(0)

    # Scan lines containing seat/roll keywords for adjacent numbers
    for line in text.splitlines():
        low = line.lower()
        if any(k in low for k in ("seat no", "roll no", "seat number", "roll number")):
            nums = re.findall(r"\b(\d{4,12})\b", line)
            if nums:
                return nums[0]

    # Regex scan for number after keyword
    m = re.search(
        r"(?:seat\s*no|roll\s*no|enroll(?:ment)?(?:\s*no)?)[^\d]{0,25}(\d{4,12})",
        text, flags=re.IGNORECASE,
    )
    if m:
        return m.group(1)
    return None


def _extract_year(text: str) -> Optional[str]:
    m = re.search(rf"(?:{_MONTHS})[\s,.\-]*((?:19|20)\d{{2}})", text, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"year\s+of\s+exam\w*[^\d]{0,10}((?:19|20)\d{2})", text, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    all_years = re.findall(r"\b((?:19|20)\d{2})\b", text)
    return max(set(all_years), key=all_years.count) if all_years else None


def _extract_percentage(text: str) -> Optional[float]:
    cands: List[float] = []
    for m in re.finditer(
        r"(?:percentage|percent|per\s*cent)\s*[:\-]?\s*(\d{1,3}(?:\.\d{1,4})?)",
        text, flags=re.IGNORECASE,
    ):
        cands.append(float(m.group(1)))
    for m in re.finditer(r"(\d{1,3}(?:\.\d{1,4})?)\s*%", text):
        cands.append(float(m.group(1)))
    # 4-digit compressed percentage (e.g. "6317" → 63.17)
    for m in re.finditer(r"(?:percentage|percent)\D{0,10}(\d{4})\b", text, flags=re.IGNORECASE):
        raw = float(m.group(1)) / 100.0
        if 0 < raw <= 100:
            cands.append(round(raw, 2))
    valid = [c for c in cands if 0 < c <= 100]
    return valid[0] if valid else None


def _extract_cgpa(text: str) -> Optional[float]:
    m = re.search(
        r"\b(?:c\.?g\.?p\.?a|s\.?g\.?p\.?a|g\.?p\.?a)\b\D{0,8}(\d{1,2}(?:\.\d{1,2})?)",
        text, flags=re.IGNORECASE,
    )
    if m:
        v = float(m.group(1))
        if 0 < v <= 10:
            return v
    return None


def _extract_marks(text: str) -> Tuple[Optional[int], Optional[int]]:
    # Explicit label: "marks obtained ... / total"
    m = re.search(
        r"(?:grand\s*total|total\s*marks?\s*obtained|marks?\s*obtained|obtained\s*marks?)"
        r"\D{0,8}(\d{2,4})\s*(?:/|out\s*of)?\s*(\d{2,4})?",
        text, flags=re.IGNORECASE,
    )
    if m:
        return int(m.group(1)), int(m.group(2)) if m.group(2) else None

    # MSB layout: "Total Marks | 600 379"
    m = re.search(
        r"total\s*marks?[^\d]{0,10}(\d{2,4})[^\d]{0,5}(\d{2,4})",
        text, flags=re.IGNORECASE,
    )
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        total, obtained = (a, b) if a > b else (b, a)
        if obtained <= total:
            return obtained, total

    # x / y fractions
    _STD = {50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,900,1000}
    for mm in re.finditer(r"(\d{2,4})\s*/\s*(\d{2,4})", text):
        a, b = int(mm.group(1)), int(mm.group(2))
        if b in _STD and a <= b:
            return a, b

    # Two numbers close together, one a known total
    for mm in re.finditer(r"\b(\d{2,4})\b[^\d]{0,10}\b(\d{2,4})\b", text):
        a, b = int(mm.group(1)), int(mm.group(2))
        if b in _STD and a <= b:
            return a, b
        if a in _STD and b <= a:
            return b, a
    return None, None


def _extract_result(text: str) -> Optional[str]:
    val = _first_match(text, _RESULT_PATTERNS)
    if val:
        return val.title()
    m = re.search(
        r"\bgrade\b\D{0,5}([A-FO][+\-]?\d?|A1|A2|B1|B2|C1|C2)\b",
        text, flags=re.IGNORECASE,
    )
    if m:
        return f"Grade {m.group(1).upper()}"
    return None


def _extract_subjects(lines: List[str]) -> List[dict]:
    subjects: List[dict] = []
    _SKIP = {
        "total", "grand", "percentage", "result", "division", "name",
        "seat", "roll", "board", "university", "certificate", "signature",
        "date", "stream", "centre", "district", "school",
    }
    for line in lines:
        low = line.lower()
        if any(s in low for s in _SKIP):
            continue
        # Must contain a number (marks)
        nums = re.findall(r"\b(\d{1,3})\b", line)
        if not nums:
            continue
        # Extract leading name portion (Latin + Devanagari)
        name_m = re.match(r"([A-Za-z\u0900-\u097F .&()/-]{3,45})", line.strip())
        if not name_m:
            continue
        name = _clean(name_m.group(1))
        if len(re.sub(r"[ .]", "", name)) < 3:
            continue
        mark_list = [int(n) for n in nums if int(n) <= 100]
        if not mark_list:
            continue
        # Boost confidence if name contains known subject keyword
        is_subject = any(kw in name.lower() for kw in _SUBJECT_KEYWORDS)
        subjects.append({
            "name":        name,
            "marks":       mark_list[0],
            "all_numbers": [int(n) for n in nums],
            "confident":   is_subject,
        })
    # Sort confident entries first, then by order of appearance
    subjects.sort(key=lambda s: (not s["confident"],))
    return subjects[:20]


def extract_marksheet_fields(
    text: str, lines: Optional[List[str]] = None
) -> dict:
    text  = text or ""
    if lines is None:
        lines = [ln for ln in text.splitlines() if ln.strip()]

    obtained, total = _extract_marks(text)
    pct = _extract_percentage(text)
    if pct is None and obtained and total:
        pct = round(obtained / total * 100.0, 2)

    return {
        "name":             _extract_name(text),
        "roll_no":          _extract_roll(text),
        "board_university": _first_match(text, _BOARD_PATTERNS),
        "exam":             _first_match(text, _EXAM_PATTERNS),
        "year":             _extract_year(text),
        "obtained_marks":   obtained,
        "total_marks":      total,
        "percentage":       pct,
        "cgpa":             _extract_cgpa(text),
        "result":           _extract_result(text),
        "subjects":         _extract_subjects(lines),
    }
