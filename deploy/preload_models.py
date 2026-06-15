"""Download and validate all production models before starting the API."""

import os

from dotenv import load_dotenv
from insightface.app import FaceAnalysis
from paddleocr import PaddleOCR

load_dotenv()

ocr_lang = os.getenv("OCR_LANG", "devanagari")
print(f"Initializing PaddleOCR {ocr_lang} model...")
PaddleOCR(
    use_angle_cls=False,
    lang=ocr_lang,
    show_log=False,
    use_gpu=False,
)
print("PaddleOCR model is ready.")

print("Initializing InsightFace buffalo_l model...")
face = FaceAnalysis(name="buffalo_l")
face.prepare(ctx_id=0, det_size=(640, 640))
print("InsightFace model is ready.")
