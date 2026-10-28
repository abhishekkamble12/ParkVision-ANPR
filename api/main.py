"""
=============================================================
  ParkVision — FastAPI Backend
=============================================================
REST API endpoints:
  POST /detect/image        Upload an image, get plate + OCR result
  POST /parking/entry       Manually log a vehicle entry
  POST /parking/exit        Manually log a vehicle exit
  GET  /parking/sessions    List all sessions
  GET  /parking/active      List currently active sessions
  GET  /parking/stats       Dashboard statistics
  GET  /detections/log      Raw detection event log
  GET  /health              Health check

Run with:
    uvicorn api.main:app --reload --port 8000
"""

import io
import os
import sys
import uuid
import datetime

# Add project root to path so imports work from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.detector import PlateDetector
from core.preprocessor import preprocess_plate
from core.ocr_engine import PlateOCREngine
from parking.session_manager import ParkingSessionManager

# ─── App Init ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ParkVision API",
    description="Indian Smart Parking & ANPR System — REST API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Globals (loaded once at startup) ────────────────────────────────────────

_detector: PlateDetector = None
_ocr: PlateOCREngine = None
_parking_mgr: ParkingSessionManager = None
_CROPS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "crops")
os.makedirs(_CROPS_DIR, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    global _detector, _ocr, _parking_mgr
    print("[API] Loading models...")
    _detector = PlateDetector(confidence_threshold=0.40)
    _ocr = PlateOCREngine(use_gpu=False)
    _parking_mgr = ParkingSessionManager()
    print("[API] ✅ All models loaded. Server ready.")


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class PlateDetectionResponse(BaseModel):
    plate_number: str
    confidence: float
    raw_ocr: str
    is_valid_indian_plate: bool
    bbox: list[int]             # [x1, y1, x2, y2]
    crop_saved_path: str = ""

class ParkingEntryRequest(BaseModel):
    plate_number: str
    slot_id: str = None

class ParkingExitRequest(BaseModel):
    plate_number: str


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.datetime.utcnow().isoformat()}


@app.post("/detect/image", response_model=list[PlateDetectionResponse])
async def detect_from_image(file: UploadFile = File(...)):
    """
    Upload a vehicle image (JPG/PNG).
    Returns detected plates with OCR text, confidence, and bounding box.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPG or PNG)")

    # Read image bytes → numpy array
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image")

    # ── Detect plates ──────────────────────────────────────────────────────────
    detections = _detector.detect(frame)
    if not detections:
        return []

    results = []
    for det in detections:
        # ── Preprocess crop ────────────────────────────────────────────────────
        processed, _ = preprocess_plate(det.plate_crop)

        # ── OCR ────────────────────────────────────────────────────────────────
        ocr_result = _ocr.read_plate(det.plate_crop, preprocessed=processed)

        # ── Save crop ──────────────────────────────────────────────────────────
        crop_path = ""
        if det.plate_crop is not None and det.plate_crop.size > 0:
            crop_name = f"{uuid.uuid4().hex[:8]}_{ocr_result.cleaned_text.replace(' ', '_')}.jpg"
            crop_path = os.path.join(_CROPS_DIR, crop_name)
            cv2.imwrite(crop_path, det.plate_crop)

        # ── Log detection ──────────────────────────────────────────────────────
        _parking_mgr.log_detection(
            plate_number=ocr_result.cleaned_text or "UNKNOWN",
            raw_text=ocr_result.raw_text,
            confidence=ocr_result.confidence,
            source="image",
            is_valid_ind=ocr_result.is_valid_ind
        )

        results.append(PlateDetectionResponse(
            plate_number=ocr_result.cleaned_text,
            confidence=round(ocr_result.confidence, 4),
            raw_ocr=ocr_result.raw_text,
            is_valid_indian_plate=ocr_result.is_valid_ind,
            bbox=list(det.bbox),
            crop_saved_path=crop_path,
        ))

    return results


@app.post("/parking/entry")
def log_parking_entry(req: ParkingEntryRequest):
    """Log a vehicle entry. Returns the new session."""
    session = _parking_mgr.log_entry(
        plate_number=req.plate_number,
        slot_id=req.slot_id
    )
    if session is None:
        return JSONResponse({"detail": "Duplicate suppressed (cooldown active)"}, status_code=200)
    return session


@app.post("/parking/exit")
def log_parking_exit(req: ParkingExitRequest):
    """Log a vehicle exit. Returns the closed session with duration."""
    session = _parking_mgr.log_exit(plate_number=req.plate_number)
    if session is None:
        raise HTTPException(404, f"No active session found for {req.plate_number}")
    return session


@app.get("/parking/sessions")
def get_all_sessions(limit: int = 100):
    """Returns recent parking sessions (newest first)."""
    return _parking_mgr.get_all_sessions(limit=limit)


@app.get("/parking/active")
def get_active_sessions():
    """Returns all currently parked vehicles."""
    return _parking_mgr.get_active_sessions()


@app.get("/parking/stats")
def get_stats():
    """Returns summary stats for the Streamlit dashboard."""
    return _parking_mgr.get_stats()


@app.get("/detections/log")
def get_detection_log(limit: int = 200):
    """Returns raw OCR detection events."""
    return _parking_mgr.get_detection_logs(limit=limit)
