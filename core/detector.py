"""
=============================================================
  ParkVision Core — YOLOv8 Number Plate Detector
=============================================================
Uses two detection strategies:

  Strategy A (Recommended): License-plate-specific YOLO model
    - Downloads a fine-tuned plate detector from Hugging Face
    - Detects only number plates (1-class model = very fast)

  Strategy B (Fallback): YOLOv8 generic COCO model
    - Detects 'car' (class 2) and 'truck' (class 7)
    - Then searches for plate-like rectangles inside vehicle ROI
    - Uses contour analysis (aspect ratio + area filtering)

This dual-strategy approach is a great interview talking point!
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PlateDetection:
    """Represents a single number plate detection result."""
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) in pixels
    confidence: float                   # 0.0 – 1.0
    plate_crop: np.ndarray             # Cropped BGR plate image
    frame_idx: Optional[int] = None    # Frame number (video mode)


class PlateDetector:
    """
    YOLOv8-based number plate detector.

    Usage:
        detector = PlateDetector()
        detections = detector.detect(frame)
        for d in detections:
            print(d.bbox, d.confidence)
    """

    # Plate-specific YOLO model (fine-tuned on Indian / global plates)
    # Source: keremberke/license-plate-object-detection (Roboflow HF)
    PLATE_MODEL_NAME = "keremberke/yolov8n-license-plate-detection"

    def __init__(self, confidence_threshold: float = 0.40, use_plate_model: bool = True):
        """
        Args:
            confidence_threshold: Minimum detection confidence (0-1).
            use_plate_model:      Use plate-specific model vs generic YOLO.
        """
        self.conf_threshold = confidence_threshold
        self.model = None
        self._load_model(use_plate_model)

    def _load_model(self, use_plate_model: bool) -> None:
        """Loads the YOLO model, falling back gracefully if unavailable."""
        try:
            from ultralytics import YOLO
        except ImportError:
            print("  [Detector] ERROR: 'ultralytics' not installed. Run: pip install ultralytics")
            return

        if use_plate_model:
            try:
                # Try loading the plate-specific fine-tuned model
                self.model = YOLO(self.PLATE_MODEL_NAME)
                self.model_type = "plate_specific"
                print(f"  [Detector] ✅ Loaded plate-specific model: {self.PLATE_MODEL_NAME}")
                return
            except Exception as e:
                print(f"  [Detector] ⚠️  Plate model unavailable ({e}). Falling back to YOLOv8n.")

        # Fallback: Generic YOLOv8 Nano (downloads ~6MB if not cached)
        self.model = YOLO("yolov8n.pt")
        self.model_type = "generic_yolo"
        print("  [Detector] ✅ Loaded generic YOLOv8n (fallback mode)")

    def detect(self, frame: np.ndarray) -> list[PlateDetection]:
        """
        Runs plate detection on a single BGR frame.

        Returns:
            List of PlateDetection objects (sorted by confidence, highest first).
        """
        if self.model is None:
            return []

        if self.model_type == "plate_specific":
            return self._detect_with_plate_model(frame)
        else:
            return self._detect_with_generic_yolo(frame)

    def _detect_with_plate_model(self, frame: np.ndarray) -> list[PlateDetection]:
        """Direct plate detection using fine-tuned YOLO model."""
        results = self.model.predict(
            source=frame, conf=self.conf_threshold, verbose=False
        )
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                detections.append(PlateDetection(
                    bbox=(x1, y1, x2, y2), confidence=conf, plate_crop=crop
                ))
        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    def _detect_with_generic_yolo(self, frame: np.ndarray) -> list[PlateDetection]:
        """
        Fallback: Find vehicles → search for plate-like rectangles inside ROI.
        Plate heuristic: aspect ratio 2:1 to 6:1, area > 500 px².
        """
        results = self.model.predict(
            source=frame, conf=0.35, verbose=False, classes=[2, 7]  # car, truck
        )
        detections = []

        for box in results[0].boxes:
            vx1, vy1, vx2, vy2 = map(int, box.xyxy[0].cpu().numpy())
            # Look for plate in the bottom 40% of the vehicle bounding box
            search_y1 = vy1 + int((vy2 - vy1) * 0.55)
            vehicle_roi = frame[search_y1:vy2, vx1:vx2]

            if vehicle_roi.size == 0:
                continue

            plate_detections = _find_plate_contours(vehicle_roi)
            for (px1, py1, px2, py2), conf in plate_detections:
                # Convert back to full-frame coordinates
                abs_x1, abs_y1 = vx1 + px1, search_y1 + py1
                abs_x2, abs_y2 = vx1 + px2, search_y1 + py2
                crop = frame[abs_y1:abs_y2, abs_x1:abs_x2]
                if crop.size > 0:
                    detections.append(PlateDetection(
                        bbox=(abs_x1, abs_y1, abs_x2, abs_y2),
                        confidence=conf,
                        plate_crop=crop
                    ))
        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    def annotate_frame(self, frame: np.ndarray,
                       detections: list[PlateDetection],
                       plate_texts: Optional[list[str]] = None) -> np.ndarray:
        """
        Draws bounding boxes and optional OCR text on frame.

        Args:
            frame:       BGR frame to annotate.
            detections:  List of plate detections.
            plate_texts: OCR-read text for each detection (optional).

        Returns:
            Annotated frame copy.
        """
        annotated = frame.copy()
        color = (0, 255, 100)    # Bright green
        text_bg = (0, 60, 0)

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox

            # ─── Bounding box ─────────────────────────────────────────────
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # ─── Corner accent marks (premium look) ───────────────────────
            _draw_corner_marks(annotated, x1, y1, x2, y2, color)

            # ─── Label badge ──────────────────────────────────────────────
            conf_pct = f"{det.confidence * 100:.0f}%"
            if plate_texts and i < len(plate_texts):
                label = f"  {plate_texts[i]}  [{conf_pct}]"
            else:
                label = f"  PLATE [{conf_pct}]"

            (fw, fh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            badge_y1 = max(0, y1 - fh - 10)
            cv2.rectangle(annotated, (x1, badge_y1), (x1 + fw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        return annotated


# ─── Helper: Corner decorations ───────────────────────────────────────────────
def _draw_corner_marks(img, x1, y1, x2, y2, color, length=15, thickness=3):
    """Draws small L-shaped corner marks inside each corner of a bounding box."""
    corners = [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]
    for cx, cy, dx, dy in corners:
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, thickness)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, thickness)


# ─── Helper: Contour-based plate search (fallback) ────────────────────────────
def _find_plate_contours(roi: np.ndarray) -> list[tuple]:
    """
    Finds plate-like rectangles within an ROI using contour analysis.
    Returns list of ((x1, y1, x2, y2), confidence) tuples.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.Canny(blur, 50, 200)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        aspect_ratio = w / h
        if 2.0 <= aspect_ratio <= 6.5:  # Indian plate: ~4.5:1
            # Confidence heuristic: how close to ideal 4.5:1 ratio
            ideal = 4.5
            conf = 1.0 - abs(aspect_ratio - ideal) / ideal
            conf = max(0.3, conf)
            results.append(((x, y, x + w, y + h), conf))

    return sorted(results, key=lambda r: r[1], reverse=True)[:3]  # Top 3
