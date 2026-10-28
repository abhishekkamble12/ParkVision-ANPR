"""
=============================================================
  ParkVision — Single Image Pipeline Demo
=============================================================
Run this first to test the full pipeline end-to-end
without needing FastAPI or Streamlit running.

Usage:
    python scripts/run_image.py
    python scripts/run_image.py --image path/to/car.jpg
    python scripts/run_image.py --debug   (saves preprocessing stages)
"""

import argparse
import os
import sys
import urllib.request

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.detector import PlateDetector
from core.preprocessor import preprocess_plate, save_debug_stages
from core.ocr_engine import PlateOCREngine
from parking.session_manager import ParkingSessionManager

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "output_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Sample Indian Number Plate Images (public domain / CC0) ──────────────────
SAMPLE_IMAGES = {
    "test_car_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/2019_Honda_City_1.5_V%2B_CVT_%28facelift%2C_red%29%2C_front_8.21.19.jpg/1280px-2019_Honda_City_1.5_V%2B_CVT_%28facelift%2C_red%29%2C_front_8.21.19.jpg",
    "test_car_2.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/2018_Honda_City_1.5_E_%28facelift%29%2C_front_9.12.19.jpg/1280px-2018_Honda_City_1.5_E_%28facelift%29%2C_front_9.12.19.jpg",
}


def download_sample_images() -> list[str]:
    """Downloads sample car images if not already present."""
    paths = []
    for filename, url in SAMPLE_IMAGES.items():
        fpath = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(fpath):
            try:
                print(f"  Downloading {filename}...")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp, open(fpath, "wb") as f:
                    f.write(resp.read())
                print(f"  ✅ Saved: {fpath}")
            except Exception as e:
                print(f"  ⚠️  Could not download {filename}: {e}")
                # Synthesize a test image with a white plate region
                fpath = _make_synthetic_car_image(fpath)
        paths.append(fpath)
    return paths


def _make_synthetic_car_image(save_path: str) -> str:
    """Creates a simple synthetic test image with a visible plate rectangle."""
    import numpy as np
    img = np.full((400, 640, 3), (80, 90, 100), dtype=int_type := __import__("numpy").uint8)
    # Plate area (white rectangle bottom center)
    cv2.rectangle(img, (230, 320), (420, 370), (255, 255, 255), -1)
    cv2.rectangle(img, (230, 320), (420, 370), (0, 0, 0), 2)
    # Fake plate text
    cv2.putText(img, "MH12AB1234", (235, 358), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.imwrite(save_path, img)
    print(f"  ✅ Created synthetic test image: {save_path}")
    return save_path


def run_pipeline(image_path: str, debug: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"  ParkVision — Image Pipeline")
    print(f"  Input: {os.path.basename(image_path)}")
    print(f"{'='*60}")

    # ── Load image ─────────────────────────────────────────────────────────────
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  ERROR: Could not load image: {image_path}")
        return

    print(f"  Image size: {frame.shape[1]}×{frame.shape[0]}")

    # ── Step 1: Detect Plates ──────────────────────────────────────────────────
    print("\n[1] Running YOLOv8 plate detection...")
    detector = PlateDetector(confidence_threshold=0.35)
    detections = detector.detect(frame)
    print(f"     Detected {len(detections)} plate(s)")

    if not detections:
        print("  ⚠️  No plates found. Try a clearer image or lower confidence threshold.")
        return

    # ── Step 2: Preprocess + OCR ───────────────────────────────────────────────
    print("\n[2] Running OCR pipeline...")
    ocr = PlateOCREngine(use_gpu=False)
    parking_mgr = ParkingSessionManager()

    plate_texts = []
    for i, det in enumerate(detections):
        print(f"\n  -- Plate {i+1} (conf: {det.confidence:.2f}) --")

        # Preprocess
        debug_dir = os.path.join(OUTPUT_DIR, f"debug_plate_{i+1}") if debug else None
        processed, stages = preprocess_plate(det.plate_crop, debug=debug)
        if debug and stages:
            save_debug_stages(stages, debug_dir, prefix="stage")

        # OCR
        result = ocr.read_plate(det.plate_crop, preprocessed=processed)
        print(f"  Raw OCR text:    '{result.raw_text}'")
        print(f"  Cleaned text:    '{result.cleaned_text}'")
        print(f"  OCR Confidence:  {result.confidence * 100:.1f}%")
        print(f"  Valid IND plate: {'✅ YES' if result.is_valid_ind else '❌ NO'}")
        plate_texts.append(result.cleaned_text)

        # Log to DB
        parking_mgr.log_detection(
            plate_number=result.cleaned_text or "UNKNOWN",
            raw_text=result.raw_text,
            confidence=result.confidence,
            source="image",
            is_valid_ind=result.is_valid_ind
        )

        # Auto-log parking entry for valid plates
        if result.is_valid_ind and result.confidence > 0.5:
            parking_mgr.log_entry(
                plate_number=result.cleaned_text,
                confidence=result.confidence
            )

    # ── Step 3: Annotate and save ──────────────────────────────────────────────
    print("\n[3] Saving annotated output...")
    annotated = detector.annotate_frame(frame, detections, plate_texts)

    base = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"result_{base}.jpg")
    cv2.imwrite(output_path, annotated)
    print(f"  ✅ Saved → {output_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    stats = parking_mgr.get_stats()
    print(f"  Sessions in DB:  {stats['total_sessions']}")
    print(f"  Detections logged: {stats['total_detections']}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="ParkVision Image Detection Demo")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image. Defaults to downloaded samples.")
    parser.add_argument("--debug", action="store_true",
                        help="Save all preprocessing stage images for inspection.")
    args = parser.parse_args()

    if args.image:
        run_pipeline(args.image, debug=args.debug)
    else:
        print("No --image provided. Downloading sample images...")
        image_paths = download_sample_images()
        for path in image_paths:
            run_pipeline(path, debug=args.debug)


if __name__ == "__main__":
    main()
