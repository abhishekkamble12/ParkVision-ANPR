"""
=============================================================
  ParkVision Core — OpenCV Preprocessing Pipeline
=============================================================
Prepares a raw cropped plate image for OCR by:
  1. Grayscale conversion
  2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
  3. Gaussian Blur (noise removal)
  4. Adaptive Threshold (handles shadows, uneven lighting)
  5. Morphological dilation (fills character gaps)
  6. Deskew via Hough lines (corrects tilt)

These are the 6 stages to explain confidently in any interview!
"""

import cv2
import numpy as np


def preprocess_plate(plate_img: np.ndarray, debug: bool = False) -> tuple[np.ndarray, dict]:
    """
    Full preprocessing pipeline for a cropped number plate image.

    Args:
        plate_img: BGR image of the cropped plate region.
        debug:     If True, returns all intermediate stage images.

    Returns:
        processed_img: Binary image ready for OCR.
        stages:        Dict of {stage_name: image} for debugging.
    """
    stages = {}

    # ─── Step 1: Resize to standard height for consistent OCR ────────────────
    # EasyOCR performs best when text height is 30-50 px
    h, w = plate_img.shape[:2]
    target_height = 64
    scale = target_height / h
    resized = cv2.resize(plate_img, (int(w * scale), target_height), interpolation=cv2.INTER_CUBIC)
    stages["1_resized"] = resized

    # ─── Step 2: Grayscale ───────────────────────────────────────────────────
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    stages["2_gray"] = gray

    # ─── Step 3: CLAHE — Adaptive contrast enhancement ───────────────────────
    # Works much better than global histogram equalization for plates in
    # varied lighting (bright sun, nighttime with IR cameras, shadows)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    stages["3_clahe"] = enhanced

    # ─── Step 4: Gaussian Blur (3x3 kernel removes high-freq noise) ──────────
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    stages["4_blurred"] = blurred

    # ─── Step 5: Adaptive Thresholding ───────────────────────────────────────
    # ADAPTIVE_THRESH_GAUSSIAN_C: computes threshold per 11×11 pixel block.
    # This handles reflections and uneven illumination across the plate.
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=4
    )
    stages["5_binary"] = binary

    # ─── Step 6: Morphological closing (fills gaps in broken characters) ─────
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    stages["6_closed"] = closed

    # ─── Step 7: Deskew using Hough lines ────────────────────────────────────
    deskewed = _deskew(closed)
    stages["7_deskewed"] = deskewed

    if debug:
        return deskewed, stages
    return deskewed, {}


def _deskew(binary_img: np.ndarray) -> np.ndarray:
    """
    Detects the dominant text angle via Hough Line Transform
    and corrects tilt up to ±10 degrees.
    """
    edges = cv2.Canny(binary_img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                            minLineLength=30, maxLineGap=10)
    if lines is None:
        return binary_img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 10:  # Only correct small tilts
            angles.append(angle)

    if not angles:
        return binary_img

    median_angle = np.median(angles)
    h, w = binary_img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    deskewed = cv2.warpAffine(binary_img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return deskewed


def save_debug_stages(stages: dict, output_dir: str, prefix: str = "plate") -> None:
    """Saves all preprocessing stage images to output_dir for visual inspection."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    for name, img in stages.items():
        path = os.path.join(output_dir, f"{prefix}_{name}.png")
        cv2.imwrite(path, img)
    print(f"  [Preprocessor] Saved {len(stages)} debug stage images → {output_dir}/")
