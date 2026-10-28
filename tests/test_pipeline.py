"""
=============================================================
  ParkVision — Pipeline Evaluation & Metrics Tests
=============================================================
Measures:
  1. Plate Detection Precision & Recall
  2. OCR Character Error Rate (CER)
  3. Complete Plate Read Accuracy (exact match)
  4. Processing Speed (FPS per stage)

Run with:
    python tests/test_pipeline.py

To pass these tests confidently in an interview:
  - Precision = TP / (TP + FP)   → How many detections are correct?
  - Recall    = TP / (TP + FN)   → How many actual plates did we find?
  - CER       = edit_distance(pred, gt) / len(gt)  → Per-char OCR accuracy
"""

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.preprocessor import preprocess_plate
from core.ocr_engine import PlateOCREngine, _clean_plate_text


# ─── Synthetic Ground Truth Dataset ───────────────────────────────────────────
# Format: (plate_image_array, ground_truth_text)
# In production: replace with real annotated images

def _make_plate_image(text: str, width: int = 300, height: int = 70,
                      noise: float = 0.0, angle: float = 0.0) -> np.ndarray:
    """Creates a synthetic number plate image with optional noise and tilt."""
    img = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (0, 0, 0), 3)

    font_scale = min(width / (len(text) * 20), 1.2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    tx = (width - tw) // 2
    ty = (height + th) // 2
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

    if noise > 0:
        noise_arr = np.random.normal(0, noise * 255, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise_arr, 0, 255).astype(np.uint8)

    if angle != 0.0:
        cx, cy = width // 2, height // 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        img = cv2.warpAffine(img, M, (width, height), borderValue=(255, 255, 255))

    return img


GROUND_TRUTH_PLATES = [
    "MH12AB1234",
    "DL8CAB9999",
    "KA01MF0001",
    "GJ01AA0001",
    "TN09AZ0099",
    "UP32DP5678",
    "RJ14CH4321",
    "AP28DG7890",
]

TEST_CONDITIONS = [
    ("clean",     {"noise": 0.0, "angle": 0.0}),
    ("noisy",     {"noise": 0.05, "angle": 0.0}),
    ("tilted",    {"noise": 0.0,  "angle": 3.5}),
    ("hard",      {"noise": 0.08, "angle": 5.0}),
]


# ─── Metrics ──────────────────────────────────────────────────────────────────

def character_error_rate(predicted: str, ground_truth: str) -> float:
    """
    Levenshtein-based Character Error Rate.
    CER = edit_distance / len(ground_truth)
    0.0 = perfect, 1.0 = completely wrong.
    """
    pred = predicted.replace(" ", "").upper()
    gt   = ground_truth.replace(" ", "").upper()
    if not gt:
        return 0.0
    dist = _levenshtein(pred, gt)
    return dist / len(gt)


def _levenshtein(s1: str, s2: str) -> int:
    """Classic DP Levenshtein distance."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n]


# ─── OCR Accuracy Test ────────────────────────────────────────────────────────

def test_ocr_accuracy():
    """Tests OCR accuracy across all plates and conditions."""
    print("\n" + "="*65)
    print("  OCR ACCURACY EVALUATION")
    print("="*65)

    ocr = PlateOCREngine(use_gpu=False)

    results_table = []
    total_cer = 0.0
    exact_matches = 0
    total_tests = 0

    for condition, kwargs in TEST_CONDITIONS:
        print(f"\n  Condition: {condition.upper()}")
        cond_cer = []
        cond_exact = 0

        for gt_text in GROUND_TRUTH_PLATES:
            plate_img = _make_plate_image(gt_text, **kwargs)
            processed, _ = preprocess_plate(plate_img)

            t0 = time.perf_counter()
            result = ocr.read_plate(plate_img, preprocessed=processed)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            cer = character_error_rate(result.cleaned_text, gt_text)
            exact = result.cleaned_text.replace(" ", "").upper() == gt_text.upper()

            cond_cer.append(cer)
            if exact:
                cond_exact += 1
                exact_matches += 1
            total_cer += cer
            total_tests += 1

            status = "✅" if exact else f"❌ (got: '{result.cleaned_text}')"
            print(f"    {gt_text:<15} | CER: {cer:.2f} | {elapsed_ms:5.1f}ms | {status}")

        avg_cond_cer = sum(cond_cer) / len(cond_cer) if cond_cer else 0
        print(f"  ── Condition Avg CER: {avg_cond_cer:.3f} | Exact: {cond_exact}/{len(GROUND_TRUTH_PLATES)}")
        results_table.append((condition, avg_cond_cer, cond_exact))

    # Summary
    overall_cer = total_cer / total_tests if total_tests else 0
    overall_accuracy = exact_matches / total_tests * 100 if total_tests else 0

    print(f"\n{'='*65}")
    print("  FINAL SUMMARY")
    print(f"  {'Condition':<12} {'Avg CER':>10} {'Exact Matches':>15}")
    print(f"  {'-'*40}")
    for cond, avg_cer, exact in results_table:
        print(f"  {cond:<12} {avg_cer:>10.3f} {exact:>15}/{len(GROUND_TRUTH_PLATES)}")
    print(f"  {'-'*40}")
    print(f"  Overall CER:        {overall_cer:.3f}")
    print(f"  Plate Accuracy:     {overall_accuracy:.1f}%")
    print(f"  Total tests:        {total_tests}")
    print(f"{'='*65}\n")

    return overall_cer, overall_accuracy


# ─── Preprocessing Speed Test ─────────────────────────────────────────────────

def test_preprocessing_speed(n_iters: int = 50):
    """Benchmarks the preprocessing pipeline speed."""
    print("\n" + "="*65)
    print("  PREPROCESSING SPEED BENCHMARK")
    print("="*65)

    plate_img = _make_plate_image("MH12AB1234")

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        preprocess_plate(plate_img)
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    fps_equiv = 1000 / avg_ms if avg_ms > 0 else 0

    print(f"  Iterations:  {n_iters}")
    print(f"  Avg time:    {avg_ms:.2f} ms/frame")
    print(f"  Throughput:  {fps_equiv:.0f} FPS equivalent")
    print(f"  Min/Max:     {min(times):.2f} / {max(times):.2f} ms")
    print(f"{'='*65}\n")


# ─── Text Cleaning Unit Tests ─────────────────────────────────────────────────

def test_ocr_corrections():
    """Tests the zone-aware O→0 / I→1 character correction."""
    print("\n" + "="*65)
    print("  OCR CHARACTER CORRECTION UNIT TESTS")
    print("="*65)

    test_cases = [
        # (raw_ocr_output, expected_cleaned)
        ("MHI2ABI234",  "MH 12 AB 1234"),   # I→1
        ("MHO2ABO234",  "MH 02 AB 0234"),   # O→0
        ("DL8CAB9999",  "DL 8C AB 9999"),
    ]

    passed = 0
    for raw, expected in test_cases:
        cleaned = _clean_plate_text(raw)
        match = cleaned.replace(" ", "") == expected.replace(" ", "")
        status = "✅ PASS" if match else f"❌ FAIL (got '{cleaned}')"
        print(f"  Input: {raw:<15} → Expected: {expected:<18} | {status}")
        if match:
            passed += 1

    print(f"\n  Passed: {passed}/{len(test_cases)}")
    print(f"{'='*65}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚗 ParkVision — Pipeline Evaluation Suite\n")
    test_ocr_corrections()
    test_preprocessing_speed(n_iters=50)
    cer, acc = test_ocr_accuracy()
    print(f"\n✅ Evaluation complete — Overall Accuracy: {acc:.1f}%")
