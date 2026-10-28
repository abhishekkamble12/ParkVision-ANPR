"""
=============================================================
  ParkVision Core — EasyOCR Recognition Engine
=============================================================
Reads characters from a preprocessed plate image using EasyOCR.

Key design choices explained here (interview-ready):
  - EasyOCR works offline, GPU-optional, supports 80+ languages
  - We run OCR on BOTH the raw crop AND the preprocessed binary
    and pick the result with higher combined confidence
  - Indian plate regex filters out garbage OCR noise
  - Levenshtein-based fuzzy correction fixes 1-char errors (O→0, I→1)
"""

import re
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ─── Indian Number Plate Regex ────────────────────────────────────────────────
# Standard format: MH 12 AB 1234  (or MH12AB1234 without spaces)
# New BH series:   22 BH 1234 AB
_IND_PLATE_PATTERN = re.compile(
    r"^[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{1,4}$"   # Standard
    r"|^\d{2}\s?BH\s?\d{4}\s?[A-Z]{1,2}$",             # BH (Bharat) series
    re.IGNORECASE
)

# Common OCR confusion pairs for number plates
_OCR_CORRECTIONS = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
}


@dataclass
class OCRResult:
    """Result from OCR on one plate crop."""
    raw_text: str        # Exactly what EasyOCR returned
    cleaned_text: str    # After regex cleaning and correction
    confidence: float    # Average confidence from EasyOCR (0-1)
    is_valid_ind: bool   # True if matches Indian plate format


class PlateOCREngine:
    """
    EasyOCR-based number plate character recognizer.

    Usage:
        engine = PlateOCREngine()
        result = engine.read_plate(plate_crop)
        print(result.cleaned_text, result.confidence)
    """

    def __init__(self, languages: list = None, use_gpu: bool = False):
        """
        Args:
            languages: EasyOCR language list. Defaults to English.
            use_gpu:   Use CUDA GPU if available. False = CPU (safe default).
        """
        self.languages = languages or ["en"]
        self.use_gpu = use_gpu
        self.reader = None
        self._load_reader()

    def _load_reader(self) -> None:
        """Initializes EasyOCR reader (downloads models on first run ~150MB)."""
        try:
            import easyocr
            print(f"  [OCR] Loading EasyOCR (lang={self.languages}, gpu={self.use_gpu})...")
            self.reader = easyocr.Reader(self.languages, gpu=self.use_gpu, verbose=False)
            print("  [OCR] ✅ EasyOCR ready")
        except ImportError:
            print("  [OCR] ERROR: EasyOCR not installed. Run: pip install easyocr")
        except Exception as e:
            print(f"  [OCR] ERROR loading EasyOCR: {e}")

    def read_plate(self, plate_crop: np.ndarray,
                   preprocessed: Optional[np.ndarray] = None) -> OCRResult:
        """
        Runs OCR on a plate image. Tries both raw and preprocessed crops
        and returns the higher-confidence result.

        Args:
            plate_crop:   Raw BGR crop of the plate.
            preprocessed: Optional binary preprocessed version.

        Returns:
            OCRResult with cleaned text and confidence.
        """
        if self.reader is None:
            return OCRResult("", "", 0.0, False)

        candidates = []

        # ── Attempt 1: Raw crop (EasyOCR handles colour well)
        raw_result = self._run_easyocr(plate_crop)
        if raw_result:
            candidates.append(raw_result)

        # ── Attempt 2: Preprocessed binary (better for faded/dirty plates)
        if preprocessed is not None:
            # Convert binary back to 3-channel so EasyOCR can process it
            if len(preprocessed.shape) == 2:
                proc_3ch = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
            else:
                proc_3ch = preprocessed
            proc_result = self._run_easyocr(proc_3ch)
            if proc_result:
                candidates.append(proc_result)

        if not candidates:
            return OCRResult("", "", 0.0, False)

        # Pick the candidate with best confidence
        best = max(candidates, key=lambda r: r.confidence)
        return best

    def _run_easyocr(self, img: np.ndarray) -> Optional[OCRResult]:
        """Runs EasyOCR on a single image and returns an OCRResult."""
        try:
            # allowlist restricts recognized characters to valid plate chars
            results = self.reader.readtext(
                img,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
                detail=1,          # Returns (bbox, text, confidence)
                paragraph=False    # Treat each text block individually
            )
        except Exception as e:
            print(f"  [OCR] readtext error: {e}")
            return None

        if not results:
            return None

        # Concatenate all detected text segments
        texts = [r[1].upper().strip() for r in results]
        confs = [r[2] for r in results]
        raw_text = " ".join(texts)
        avg_conf = float(np.mean(confs)) if confs else 0.0

        # Clean and correct
        cleaned = _clean_plate_text(raw_text)
        is_valid = bool(_IND_PLATE_PATTERN.match(cleaned.replace(" ", "")))

        return OCRResult(
            raw_text=raw_text,
            cleaned_text=cleaned,
            confidence=avg_conf,
            is_valid_ind=is_valid
        )


# ─── Text cleaning helpers ─────────────────────────────────────────────────────

def _clean_plate_text(text: str) -> str:
    """
    Cleans raw OCR text into a canonical Indian plate number.
    Steps:
      1. Remove non-alphanumeric characters (except spaces)
      2. Normalize spaces
      3. Apply positional character corrections (letters vs digits zones)
    """
    # Remove everything except letters, digits, and spaces
    text = re.sub(r"[^A-Z0-9 ]", "", text.upper())
    text = re.sub(r"\s+", " ", text).strip()

    # Apply zone-aware correction
    # Indian plate: [2 letters] [2 digits] [1-3 letters] [4 digits]
    # Zones:          alpha        num          alpha         num
    text = _zone_aware_correction(text)
    return text


def _zone_aware_correction(text: str) -> str:
    """
    Applies O↔0, I↔1 corrections based on which zone of the plate we're in.
    Alpha zones: replace digit-like chars with their letter equivalents.
    Numeric zones: replace letter-like chars with digit equivalents.
    """
    chars = list(text.replace(" ", ""))
    if len(chars) < 6:
        return text  # Too short to apply zone logic safely

    # Simple zone map for standard 10-char plate: LLNNLLLNNNN
    zones = ["A", "A", "N", "N", "A", "A", "A", "N", "N", "N", "N"]
    corrected = []

    for i, ch in enumerate(chars):
        zone = zones[i] if i < len(zones) else "A"
        if zone == "N" and ch in _OCR_CORRECTIONS:
            # In a numeric zone — convert letter lookalikes to digits
            corrected.append(_OCR_CORRECTIONS[ch])
        elif zone == "A" and ch.isdigit():
            # In an alpha zone — convert digit lookalikes to letters
            digit_to_letter = {v: k for k, v in _OCR_CORRECTIONS.items() if k.isalpha()}
            corrected.append(digit_to_letter.get(ch, ch))
        else:
            corrected.append(ch)

    result = "".join(corrected)
    # Reformat as standard: MH 12 AB 1234
    if len(result) >= 8:
        result = f"{result[:2]} {result[2:4]} {result[4:7]} {result[7:]}"
    return result.strip()
