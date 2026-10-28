"""
=============================================================
  ParkVision — Video Stream Pipeline Demo
=============================================================
Processes a video file (or webcam) frame by frame:
  1. Detects plates with YOLOv8 every N frames
  2. Tracks plates across frames (deduplication)
  3. Runs OCR on the best crop per track
  4. Logs entry events to SQLite
  5. Saves annotated output video

Usage:
    python scripts/run_video.py                      # Webcam (cam 0)
    python scripts/run_video.py --video car.mp4      # Video file
    python scripts/run_video.py --detect-every 5     # Run YOLO every 5 frames
"""

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.detector import PlateDetector
from core.preprocessor import preprocess_plate
from core.ocr_engine import PlateOCREngine
from core.tracker import CentroidTracker
from parking.session_manager import ParkingSessionManager

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "output_videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_video_pipeline(
    video_source,           # Path str or int (webcam index)
    detect_every: int = 3,  # Run YOLO every N frames (performance trade-off)
    output_path: str = None,
    max_frames: int = None,
    headless: bool = False
) -> None:
    """
    Full video ANPR pipeline with tracking and parking management.

    Args:
        video_source:  Video file path or webcam index (0).
        detect_every:  Run YOLO detection every N frames. Tracker fills the gaps.
        output_path:   Where to save annotated video. None = auto.
        max_frames:    Stop after N frames (useful for testing).
        headless:      Skip cv2.imshow (for servers without display).
    """
    print(f"\n{'='*60}")
    print("  ParkVision — Video ANPR Pipeline")
    print(f"  Source: {video_source}  |  Detect every: {detect_every} frames")
    print(f"{'='*60}\n")

    # ── Initialize components ─────────────────────────────────────────────────
    detector  = PlateDetector(confidence_threshold=0.40)
    ocr       = PlateOCREngine(use_gpu=False)
    tracker   = CentroidTracker(max_distance=80, max_missing_frames=25)
    parking   = ParkingSessionManager()

    # ── Open video source ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video source: {video_source}")
        return

    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {w}×{h} @ {fps_src:.1f} FPS, {total_frames} frames")

    # ── Output video writer ────────────────────────────────────────────────────
    if output_path is None:
        src_name = os.path.splitext(os.path.basename(str(video_source)))[0]
        output_path = os.path.join(OUTPUT_DIR, f"anpr_{src_name}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps_src, (w, h))

    # ── Main loop ──────────────────────────────────────────────────────────────
    frame_idx = 0
    last_detections = []
    fps_history = []
    ocr_results_cache: dict[int, str] = {}   # {track_id: plate_text}

    print("\n  Processing... Press Q to quit (if window visible)\n")

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        # ── A: Detect plates every N frames ───────────────────────────────────
        if frame_idx % detect_every == 0:
            last_detections = detector.detect(frame)
            for d in last_detections:
                d.frame_idx = frame_idx

        # ── B: Update tracker ─────────────────────────────────────────────────
        active_tracks = tracker.update(last_detections, frame_idx)

        # ── C: Run OCR on tracks that need it ─────────────────────────────────
        for track in tracker.get_tracks_needing_ocr():
            if track.best_crop is None:
                continue
            processed, _ = preprocess_plate(track.best_crop)
            result = ocr.read_plate(track.best_crop, preprocessed=processed)
            track.ocr_text = result.cleaned_text
            track.ocr_done = True

            # Cache and log
            ocr_results_cache[track.track_id] = result.cleaned_text
            parking.log_detection(
                plate_number=result.cleaned_text or "UNKNOWN",
                raw_text=result.raw_text,
                confidence=result.confidence,
                source="video",
                is_valid_ind=result.is_valid_ind
            )

            # Auto-log entry for valid plates
            if result.is_valid_ind and result.confidence > 0.45:
                parking.log_entry(
                    plate_number=result.cleaned_text,
                    confidence=result.confidence
                )
                print(f"  Frame {frame_idx:04d} | Track #{track.track_id} | "
                      f"PLATE: {result.cleaned_text} ({result.confidence*100:.0f}%)")

        # ── D: Annotate frame ─────────────────────────────────────────────────
        # Draw detection boxes from last_detections
        plate_texts = [ocr_results_cache.get(t.track_id, "") for t in active_tracks]
        annotated = detector.annotate_frame(frame, last_detections)

        # Draw track IDs and OCR text above boxes
        for track in active_tracks:
            cx, cy = track.centroid
            text = ocr_results_cache.get(track.track_id, f"Track #{track.track_id}")
            cv2.putText(annotated, text, (cx - 40, cy - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

        # ── E: HUD overlay ────────────────────────────────────────────────────
        t1 = time.time()
        fps_live = 1.0 / max(t1 - t0, 1e-5)
        fps_history.append(fps_live)

        stats = parking.get_stats()
        hud_lines = [
            f"Frame: {frame_idx}/{total_frames or '?'}",
            f"FPS:   {fps_live:.1f}",
            f"Tracks: {len(active_tracks)}",
            f"Logged: {stats['total_sessions']} sessions",
        ]
        _draw_hud(annotated, hud_lines)

        # ── F: Write frame + show ─────────────────────────────────────────────
        writer.write(annotated)

        if not headless:
            cv2.imshow("ParkVision — ANPR", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("  User pressed Q — stopping.")
                break

        # Progress log every 50 frames
        if frame_idx % 50 == 0:
            print(f"  [{frame_idx:04d}] FPS={fps_live:.1f}  Tracks={len(active_tracks)}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    if not headless:
        cv2.destroyAllWindows()

    avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
    final_stats = parking.get_stats()

    print(f"\n{'='*60}")
    print("  VIDEO PIPELINE COMPLETE")
    print(f"  Frames processed:  {frame_idx}")
    print(f"  Avg FPS:           {avg_fps:.1f}")
    print(f"  Sessions logged:   {final_stats['total_sessions']}")
    print(f"  Total detections:  {final_stats['total_detections']}")
    print(f"  Output saved to:   {output_path}")
    print(f"{'='*60}\n")


def _draw_hud(frame, lines: list[str], x: int = 15, y_start: int = 30, line_h: int = 28):
    """Draws a semi-transparent HUD panel on the top-left of the frame."""
    import numpy as np
    overlay = frame.copy()
    panel_h = len(lines) * line_h + 20
    panel_w = 260
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y_start + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 229, 170), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="ParkVision Video ANPR Demo")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file. Defaults to webcam (0).")
    parser.add_argument("--detect-every", type=int, default=3,
                        help="Run YOLO every N frames (default: 3).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Process only first N frames (for quick testing).")
    parser.add_argument("--headless", action="store_true",
                        help="Run without cv2.imshow (useful on servers).")
    args = parser.parse_args()

    source = args.video if args.video else 0
    run_video_pipeline(
        video_source=source,
        detect_every=args.detect_every,
        max_frames=args.max_frames,
        headless=args.headless
    )


if __name__ == "__main__":
    main()
