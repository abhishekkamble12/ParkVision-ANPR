"""
=============================================================
  ParkVision Core — Frame Deduplication Tracker
=============================================================
Prevents the same vehicle from being logged multiple times
across consecutive video frames using centroid tracking.

Key concepts:
  - Each detected plate gets a persistent track ID
  - A plate is only passed to OCR once per track (when confidence peaks)
  - Tracks expire after max_missing_frames (vehicle left camera view)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Track:
    """Represents one active vehicle track across frames."""
    track_id: int
    centroid: tuple[int, int]           # (cx, cy) current center
    best_confidence: float              # Highest detection conf seen
    best_crop: Optional[np.ndarray]     # Crop at peak confidence
    ocr_text: str = ""                  # OCR result (set once)
    ocr_done: bool = False              # Whether OCR has run
    frames_missing: int = 0            # Consecutive frames without detection
    frame_first_seen: int = 0
    frame_last_seen: int = 0


class CentroidTracker:
    """
    Lightweight centroid-based tracker for number plate deduplication.

    Algorithm:
      1. For each new frame's detections, compute centroids
      2. Match them to existing tracks by minimum Euclidean distance
      3. Unmatched detections → new tracks
      4. Tracks not seen for max_missing_frames → deregistered

    This is ByteTrack-lite — good enough for fixed-camera parking
    without needing heavy dependencies.
    """

    def __init__(self, max_distance: int = 80, max_missing_frames: int = 30):
        """
        Args:
            max_distance:      Max pixel distance to match a detection to a track.
            max_missing_frames: Frames before a track is considered gone.
        """
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self._next_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, detections: list, frame_idx: int = 0) -> list[Track]:
        """
        Updates tracker with new frame detections.

        Args:
            detections: List of PlateDetection objects from detector.
            frame_idx:  Current frame number.

        Returns:
            List of all active Track objects (including unmatched existing ones).
        """
        if not detections:
            self._age_tracks()
            return list(self.tracks.values())

        # Compute centroids for new detections
        new_centroids = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            new_centroids.append(((x1 + x2) // 2, (y1 + y2) // 2))

        if not self.tracks:
            # No existing tracks — register all new detections
            for i, det in enumerate(detections):
                self._register(det, new_centroids[i], frame_idx)
            return list(self.tracks.values())

        # ── Hungarian-lite: greedy nearest-neighbour matching ────────────────
        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid].centroid for tid in track_ids]
        matched_det_indices = set()
        matched_track_ids = set()

        # Build distance matrix
        dist_matrix = np.zeros((len(track_ids), len(new_centroids)))
        for ti, tc in enumerate(track_centroids):
            for di, dc in enumerate(new_centroids):
                dist_matrix[ti, di] = np.hypot(tc[0] - dc[0], tc[1] - dc[1])

        # Greedily match closest pairs
        while True:
            if dist_matrix.size == 0:
                break
            min_val = dist_matrix.min()
            if min_val > self.max_distance:
                break
            ti, di = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)
            tid = track_ids[ti]

            if tid not in matched_track_ids and di not in matched_det_indices:
                det = detections[di]
                track = self.tracks[tid]

                # Update track centroid and best crop
                track.centroid = new_centroids[di]
                track.frame_last_seen = frame_idx
                track.frames_missing = 0

                # Keep the crop with highest confidence for OCR
                if det.confidence > track.best_confidence:
                    track.best_confidence = det.confidence
                    track.best_crop = det.plate_crop.copy()

                matched_track_ids.add(tid)
                matched_det_indices.add(di)

            # Invalidate this row and column
            dist_matrix[ti, :] = np.inf
            dist_matrix[:, di] = np.inf

        # Register unmatched detections as new tracks
        for di, det in enumerate(detections):
            if di not in matched_det_indices:
                self._register(det, new_centroids[di], frame_idx)

        # Age unmatched existing tracks
        for tid in track_ids:
            if tid not in matched_track_ids:
                self.tracks[tid].frames_missing += 1

        # Remove expired tracks
        expired = [tid for tid, t in self.tracks.items()
                   if t.frames_missing > self.max_missing_frames]
        for tid in expired:
            del self.tracks[tid]

        return list(self.tracks.values())

    def get_tracks_needing_ocr(self) -> list[Track]:
        """Returns tracks that have a good crop but haven't been OCR'd yet."""
        return [t for t in self.tracks.values()
                if not t.ocr_done and t.best_crop is not None
                and t.best_confidence >= 0.45]

    def _register(self, det, centroid, frame_idx):
        """Creates a new Track from a detection."""
        self.tracks[self._next_id] = Track(
            track_id=self._next_id,
            centroid=centroid,
            best_confidence=det.confidence,
            best_crop=det.plate_crop.copy(),
            frame_first_seen=frame_idx,
            frame_last_seen=frame_idx,
        )
        self._next_id += 1

    def _age_tracks(self):
        """Increments missing counter for all tracks (called on empty frames)."""
        expired = []
        for tid, track in self.tracks.items():
            track.frames_missing += 1
            if track.frames_missing > self.max_missing_frames:
                expired.append(tid)
        for tid in expired:
            del self.tracks[tid]
