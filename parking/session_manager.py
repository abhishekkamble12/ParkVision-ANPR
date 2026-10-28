"""
=============================================================
  ParkVision — Parking Session Manager
=============================================================
Core business logic:
  - Log vehicle ENTRY (create session)
  - Log vehicle EXIT (close session, compute duration)
  - Suppress duplicate detections within cooldown window
  - Query active sessions and history
"""

import datetime
import os
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from parking.models import Base, ParkingSession, DetectionLog


# Default SQLite database path
_DEFAULT_DB = os.path.join(os.path.dirname(__file__), "..", "data", "parkvision.db")


class ParkingSessionManager:
    """
    Manages parking entry/exit sessions backed by SQLite.

    Usage:
        mgr = ParkingSessionManager()
        mgr.log_entry("MH 12 AB 1234", confidence=0.92)
        mgr.log_exit("MH 12 AB 1234")
        sessions = mgr.get_active_sessions()
    """

    # Minimum seconds between two detections of the same plate
    # Prevents the same vehicle being logged twice in 2 consecutive frames
    DEDUP_COOLDOWN_SECONDS = 30

    def __init__(self, db_path: str = None):
        db_path = db_path or _DEFAULT_DB
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        self._engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self._engine)
        self._SessionLocal = sessionmaker(bind=self._engine)

        # In-memory cooldown tracker: {plate_number: last_seen_datetime}
        self._last_seen: dict[str, datetime.datetime] = {}
        print(f"  [ParkingMgr] ✅ Database ready at: {db_path}")

    # ─── Entry / Exit ──────────────────────────────────────────────────────────

    def log_entry(self, plate_number: str, confidence: float = 0.0,
                  entry_image_path: str = None, slot_id: str = None) -> Optional[ParkingSession]:
        """
        Logs a vehicle entry. Returns None if within dedup cooldown window.

        Args:
            plate_number:     Cleaned OCR plate text (e.g. "MH 12 AB 1234").
            confidence:       OCR confidence score.
            entry_image_path: Path to saved plate crop image.
            slot_id:          Parking slot (optional).

        Returns:
            New ParkingSession if logged, None if suppressed as duplicate.
        """
        plate_number = plate_number.strip().upper()

        # ── Deduplication check ──────────────────────────────────────────────
        if self._is_duplicate(plate_number):
            return None

        with self._SessionLocal() as db:
            # Check if there's already an active session for this plate
            existing = db.query(ParkingSession).filter_by(
                plate_number=plate_number, is_active=True
            ).first()

            if existing:
                print(f"  [ParkingMgr] ⚠️  {plate_number} already has an active session (ID {existing.id})")
                return existing

            # Create new session
            session = ParkingSession(
                plate_number=plate_number,
                entry_time=datetime.datetime.utcnow(),
                ocr_confidence=confidence,
                entry_image=entry_image_path,
                slot_id=slot_id,
                is_active=True
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            session_id = session.id

        self._last_seen[plate_number] = datetime.datetime.utcnow()
        print(f"  [ParkingMgr] 🟢 ENTRY logged → {plate_number} (Session #{session_id})")
        return session

    def log_exit(self, plate_number: str,
                 exit_image_path: str = None) -> Optional[ParkingSession]:
        """
        Closes an active parking session and computes duration.

        Returns:
            Closed ParkingSession, or None if no active session found.
        """
        plate_number = plate_number.strip().upper()

        with self._SessionLocal() as db:
            session = db.query(ParkingSession).filter_by(
                plate_number=plate_number, is_active=True
            ).order_by(ParkingSession.entry_time.desc()).first()

            if not session:
                print(f"  [ParkingMgr] ⚠️  No active session for {plate_number}")
                return None

            exit_time = datetime.datetime.utcnow()
            duration = (exit_time - session.entry_time).total_seconds() / 60.0

            session.exit_time = exit_time
            session.duration_mins = round(duration, 2)
            session.exit_image = exit_image_path
            session.is_active = False
            db.commit()
            db.refresh(session)

            print(f"  [ParkingMgr] 🔴 EXIT logged  → {plate_number} "
                  f"(Duration: {duration:.1f} min)")
            return session

    def log_detection(self, plate_number: str, raw_text: str,
                      confidence: float, source: str = "image",
                      is_valid_ind: bool = False) -> None:
        """Appends a raw detection event to the detection log table."""
        with self._SessionLocal() as db:
            log = DetectionLog(
                plate_number=plate_number,
                raw_ocr_text=raw_text,
                confidence=confidence,
                source=source,
                is_valid_ind=is_valid_ind
            )
            db.add(log)
            db.commit()

    # ─── Queries ───────────────────────────────────────────────────────────────

    def get_active_sessions(self) -> list[dict]:
        """Returns all currently active (parked) sessions as dicts."""
        with self._SessionLocal() as db:
            sessions = db.query(ParkingSession).filter_by(is_active=True).all()
            return [_session_to_dict(s) for s in sessions]

    def get_all_sessions(self, limit: int = 100) -> list[dict]:
        """Returns recent sessions (newest first)."""
        with self._SessionLocal() as db:
            sessions = (db.query(ParkingSession)
                        .order_by(ParkingSession.entry_time.desc())
                        .limit(limit).all())
            return [_session_to_dict(s) for s in sessions]

    def get_detection_logs(self, limit: int = 200) -> list[dict]:
        """Returns recent raw detection events."""
        with self._SessionLocal() as db:
            logs = (db.query(DetectionLog)
                    .order_by(DetectionLog.timestamp.desc())
                    .limit(limit).all())
            return [_log_to_dict(l) for l in logs]

    def get_stats(self) -> dict:
        """Returns summary statistics for the dashboard."""
        with self._SessionLocal() as db:
            total = db.query(ParkingSession).count()
            active = db.query(ParkingSession).filter_by(is_active=True).count()
            completed = db.query(ParkingSession).filter_by(is_active=False).count()
            logs_total = db.query(DetectionLog).count()

            avg_dur = 0.0
            if completed > 0:
                from sqlalchemy import func
                result = db.query(func.avg(ParkingSession.duration_mins)).filter_by(
                    is_active=False
                ).scalar()
                avg_dur = round(result or 0.0, 2)

        return {
            "total_sessions": total,
            "active_sessions": active,
            "completed_sessions": completed,
            "total_detections": logs_total,
            "avg_duration_mins": avg_dur,
        }

    # ─── Helpers ───────────────────────────────────────────────────────────────

    def _is_duplicate(self, plate_number: str) -> bool:
        """Returns True if this plate was seen within the cooldown window."""
        last = self._last_seen.get(plate_number)
        if last is None:
            return False
        elapsed = (datetime.datetime.utcnow() - last).total_seconds()
        return elapsed < self.DEDUP_COOLDOWN_SECONDS


# ─── Serializers ──────────────────────────────────────────────────────────────

def _session_to_dict(s: ParkingSession) -> dict:
    return {
        "id": s.id,
        "plate_number": s.plate_number,
        "entry_time": s.entry_time.isoformat() if s.entry_time else None,
        "exit_time": s.exit_time.isoformat() if s.exit_time else None,
        "duration_mins": s.duration_mins,
        "is_active": s.is_active,
        "slot_id": s.slot_id,
        "ocr_confidence": s.ocr_confidence,
    }


def _log_to_dict(l: DetectionLog) -> dict:
    return {
        "id": l.id,
        "timestamp": l.timestamp.isoformat() if l.timestamp else None,
        "plate_number": l.plate_number,
        "raw_ocr_text": l.raw_ocr_text,
        "confidence": l.confidence,
        "source": l.source,
        "is_valid_ind": l.is_valid_ind,
    }
