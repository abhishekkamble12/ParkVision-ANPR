"""
=============================================================
  ParkVision — Parking Session Data Models
=============================================================
Defines the SQLAlchemy ORM models for SQLite storage.
"""

import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ParkingSession(Base):
    """
    Represents one complete parking session (entry → exit).

    A session is created on vehicle ENTRY and closed on EXIT.
    plate_number is stored as-is for demo; in production, hash it.
    """
    __tablename__ = "parking_sessions"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    plate_number   = Column(String(20), nullable=False, index=True)
    entry_time     = Column(DateTime, default=datetime.datetime.utcnow)
    exit_time      = Column(DateTime, nullable=True)
    duration_mins  = Column(Float, nullable=True)      # Filled on exit
    entry_image    = Column(String(255), nullable=True) # Path to saved crop
    exit_image     = Column(String(255), nullable=True)
    ocr_confidence = Column(Float, default=0.0)
    is_active      = Column(Boolean, default=True)     # True = vehicle still parked
    slot_id        = Column(String(10), nullable=True)  # e.g. "A3"


class DetectionLog(Base):
    """
    Raw log of every plate detection event (for metrics / debugging).
    Separate from ParkingSession so we don't clutter the session table.
    """
    __tablename__ = "detection_logs"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    timestamp    = Column(DateTime, default=datetime.datetime.utcnow)
    plate_number = Column(String(20), nullable=False)
    raw_ocr_text = Column(String(50), nullable=True)
    confidence   = Column(Float, default=0.0)
    source       = Column(String(20), default="image")  # "image" | "video"
    is_valid_ind = Column(Boolean, default=False)
