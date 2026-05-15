"""
database/models.py
Real SQLite database with SQLAlchemy ORM
Schema covers: plant history, scan logs, outbreak reports, user sessions
"""

from sqlalchemy import (
    create_engine, Column, String, Float, Integer,
    DateTime, Boolean, Text, JSON, ForeignKey
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

DATABASE_URL = "sqlite:///./agromind.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


class Base(DeclarativeBase):
    pass


class PlantHistory(Base):
    """
    Core schema: longitudinal health records per plant per user
    Supports trend detection across multiple scans
    """
    __tablename__ = "plant_history"

    id           = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id      = Column(String, nullable=False, index=True)
    plant_id     = Column(String, nullable=False, index=True)   # user-named plant
    timestamp    = Column(DateTime, default=datetime.utcnow, index=True)
    species      = Column(String)
    common_name  = Column(String)
    disease      = Column(String)
    disease_prob = Column(Float, default=0.0)
    confidence   = Column(Float, nullable=False)
    is_healthy   = Column(Boolean, default=True)
    severity     = Column(String, default="none")   # none | mild | moderate | severe
    image_url    = Column(String)
    image_b64    = Column(Text)                     # stored as base64 for demo
    treatment    = Column(JSON)
    risk_score   = Column(Float, default=0.0)
    latitude     = Column(Float)
    longitude    = Column(Float)
    universe     = Column(String, default="farmer")
    language     = Column(String, default="en")
    raw_api_resp = Column(JSON)

    def to_dict(self):
        return {
            "id": self.id,
            "plant_id": self.plant_id,
            "timestamp": self.timestamp.isoformat(),
            "species": self.species,
            "common_name": self.common_name,
            "disease": self.disease,
            "disease_prob": self.disease_prob,
            "confidence": self.confidence,
            "is_healthy": self.is_healthy,
            "severity": self.severity,
            "risk_score": self.risk_score,
            "universe": self.universe,
            "language": self.language,
        }


class OutbreakReport(Base):
    """
    Anonymised geo-tagged disease reports for DBSCAN clustering
    No PII stored — only lat/lon + disease + timestamp
    """
    __tablename__ = "outbreak_reports"

    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp   = Column(DateTime, default=datetime.utcnow, index=True)
    disease     = Column(String, nullable=False, index=True)
    species     = Column(String)
    latitude    = Column(Float, nullable=False)
    longitude   = Column(Float, nullable=False)
    severity    = Column(String)
    confidence  = Column(Float)
    cluster_id  = Column(Integer)                   # assigned by DBSCAN


class ScanLog(Base):
    """
    Full audit log of every API call for monitoring + debugging
    """
    __tablename__ = "scan_logs"

    id            = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp     = Column(DateTime, default=datetime.utcnow)
    endpoint      = Column(String)
    user_id       = Column(String)
    ip_address    = Column(String)
    input_type    = Column(String)                  # image | text | voice
    validation_ok = Column(Boolean)
    rejection_reason = Column(String)
    prediction    = Column(String)
    confidence    = Column(Float)
    latency_ms    = Column(Integer)
    error         = Column(String)
    universe      = Column(String)
    language      = Column(String)


class User(Base):
    __tablename__ = "users"

    id         = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username   = Column(String, unique=True, nullable=False)
    email      = Column(String, unique=True)
    password   = Column(String, nullable=False)     # hashed in production
    name       = Column(String)
    universe   = Column(String, default="farmer")
    language   = Column(String, default="en")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    latitude   = Column(Float)
    longitude  = Column(Float)


def init_db():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)
    print("[DB] Tables created: plant_history, outbreak_reports, scan_logs, users")


def get_db():
    """FastAPI dependency injection"""
    db = Session(engine)
    try:
        yield db
    finally:
        db.close()


# ─── TREND DETECTION LOGIC ─────────────────────────────────────────
def compute_plant_trend(records: list) -> dict:
    """
    Analyse last N scans to detect health trend.
    Returns: declining | improving | stable | insufficient_data
    """
    if len(records) < 2:
        return {"trend": "insufficient_data", "message": "Need at least 2 scans to detect trend"}

    # Score each record: healthy=1.0, mild=0.7, moderate=0.4, severe=0.1
    severity_map = {"none": 1.0, "mild": 0.7, "moderate": 0.4, "severe": 0.1}
    scores = [severity_map.get(r.severity, 0.5) for r in records[-5:]]  # last 5

    if len(scores) < 2:
        return {"trend": "insufficient_data", "message": "Insufficient records"}

    # Linear trend: compare first half vs second half
    mid = len(scores) // 2
    first_avg = sum(scores[:mid]) / max(len(scores[:mid]), 1)
    second_avg = sum(scores[mid:]) / max(len(scores[mid:]), 1)
    delta = second_avg - first_avg

    if delta < -0.2:
        trend = "declining"
        msg = "⚠️ Plant health is worsening across recent scans. Take action immediately."
    elif delta > 0.2:
        trend = "improving"
        msg = "✅ Plant health is improving. Continue current treatment."
    else:
        trend = "stable"
        msg = "📊 Plant health is stable. Monitor regularly."

    return {
        "trend": trend,
        "message": msg,
        "score_history": scores,
        "delta": round(delta, 3),
        "scan_count": len(records)
    }
