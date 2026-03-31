"""
AgriFarm — Database Layer
SQLAlchemy models + session management.
Uses SQLite by default (zero config); swap DATABASE_URL for PostgreSQL/MySQL.
"""

from __future__ import annotations
import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    DateTime, Text, Boolean, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./agrifarm.db")

# connect_args only needed for SQLite (thread safety)
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─────────────────────────────────────────────
# TABLE: sensor_data  (from your project, kept intact)
# ─────────────────────────────────────────────
class SensorData(Base):
    __tablename__ = "sensor_data"

    id            = Column(Integer, primary_key=True, index=True)
    field_id      = Column(String(50), index=True, nullable=False)
    timestamp     = Column(DateTime, default=datetime.utcnow, index=True)
    temperature   = Column(Float)   # °C
    soil_moisture = Column(Float)   # %
    ph_level      = Column(Float)
    nitrogen      = Column(Float)   # ppm
    phosphorus    = Column(Float)   # ppm
    potassium     = Column(Float)   # ppm
    humidity      = Column(Float)   # %
    rainfall_mm   = Column(Float, default=0.0)

    __table_args__ = (
        Index("ix_sensor_field_ts", "field_id", "timestamp"),
    )


# ─────────────────────────────────────────────
# TABLE: alerts  (from your project, extended)
# ─────────────────────────────────────────────
class Alert(Base):
    __tablename__ = "alerts"

    id                         = Column(Integer, primary_key=True, index=True)
    field_id                   = Column(String(50), index=True)
    timestamp                  = Column(DateTime, default=datetime.utcnow)
    severity                   = Column(String(20))   # low / medium / high / critical
    alert_type                 = Column(String(50))
    message                    = Column(Text)
    is_resolved                = Column(Boolean, default=False)
    auto_remediation_applied   = Column(Boolean, default=False)
    remediation_action         = Column(Text, nullable=True)


# ─────────────────────────────────────────────
# TABLE: remediation_logs  (from your project)
# ─────────────────────────────────────────────
class RemediationLog(Base):
    __tablename__ = "remediation_logs"

    id             = Column(Integer, primary_key=True, index=True)
    alert_id       = Column(Integer, index=True)
    field_id       = Column(String(50))
    timestamp      = Column(DateTime, default=datetime.utcnow)
    action_type    = Column(String(50))
    action_details = Column(Text)
    success        = Column(Boolean, default=True)
    cost_estimate  = Column(Float, nullable=True)   # INR


# ─────────────────────────────────────────────
# TABLE: agent_logs  (from your project)
# ─────────────────────────────────────────────
class AgentLog(Base):
    __tablename__ = "agent_logs"

    id             = Column(Integer, primary_key=True, index=True)
    timestamp      = Column(DateTime, default=datetime.utcnow)
    agent_type     = Column(String(50))   # diagnostic / action / disease / yield / market
    query          = Column(Text)
    response       = Column(Text)
    tools_used     = Column(Text)         # JSON list
    execution_time = Column(Float)


# ─────────────────────────────────────────────
# TABLE: disease_scans  (new — from AgriGenie)
# ─────────────────────────────────────────────
class DiseaseScan(Base):
    __tablename__ = "disease_scans"

    id           = Column(Integer, primary_key=True, index=True)
    timestamp    = Column(DateTime, default=datetime.utcnow)
    field_id     = Column(String(50), nullable=True)
    crop         = Column(String(50))
    disease      = Column(String(100))
    confidence   = Column(Float)
    is_healthy   = Column(Boolean)
    llm_advice   = Column(Text)


# ─────────────────────────────────────────────
# TABLE: market_snapshots  (new — from AgriGenie)
# ─────────────────────────────────────────────
class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"

    id           = Column(Integer, primary_key=True, index=True)
    timestamp    = Column(DateTime, default=datetime.utcnow)
    crop         = Column(String(50), index=True)
    date         = Column(String(10))
    price_inr    = Column(Float)
    market       = Column(String(100))
    state        = Column(String(50))


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def get_db():
    """FastAPI dependency — yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialised")
