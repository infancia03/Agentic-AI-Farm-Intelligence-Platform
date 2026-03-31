"""
AgriFarm Intelligence Platform — FastAPI Backend
All endpoints from your original project PLUS:
  /api/disease/*    — ResNet18 leaf disease detection
  /api/yield/*      — Prophet crop yield forecasting
  /api/market/*     — RAG + LLM market price advisory
  /api/aiops/llm-analyse — LLM-powered anomaly root-cause analysis
"""

from __future__ import annotations
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.database import get_db, init_db, SensorData, Alert
from app.models import (
    SensorDataCreate, SensorDataResponse, AlertResponse,
    AgentQuery, RemediationRequest,
)
from app.agents.orchestrator import OrchestratorAgent
from app.aiops.anomaly_detector import AnomalyDetector
from app.aiops.auto_remediation import AutoRemediationEngine
from datetime import datetime, timedelta
from loguru import logger
import psutil


# ── Startup / Shutdown ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🌾 AgriFarm starting …")
    init_db()
    # Bootstrap RAG index
    try:
        from rag.retriever import ingest_knowledge_base
        ingest_knowledge_base()
    except Exception as e:
        logger.warning(f"RAG ingest skipped: {e}")
    yield
    logger.info("AgriFarm shutting down")


app = FastAPI(
    title="AgriFarm Intelligence Platform",
    description=(
        "Merged Agentic AI + AIOps platform for precision agriculture.\n\n"
        "**Features:** Multi-agent tool-calling · ResNet18 disease detection · "
        "Prophet yield forecast · ChromaDB RAG market advisory · "
        "LLM-powered AIOps anomaly analysis · Auto-remediation engine"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

orchestrator    = OrchestratorAgent()
anomaly_detector= AnomalyDetector()
auto_remediation= AutoRemediationEngine()


# ════════════════════════════════════════════════════════════
# SENSOR ENDPOINTS  (your original — unchanged)
# ════════════════════════════════════════════════════════════
@app.post("/api/sensors/data", response_model=SensorDataResponse, tags=["Sensors"])
def create_sensor_data(
    data: SensorDataCreate,
    bg: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Submit a new IoT sensor reading and trigger background AIOps monitoring."""
    row = SensorData(**data.model_dump())
    db.add(row); db.commit(); db.refresh(row)
    bg.add_task(_bg_monitor, data.field_id)
    return row


@app.get("/api/sensors/data/{field_id}", tags=["Sensors"])
def get_sensor_history(field_id: str, hours: int = 24, db: Session = Depends(get_db)):
    cut = datetime.utcnow() - timedelta(hours=hours)
    rows = (
        db.query(SensorData)
        .filter(SensorData.field_id == field_id, SensorData.timestamp >= cut)
        .order_by(SensorData.timestamp.desc())
        .all()
    )
    return rows


@app.get("/api/sensors/latest/{field_id}", tags=["Sensors"])
def get_latest(field_id: str, db: Session = Depends(get_db)):
    row = (
        db.query(SensorData)
        .filter(SensorData.field_id == field_id)
        .order_by(SensorData.timestamp.desc())
        .first()
    )
    if not row:
        raise HTTPException(404, f"No data for field {field_id}")
    return row


@app.get("/api/sensors/fields", tags=["Sensors"])
def list_fields(db: Session = Depends(get_db)):
    fields = [r[0] for r in db.query(SensorData.field_id).distinct().all()]
    return {"fields": fields, "total": len(fields)}


# ════════════════════════════════════════════════════════════
# AGENTIC AI  (your original + extended with disease/yield/market)
# ════════════════════════════════════════════════════════════
@app.post("/api/agent/query", tags=["Agentic AI"])
def agent_query(req: AgentQuery, db: Session = Depends(get_db)):
    """
    Main agentic endpoint. Automatically routes to:
    - Diagnostic + Action agents (sensor issues)
    - Disease agent (if query mentions leaf/disease/photo)
    - Yield agent (harvest/forecast queries)
    - Market agent (price/sell queries)
    """
    result = orchestrator.process_query(
        query=req.query, db=db,
        field_id=req.field_id, auto_remediate=True,
        farm_context=req.farm_context,
    )
    return result


@app.get("/api/agent/recommendations/{field_id}", tags=["Agentic AI"])
def field_recommendations(field_id: str, db: Session = Depends(get_db)):
    return orchestrator.get_field_recommendations(field_id, db)


# ════════════════════════════════════════════════════════════
# DISEASE DETECTION  (new — ResNet18 + LLM)
# ════════════════════════════════════════════════════════════
@app.post("/api/disease/detect", tags=["Disease Detection"])
async def detect_disease(
    file:     UploadFile = File(..., description="Crop leaf image (jpg/png)"),
    crop:     str        = Form("Unknown"),
    field_id: str        = Form(""),
    location: str        = Form("India"),
):
    """
    Upload a leaf photo → ResNet18 classifies 38 disease types →
    OpenRouter LLM generates treatment plan.
    """
    from app.agents.disease_agent import run as disease_run
    try:
        img_bytes   = await file.read()
        result      = disease_run(img_bytes, farm_context={"crop": crop, "field_id": field_id, "location": location})
        return result
    except Exception as e:
        logger.exception(e)
        raise HTTPException(500, str(e))


# ════════════════════════════════════════════════════════════
# YIELD FORECASTING  (new — Prophet + LLM)
# ════════════════════════════════════════════════════════════
@app.get("/api/yield/forecast", tags=["Yield & Market"])
def yield_forecast(
    crop:  str = "Tomato",
    state: str = "Tamil Nadu",
    years: int = 3,
):
    """Facebook Prophet time-series forecast + LLM narrative."""
    from app.agents.yield_market_agent import run_yield
    try:
        return run_yield(crop, state)
    except Exception as e:
        raise HTTPException(500, str(e))


# ════════════════════════════════════════════════════════════
# MARKET ADVISORY  (new — ChromaDB RAG + LLM)
# ════════════════════════════════════════════════════════════
@app.get("/api/market/advisory", tags=["Yield & Market"])
def market_advisory(
    crop:              str   = "Tomato",
    quantity_quintals: float = 10.0,
):
    """Price trend analysis + RAG knowledge retrieval + LLM sell/hold recommendation."""
    from app.agents.yield_market_agent import run_market
    try:
        return run_market(crop, quantity_quintals)
    except Exception as e:
        raise HTTPException(500, str(e))


# ════════════════════════════════════════════════════════════
# AIOPS  (your original + LLM analysis layer)
# ════════════════════════════════════════════════════════════
@app.post("/api/aiops/monitor", tags=["AIOps"])
def aiops_monitor(field_id: Optional[str] = None, db: Session = Depends(get_db)):
    """Trigger rule-based anomaly detection and auto-remediation."""
    result = anomaly_detector.monitor_and_alert(db, field_id)
    if result["alerts_created"] > 0:
        result["auto_remediation"] = auto_remediation.process_alerts(db, field_id)
    return result


@app.post("/api/aiops/llm-analyse", tags=["AIOps"])
def aiops_llm_analyse(field_id: Optional[str] = None, db: Session = Depends(get_db)):
    """LLM-powered root-cause analysis of current anomalies."""
    return anomaly_detector.llm_analyse(db, field_id)


@app.get("/api/aiops/trends/{field_id}", tags=["AIOps"])
def aiops_trends(field_id: str, hours: int = 24, db: Session = Depends(get_db)):
    return anomaly_detector.get_trend_analysis(db, field_id, hours)


@app.get("/api/aiops/status", tags=["AIOps"])
def aiops_status():
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    return {
        "status":     "warning" if cpu > 80 or mem.percent > 85 else "healthy",
        "cpu_pct":    round(cpu, 1),
        "memory_pct": round(mem.percent, 1),
        "memory_mb":  round(mem.used / 1024 / 1024, 0),
        "timestamp":  datetime.utcnow().isoformat(),
    }


# ════════════════════════════════════════════════════════════
# ALERTS  (your original — unchanged)
# ════════════════════════════════════════════════════════════
@app.get("/api/alerts", tags=["Alerts"])
def list_alerts(
    field_id: Optional[str]  = None,
    resolved: Optional[bool] = None,
    severity: Optional[str]  = None,
    db: Session = Depends(get_db),
):
    q = db.query(Alert)
    if field_id: q = q.filter(Alert.field_id == field_id)
    if resolved is not None: q = q.filter(Alert.is_resolved == resolved)
    if severity: q = q.filter(Alert.severity == severity)
    return q.order_by(Alert.timestamp.desc()).limit(100).all()


@app.patch("/api/alerts/{alert_id}/resolve", tags=["Alerts"])
def resolve_alert(alert_id: int, db: Session = Depends(get_db)):
    a = db.query(Alert).filter(Alert.id == alert_id).first()
    if not a: raise HTTPException(404, "Alert not found")
    a.is_resolved = True; db.commit()
    return {"message": "Alert resolved", "alert_id": alert_id}


# ════════════════════════════════════════════════════════════
# REMEDIATION  (your original — unchanged)
# ════════════════════════════════════════════════════════════
@app.post("/api/remediation/execute", tags=["Remediation"])
def execute_remediation(req: RemediationRequest, db: Session = Depends(get_db)):
    a = db.query(Alert).filter(Alert.id == req.alert_id).first()
    if not a: raise HTTPException(404, "Alert not found")
    return auto_remediation.execute_remediation(db, a)


@app.get("/api/remediation/history", tags=["Remediation"])
def remediation_history(field_id: Optional[str] = None, hours: int = 24, db: Session = Depends(get_db)):
    return auto_remediation.get_remediation_history(db, field_id, hours)


# ════════════════════════════════════════════════════════════
# DASHBOARD STATS
# ════════════════════════════════════════════════════════════
@app.get("/api/dashboard/stats", tags=["Dashboard"])
def dashboard_stats(db: Session = Depends(get_db)):
    fields          = [r[0] for r in db.query(SensorData.field_id).distinct().all()]
    active_alerts   = db.query(Alert).filter(Alert.is_resolved == False).count()
    critical_alerts = db.query(Alert).filter(Alert.is_resolved == False, Alert.severity == "critical").count()
    hour_ago        = datetime.utcnow() - timedelta(hours=1)
    recent_readings = db.query(SensorData).filter(SensorData.timestamp >= hour_ago).count()
    rem_stats       = auto_remediation.get_remediation_history(db, hours=24)
    return {
        "total_fields":          len(fields),
        "active_alerts":         active_alerts,
        "critical_alerts":       critical_alerts,
        "recent_readings_1h":    recent_readings,
        "remediation_summary":   {"total_24h": rem_stats["total_remediations"], "cost_24h": rem_stats["total_cost_inr"]},
        "field_ids":             fields,
        "timestamp":             datetime.utcnow().isoformat(),
    }


# ════════════════════════════════════════════════════════════
# HEALTH
# ════════════════════════════════════════════════════════════
@app.get("/health", tags=["System"])
def health():
    return {"status": "healthy", "service": "AgriFarm Intelligence Platform v2.0", "timestamp": datetime.utcnow().isoformat()}


# ── Background task ───────────────────────────────────────────
def _bg_monitor(field_id: str):
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        r = anomaly_detector.monitor_and_alert(db, field_id)
        if r["alerts_created"] > 0:
            auto_remediation.process_alerts(db, field_id)
    finally:
        db.close()


if __name__ == "__main__":
    import os, uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8000)),
        reload=os.getenv("DEBUG", "false").lower() == "true",
    )
