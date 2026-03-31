"""AgriFarm — Pydantic request/response schemas."""

from __future__ import annotations
from datetime import datetime
from typing import Optional, List, Any
from pydantic import BaseModel, Field


# ── Sensor ────────────────────────────────────────────────────
class SensorDataCreate(BaseModel):
    field_id:      str
    temperature:   float = Field(..., ge=-10,  le=60)
    soil_moisture: float = Field(..., ge=0,    le=100)
    ph_level:      float = Field(..., ge=0,    le=14)
    nitrogen:      float = Field(..., ge=0,    le=500)
    phosphorus:    float = Field(..., ge=0,    le=500)
    potassium:     float = Field(..., ge=0,    le=500)
    humidity:      float = Field(..., ge=0,    le=100)
    rainfall_mm:   float = Field(0.0, ge=0)


class SensorDataResponse(SensorDataCreate):
    id:        int
    timestamp: datetime
    class Config:
        from_attributes = True


# ── Alert ─────────────────────────────────────────────────────
class AlertResponse(BaseModel):
    id:                       int
    field_id:                 str
    timestamp:                datetime
    severity:                 str
    alert_type:               str
    message:                  str
    is_resolved:              bool
    auto_remediation_applied: bool
    remediation_action:       Optional[str] = None
    class Config:
        from_attributes = True


# ── Agent ─────────────────────────────────────────────────────
class AgentQuery(BaseModel):
    query:        str
    field_id:     Optional[str] = None
    farm_context: dict           = {}


class RemediationRequest(BaseModel):
    alert_id:        int
    action_type:     str
    manual_override: bool = False


# ── Disease scan ──────────────────────────────────────────────
class DiseaseScanResponse(BaseModel):
    crop:       str
    disease:    str
    confidence: float
    is_healthy: bool
    top3:       List[dict]
    llm_advice: str
    agent:      str


# ── Market ────────────────────────────────────────────────────
class MarketQueryResponse(BaseModel):
    crop:      str
    stats:     dict
    advisory:  str
    rag_used:  bool
    agent:     str


# ── AIOps ─────────────────────────────────────────────────────
class AIOpsStatus(BaseModel):
    status:         str
    cpu_pct:        float
    memory_pct:     float
    agent_runs_1h:  int
    timestamp:      str
