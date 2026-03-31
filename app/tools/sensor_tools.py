"""
AgriFarm — Sensor Tools
Function-calling tools used by the Diagnostic Agent.
Kept from original Smart Farm project; extended with RAG lookup.
"""

from __future__ import annotations
import statistics
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session
from app.database import SensorData, Alert


class SensorTools:
    """All tools callable by the LLM diagnostic agent."""

    # ── Tool: latest sensor reading ──────────────────────────
    @staticmethod
    def get_latest_sensor_data(db: Session, field_id: str) -> dict:
        row = (
            db.query(SensorData)
            .filter(SensorData.field_id == field_id)
            .order_by(SensorData.timestamp.desc())
            .first()
        )
        if not row:
            return {"error": f"No data for field {field_id}"}
        return {
            "field_id":      row.field_id,
            "timestamp":     row.timestamp.isoformat(),
            "temperature":   row.temperature,
            "soil_moisture": row.soil_moisture,
            "ph_level":      row.ph_level,
            "nitrogen":      row.nitrogen,
            "phosphorus":    row.phosphorus,
            "potassium":     row.potassium,
            "humidity":      row.humidity,
            "rainfall_mm":   row.rainfall_mm,
        }

    # ── Tool: soil health analysis ────────────────────────────
    @staticmethod
    def analyze_soil_health(db: Session, field_id: str) -> dict:
        row = (
            db.query(SensorData)
            .filter(SensorData.field_id == field_id)
            .order_by(SensorData.timestamp.desc())
            .first()
        )
        if not row:
            return {"error": f"No data for field {field_id}"}

        IDEAL = {
            "nitrogen":   (40, 150),
            "phosphorus": (15, 60),
            "potassium":  (40, 120),
            "ph_level":   (6.0, 7.5),
        }
        status = {}
        recs   = []

        def _check(name, val, lo, hi, low_msg, high_msg):
            if val < lo:
                status[name] = "deficient"
                recs.append(low_msg)
            elif val > hi:
                status[name] = "excess"
                recs.append(high_msg)
            else:
                status[name] = "optimal"

        _check("nitrogen",   row.nitrogen,   *IDEAL["nitrogen"],
               "Apply urea or ammonium sulphate (nitrogen fertiliser)",
               "Reduce N application — risk of leaf burn and groundwater leaching")
        _check("phosphorus", row.phosphorus, *IDEAL["phosphorus"],
               "Apply DAP or SSP (phosphorus fertiliser)",
               "Reduce phosphorus application")
        _check("potassium",  row.potassium,  *IDEAL["potassium"],
               "Apply MOP or SOP (potassium fertiliser)",
               "Reduce potassium application")

        if row.ph_level < IDEAL["ph_level"][0]:
            status["ph"] = "acidic"
            recs.append("Apply agricultural lime to raise pH")
        elif row.ph_level > IDEAL["ph_level"][1]:
            status["ph"] = "alkaline"
            recs.append("Apply elemental sulphur or organic compost to lower pH")
        else:
            status["ph"] = "optimal"

        overall = "healthy" if all(v == "optimal" for v in status.values()) else "needs_attention"
        return {
            "field_id":        field_id,
            "overall_health":  overall,
            "nutrient_status": status,
            "current_values": {
                "nitrogen":   row.nitrogen,
                "phosphorus": row.phosphorus,
                "potassium":  row.potassium,
                "ph_level":   row.ph_level,
            },
            "recommendations": recs,
        }

    # ── Tool: irrigation check ────────────────────────────────
    @staticmethod
    def check_irrigation_efficiency(db: Session, field_id: str) -> dict:
        row = (
            db.query(SensorData)
            .filter(SensorData.field_id == field_id)
            .order_by(SensorData.timestamp.desc())
            .first()
        )
        if not row:
            return {"error": f"No data for field {field_id}"}

        hist = (
            db.query(SensorData)
            .filter(
                SensorData.field_id == field_id,
                SensorData.timestamp >= datetime.utcnow() - timedelta(hours=24),
            )
            .order_by(SensorData.timestamp)
            .all()
        )

        trend = "stable"
        if len(hist) >= 5:
            recent = [h.soil_moisture for h in hist[-5:]]
            if all(recent[i] > recent[i + 1] for i in range(4)):
                trend = "decreasing"
            elif all(recent[i] < recent[i + 1] for i in range(4)):
                trend = "increasing"

        needed, urgency, litres = False, "none", 0
        m = row.soil_moisture
        if m < 30:
            needed, urgency, litres = True, "critical", 500
        elif m < 45:
            needed, urgency, litres = True, "moderate", 300
        elif m < 60 and trend == "decreasing":
            needed, urgency, litres = True, "low", 200

        return {
            "field_id":                    field_id,
            "current_moisture":            m,
            "moisture_trend":              trend,
            "irrigation_needed":           needed,
            "urgency":                     urgency,
            "estimated_water_litres":      litres,
            "estimated_cost_inr":          round(litres * 0.05, 2),
        }

    # ── Tool: pest risk detection ─────────────────────────────
    @staticmethod
    def detect_pest_patterns(db: Session, field_id: str) -> dict:
        row = (
            db.query(SensorData)
            .filter(SensorData.field_id == field_id)
            .order_by(SensorData.timestamp.desc())
            .first()
        )
        if not row:
            return {"error": f"No data for field {field_id}"}

        risks = []
        if row.temperature > 30 and row.humidity > 75:
            risks.append({
                "type": "fungal_disease",
                "risk_level": "high",
                "conditions": "High temp + high humidity favour fungal growth",
                "recommendation": "Apply preventive fungicide; improve air circulation",
            })
        if row.soil_moisture < 40 and row.temperature > 32:
            risks.append({
                "type": "spider_mites",
                "risk_level": "moderate",
                "conditions": "Hot and dry conditions favour spider mites",
                "recommendation": "Increase irrigation; inspect leaf undersides daily",
            })
        if row.nitrogen > 150:
            risks.append({
                "type": "aphids",
                "risk_level": "moderate",
                "conditions": "Excess nitrogen causes lush growth that attracts aphids",
                "recommendation": "Reduce N fertilisation; introduce beneficial insects",
            })

        overall = "high" if any(r["risk_level"] == "high" for r in risks) else ("moderate" if risks else "low")
        return {
            "field_id":           field_id,
            "overall_pest_risk":  overall,
            "detected_risks":     risks,
            "conditions":         {
                "temperature":   row.temperature,
                "humidity":      row.humidity,
                "soil_moisture": row.soil_moisture,
                "nitrogen":      row.nitrogen,
            },
        }

    # ── Tool: field history ───────────────────────────────────
    @staticmethod
    def get_field_history(db: Session, field_id: str, hours: int = 24) -> dict:
        rows = (
            db.query(SensorData)
            .filter(
                SensorData.field_id == field_id,
                SensorData.timestamp >= datetime.utcnow() - timedelta(hours=hours),
            )
            .order_by(SensorData.timestamp)
            .all()
        )
        if not rows:
            return {"error": f"No history for field {field_id}"}

        temps     = [r.temperature   for r in rows]
        moistures = [r.soil_moisture for r in rows]
        return {
            "field_id":     field_id,
            "period_hours": hours,
            "data_points":  len(rows),
            "temperature_stats": {
                "min": round(min(temps), 2),    "max": round(max(temps), 2),
                "avg": round(statistics.mean(temps), 2), "current": temps[-1],
            },
            "moisture_stats": {
                "min": round(min(moistures), 2), "max": round(max(moistures), 2),
                "avg": round(statistics.mean(moistures), 2), "current": moistures[-1],
            },
        }

    # ── Tool: active alerts ───────────────────────────────────
    @staticmethod
    def get_active_alerts(db: Session, field_id: str | None = None) -> dict:
        q = db.query(Alert).filter(Alert.is_resolved == False)
        if field_id:
            q = q.filter(Alert.field_id == field_id)
        alerts = q.order_by(Alert.timestamp.desc()).all()
        return {
            "total_active_alerts": len(alerts),
            "alerts": [
                {
                    "id":          a.id,
                    "field_id":    a.field_id,
                    "severity":    a.severity,
                    "type":        a.alert_type,
                    "message":     a.message,
                    "timestamp":   a.timestamp.isoformat(),
                    "remediated":  a.auto_remediation_applied,
                }
                for a in alerts
            ],
        }

    # ── Tool: knowledge base lookup (RAG) ─────────────────────
    @staticmethod
    def lookup_knowledge_base(db: Session, query: str) -> dict:
        """
        Semantic search over the agri knowledge base via ChromaDB.
        Gives the LLM access to treatment protocols, irrigation guides, etc.
        """
        try:
            from rag.retriever import query_knowledge_base
            docs = query_knowledge_base(query, n_results=2)
            if docs:
                return {
                    "results": [{"content": d["content"][:500], "title": d["metadata"].get("title", "")} for d in docs],
                    "source": "ChromaDB knowledge base",
                }
            return {"results": [], "source": "ChromaDB knowledge base"}
        except Exception as e:
            return {"error": str(e), "results": []}

    # ── Tool schema definitions (for LLM function calling) ────
    TOOL_SCHEMAS = [
        {
            "type": "function",
            "function": {
                "name": "get_latest_sensor_data",
                "description": "Get the most recent IoT sensor reading for a specific farm field",
                "parameters": {
                    "type": "object",
                    "properties": {"field_id": {"type": "string", "description": "Field ID e.g. field_A1"}},
                    "required": ["field_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_soil_health",
                "description": "Analyse NPK levels and pH for a field and return health status + recommendations",
                "parameters": {
                    "type": "object",
                    "properties": {"field_id": {"type": "string"}},
                    "required": ["field_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_irrigation_efficiency",
                "description": "Check whether irrigation is needed based on current and historic soil moisture",
                "parameters": {
                    "type": "object",
                    "properties": {"field_id": {"type": "string"}},
                    "required": ["field_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "detect_pest_patterns",
                "description": "Detect potential pest or disease risks based on environmental conditions",
                "parameters": {
                    "type": "object",
                    "properties": {"field_id": {"type": "string"}},
                    "required": ["field_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_field_history",
                "description": "Get historical sensor readings for trend analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "field_id": {"type": "string"},
                        "hours":    {"type": "integer", "description": "Hours of history (default 24)"},
                    },
                    "required": ["field_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_active_alerts",
                "description": "Get all unresolved alerts for a field or all fields",
                "parameters": {
                    "type": "object",
                    "properties": {"field_id": {"type": "string", "description": "Optional field ID"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup_knowledge_base",
                "description": "Search the agri knowledge base for disease treatments, irrigation guides, and market strategies",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Natural language search query"}},
                    "required": ["query"],
                },
            },
        },
    ]
