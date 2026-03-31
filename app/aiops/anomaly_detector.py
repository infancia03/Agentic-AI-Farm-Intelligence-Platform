"""
AgriFarm — AIOps Anomaly Detector
YOUR original rule-based detection (kept intact) +
LLM-powered root-cause analysis layer added on top.
"""

from __future__ import annotations
import json
import os
import statistics
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session
from loguru import logger
from dotenv import load_dotenv

from app.database import SensorData, Alert

load_dotenv()


class AnomalyDetector:

    def __init__(self):
        self.temp_threshold     = float(os.getenv("ANOMALY_THRESHOLD_TEMP",     "35.0"))
        self.moisture_threshold = float(os.getenv("ANOMALY_THRESHOLD_MOISTURE", "30.0"))
        self.ph_high            = float(os.getenv("ANOMALY_THRESHOLD_PH_HIGH",  "8.5"))
        self.ph_low             = float(os.getenv("ANOMALY_THRESHOLD_PH_LOW",   "5.5"))

    # ── YOUR original rule engine (preserved exactly) ─────────
    def detect_anomalies(self, db: Session, field_id: str | None = None) -> list[dict]:
        anomalies: list[dict] = []
        q = db.query(SensorData)
        if field_id:
            q = q.filter(SensorData.field_id == field_id)
        cutoff = datetime.utcnow() - timedelta(minutes=5)
        for r in q.filter(SensorData.timestamp >= cutoff).all():
            if r.temperature > self.temp_threshold:
                anomalies.append({
                    "field_id": r.field_id, "type": "temperature_spike",
                    "severity": "high" if r.temperature > 40 else "medium",
                    "value": r.temperature, "threshold": self.temp_threshold,
                    "message": f"Temperature spike: {r.temperature}°C (threshold {self.temp_threshold}°C)",
                    "timestamp": r.timestamp,
                })
            if r.soil_moisture < self.moisture_threshold:
                anomalies.append({
                    "field_id": r.field_id, "type": "low_moisture",
                    "severity": "critical" if r.soil_moisture < 20 else "high",
                    "value": r.soil_moisture, "threshold": self.moisture_threshold,
                    "message": f"Low soil moisture: {r.soil_moisture}% (threshold {self.moisture_threshold}%)",
                    "timestamp": r.timestamp,
                })
            if not (self.ph_low <= r.ph_level <= self.ph_high):
                anomalies.append({
                    "field_id": r.field_id, "type": "ph_imbalance",
                    "severity": "medium",
                    "value": r.ph_level, "threshold": f"{self.ph_low}–{self.ph_high}",
                    "message": f"pH imbalance: {r.ph_level} (ideal {self.ph_low}–{self.ph_high})",
                    "timestamp": r.timestamp,
                })
            if r.nitrogen < 30:
                anomalies.append({
                    "field_id": r.field_id, "type": "nitrogen_deficiency",
                    "severity": "medium", "value": r.nitrogen, "threshold": 30,
                    "message": f"Nitrogen deficiency: {r.nitrogen} ppm (min 30 ppm)",
                    "timestamp": r.timestamp,
                })
            # Rapid-change detection (1-hour delta)
            prev = (
                db.query(SensorData)
                .filter(
                    SensorData.field_id == r.field_id,
                    SensorData.timestamp >= datetime.utcnow() - timedelta(hours=1),
                    SensorData.timestamp < r.timestamp,
                )
                .order_by(SensorData.timestamp.desc())
                .first()
            )
            if prev:
                if abs(r.temperature - prev.temperature) > 8:
                    anomalies.append({
                        "field_id": r.field_id, "type": "rapid_temperature_change",
                        "severity": "medium",
                        "value": abs(r.temperature - prev.temperature), "threshold": 8,
                        "message": f"Rapid temp change: {abs(r.temperature - prev.temperature):.1f}°C/h",
                        "timestamp": r.timestamp,
                    })
                drop = prev.soil_moisture - r.soil_moisture
                if drop > 15:
                    anomalies.append({
                        "field_id": r.field_id, "type": "rapid_moisture_drop",
                        "severity": "high", "value": drop, "threshold": 15,
                        "message": f"Rapid moisture drop: {drop:.1f}% in 1h — possible irrigation failure",
                        "timestamp": r.timestamp,
                    })
        return anomalies

    def create_alerts(self, db: Session, anomalies: list[dict]) -> list[Alert]:
        created: list[Alert] = []
        for a in anomalies:
            existing = (
                db.query(Alert)
                .filter(
                    Alert.field_id  == a["field_id"],
                    Alert.alert_type == a["type"],
                    Alert.is_resolved == False,
                    Alert.timestamp >= datetime.utcnow() - timedelta(hours=1),
                )
                .first()
            )
            if not existing:
                alert = Alert(
                    field_id=a["field_id"], severity=a["severity"],
                    alert_type=a["type"], message=a["message"],
                    is_resolved=False, auto_remediation_applied=False,
                )
                db.add(alert)
                created.append(alert)
        db.commit()
        return created

    def monitor_and_alert(self, db: Session, field_id: str | None = None) -> dict:
        anomalies = self.detect_anomalies(db, field_id)
        alerts    = self.create_alerts(db, anomalies)
        return {
            "anomalies_detected": len(anomalies),
            "alerts_created":     len(alerts),
            "anomalies":          anomalies,
            "timestamp":          datetime.utcnow().isoformat(),
        }

    def get_trend_analysis(self, db: Session, field_id: str, hours: int = 24) -> dict:
        rows = (
            db.query(SensorData)
            .filter(
                SensorData.field_id  == field_id,
                SensorData.timestamp >= datetime.utcnow() - timedelta(hours=hours),
            )
            .order_by(SensorData.timestamp)
            .all()
        )
        if len(rows) < 2:
            return {"error": "Insufficient data for trend analysis"}

        temps     = [r.temperature   for r in rows]
        moistures = [r.soil_moisture for r in rows]

        def _trend(vals, up_delta=3, down_delta=3):
            if vals[-1] > vals[0] + up_delta:   return "increasing"
            if vals[-1] < vals[0] - down_delta:  return "decreasing"
            return "stable"

        temp_trend     = _trend(temps)
        moisture_trend = _trend(moistures, up_delta=5, down_delta=5)
        predictions    = []

        if moisture_trend == "decreasing" and moistures[-1] < 40:
            delta = moistures[0] - moistures[-1]
            eta   = ((moistures[-1] - 25) / (delta / hours)) if delta > 0 else 0
            predictions.append({
                "type": "irrigation_needed",
                "estimated_time_hours": round(max(0, eta), 1),
                "confidence": "high",
            })
        if temp_trend == "increasing" and temps[-1] > 30:
            predictions.append({
                "type": "heat_stress_risk",
                "estimated_time_hours": 2,
                "confidence": "medium",
            })

        return {
            "field_id":            field_id,
            "period_hours":        hours,
            "temperature_trend":   temp_trend,
            "moisture_trend":      moisture_trend,
            "current_temp":        temps[-1],
            "current_moisture":    moistures[-1],
            "predictions":         predictions,
            "data_points_analyzed": len(rows),
        }

    # ── NEW: LLM-powered anomaly analysis (from AgriGenie) ───
    def llm_analyse(self, db: Session, field_id: str | None = None) -> dict:
        """
        Collect recent anomalies + system metrics and ask the LLM
        for root-cause analysis and remediation recommendations.
        """
        from app.llm_client import quick_ask
        import psutil

        anomalies = self.detect_anomalies(db, field_id)
        cpu_pct   = psutil.cpu_percent(interval=1)
        mem_pct   = psutil.virtual_memory().percent

        if not anomalies and cpu_pct < 70 and mem_pct < 70:
            return {
                "status":   "healthy",
                "analysis": "All systems nominal. No anomalies detected.",
                "actions":  [],
            }

        summary = (
            f"Farm anomalies detected: {len(anomalies)}\n"
            + "\n".join(f"  [{a['severity'].upper()}] {a['field_id']}: {a['message']}" for a in anomalies[:8])
            + f"\nSystem: CPU {cpu_pct}%, RAM {mem_pct}%"
        )

        prompt = (
            f"{summary}\n\n"
            "As an AIOps + agri-AI engineer, respond in JSON:\n"
            "{\n"
            '  "severity": "critical|high|medium|low",\n'
            '  "root_cause": "...",\n'
            '  "actions": ["step1", "step2", "step3"],\n'
            '  "prevention": "...",\n'
            '  "estimated_resolution_mins": 5\n'
            "}"
        )
        try:
            raw = quick_ask(
                prompt,
                system="You are a senior AIOps and precision-agriculture engineer. Be concise and actionable.",
            )
            clean  = raw.strip().replace("```json","").replace("```","").strip()
            parsed = json.loads(clean)
            return {"status": "anomaly_detected", "analysis": parsed,
                    "actions": parsed.get("actions",[]), "raw_anomalies": anomalies}
        except Exception as e:
            logger.error(f"LLM anomaly analysis failed: {e}")
            return {
                "status":       "anomaly_detected",
                "analysis":     f"{len(anomalies)} anomalies detected. LLM analysis failed: {e}",
                "actions":      ["Check field sensors", "Review alert log", "Restart monitoring"],
                "raw_anomalies": anomalies,
            }
