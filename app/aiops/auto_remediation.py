"""
AgriFarm — Auto-Remediation Engine
YOUR original engine — kept 100% intact.
"""

from __future__ import annotations
import os
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session
from loguru import logger
from dotenv import load_dotenv

from app.database import Alert, RemediationLog
from app.agents.action_agent import ActionAgent

load_dotenv()


class AutoRemediationEngine:

    def __init__(self):
        self.action_agent = ActionAgent()
        self.enabled      = os.getenv("AUTO_REMEDIATION_ENABLED", "true").lower() == "true"

        # Rule table — maps alert_type → {action, severity_threshold, default_params}
        self.rules = {
            "low_moisture": {
                "action":             "trigger_irrigation",
                "severity_threshold": "high",
                "parameters":         {"water_amount_liters": 300, "duration_minutes": 45},
            },
            "temperature_spike": {
                "action":             "activate_cooling_system",
                "severity_threshold": "high",
                "parameters":         {"cooling_method": "misting", "duration_hours": 2},
            },
            "nitrogen_deficiency": {
                "action":             "apply_fertilizer",
                "severity_threshold": "medium",
                "parameters":         {"fertilizer_type": "nitrogen", "amount_kg": 25},
            },
            "ph_imbalance": {
                "action":             "send_farmer_alert",
                "severity_threshold": "medium",
                "parameters":         {"message": "pH imbalance — manual soil amendment recommended.", "urgency": "medium"},
            },
            "rapid_moisture_drop": {
                "action":             "send_farmer_alert",
                "severity_threshold": "high",
                "parameters":         {"message": "Rapid moisture drop — possible irrigation failure.", "urgency": "critical"},
            },
            "rapid_temperature_change": {
                "action":             "send_farmer_alert",
                "severity_threshold": "medium",
                "parameters":         {"message": "Rapid temperature change detected. Monitor closely.", "urgency": "high"},
            },
        }

        self._severity_rank = {"low": 1, "medium": 2, "high": 3, "critical": 4}

    def should_remediate(self, alert: Alert) -> bool:
        if not self.enabled:
            return False
        rule = self.rules.get(alert.alert_type)
        if not rule:
            return False
        return (
            self._severity_rank.get(alert.severity, 0)
            >= self._severity_rank.get(rule["severity_threshold"], 0)
        )

    def execute_remediation(self, db: Session, alert: Alert) -> dict:
        if not self.should_remediate(alert):
            return {"executed": False, "reason": "Does not meet auto-remediation criteria"}

        rule    = self.rules[alert.alert_type]
        action  = rule["action"]
        params  = dict(rule["parameters"])
        params["field_id"] = alert.field_id
        params["alert_id"] = alert.id

        result = self.action_agent._execute_action(action, params, db)

        alert.auto_remediation_applied = True
        alert.remediation_action       = f"{action}: {result.get('action','')}"
        db.commit()

        return {
            "executed":    True,
            "alert_id":    alert.id,
            "action_type": action,
            "result":      result,
            "cost_inr":    result.get("cost_inr", 0),
        }

    def process_alerts(self, db: Session, field_id: str | None = None) -> dict:
        q = db.query(Alert).filter(
            Alert.is_resolved == False,
            Alert.auto_remediation_applied == False,
        )
        if field_id:
            q = q.filter(Alert.field_id == field_id)

        results:   list[dict] = []
        total_cost: float     = 0.0

        for alert in q.all():
            r = self.execute_remediation(db, alert)
            if r["executed"]:
                results.append(r)
                total_cost += r.get("cost_inr", 0)

        return {
            "total_alerts_processed":  len(results) + (q.count() - len(results)),
            "remediations_executed":   len(results),
            "total_cost_inr":          round(total_cost, 2),
            "results":                 results,
            "timestamp":               datetime.utcnow().isoformat(),
        }

    def get_remediation_history(self, db: Session, field_id: str | None = None, hours: int = 24) -> dict:
        q = db.query(RemediationLog).filter(
            RemediationLog.timestamp >= datetime.utcnow() - timedelta(hours=hours)
        )
        if field_id:
            q = q.filter(RemediationLog.field_id == field_id)
        logs = q.order_by(RemediationLog.timestamp.desc()).all()

        action_counts: dict[str, int] = {}
        for log in logs:
            action_counts[log.action_type] = action_counts.get(log.action_type, 0) + 1

        return {
            "total_remediations": len(logs),
            "total_cost_inr":     round(sum(l.cost_estimate or 0 for l in logs), 2),
            "action_breakdown":   action_counts,
            "recent_actions": [
                {
                    "timestamp":   l.timestamp.isoformat(),
                    "field_id":    l.field_id,
                    "action_type": l.action_type,
                    "success":     l.success,
                    "cost_inr":    l.cost_estimate,
                }
                for l in logs[:10]
            ],
        }
