"""
AgriFarm — Test Suite
Run: pytest tests/ -v
"""

from __future__ import annotations
import io, json, sys
from pathlib import Path
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ─────────────────────────────────────────────────
@pytest.fixture(scope="session")
def tmp_db(tmp_path_factory):
    """In-memory SQLite for tests."""
    import os
    db_path = tmp_path_factory.mktemp("db") / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    from app.database import init_db, SessionLocal
    init_db()
    return SessionLocal


@pytest.fixture
def db_session(tmp_db):
    db = tmp_db()
    yield db
    db.close()


# ── Database ──────────────────────────────────────────────────
def test_database_init(tmp_db):
    db = tmp_db()
    from app.database import SensorData
    db.query(SensorData).count()   # should not raise
    db.close()


def test_sensor_insert(db_session):
    from app.database import SensorData
    row = SensorData(
        field_id="test_field", temperature=28.5, soil_moisture=55.0,
        ph_level=6.8, nitrogen=70.0, phosphorus=30.0, potassium=90.0,
        humidity=65.0, rainfall_mm=0.0,
    )
    db_session.add(row)
    db_session.commit()
    assert row.id is not None
    assert row.timestamp is not None


# ── Seed data ─────────────────────────────────────────────────
def test_seed_data_generates_csvs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    import os; os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test_seed.db"

    from data.seed_data import seed_market_prices, seed_yield_history, seed_knowledge_base
    import pandas as pd

    seed_market_prices()
    df = pd.read_csv(tmp_path / "data" / "market_prices.csv")
    assert len(df) == 365 * 7
    assert "price_per_quintal_inr" in df.columns

    seed_yield_history()
    df2 = pd.read_csv(tmp_path / "data" / "yield_history.csv")
    assert len(df2) > 0

    seed_knowledge_base()
    kb = json.loads((tmp_path / "data" / "agri_knowledge.json").read_text())
    assert len(kb) > 10
    assert all("content" in d for d in kb)


# ── Sensor Tools ──────────────────────────────────────────────
def test_sensor_tools_no_data(db_session):
    from app.tools.sensor_tools import SensorTools
    result = SensorTools.get_latest_sensor_data(db_session, "nonexistent_field")
    assert "error" in result


def test_sensor_tools_with_data(db_session):
    from app.database import SensorData
    from app.tools.sensor_tools import SensorTools
    db_session.add(SensorData(
        field_id="tool_field", temperature=30.0, soil_moisture=35.0,
        ph_level=6.5, nitrogen=25.0, phosphorus=28.0, potassium=85.0,
        humidity=70.0, rainfall_mm=0.0,
    ))
    db_session.commit()

    # get_latest
    r = SensorTools.get_latest_sensor_data(db_session, "tool_field")
    assert r["temperature"] == 30.0

    # soil health — nitrogen is 25 (deficient)
    health = SensorTools.analyze_soil_health(db_session, "tool_field")
    assert health["nutrient_status"]["nitrogen"] == "deficient"
    assert len(health["recommendations"]) > 0

    # irrigation — moisture 35 < 45 → needed
    irr = SensorTools.check_irrigation_efficiency(db_session, "tool_field")
    assert irr["irrigation_needed"] is True
    assert irr["urgency"] in ("moderate", "critical")

    # pest — temp 30, moisture 35 < 40 → spider mites risk
    pest = SensorTools.detect_pest_patterns(db_session, "tool_field")
    assert pest["overall_pest_risk"] in ("moderate", "high")


# ── Disease Agent ─────────────────────────────────────────────
def test_disease_model_loads():
    from app.agents.disease_agent import _load_model
    model = _load_model()
    assert model is not None
    import torch
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 1_000_000


def test_disease_classify_synthetic_image():
    from PIL import Image
    from app.agents.disease_agent import classify_image
    import numpy as np

    arr  = (np.random.rand(224, 224, 3) * 255).astype("uint8")
    img  = Image.fromarray(arr)
    buf  = io.BytesIO(); img.save(buf, format="JPEG"); buf.seek(0)
    result = classify_image(buf.read())

    assert "crop"       in result
    assert "disease"    in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1
    assert len(result["top3"]) == 3


def test_disease_class_parsing():
    from app.agents.disease_agent import _parse
    crop, disease = _parse("Tomato__Early_blight")
    assert crop    == "Tomato"
    assert "Early" in disease

    crop2, disease2 = _parse("Potato__healthy")
    assert crop2    == "Potato"
    assert "healthy" in disease2.lower()


# ── Yield Agent ───────────────────────────────────────────────
def test_yield_forecast_synthetic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    from app.agents.yield_market_agent import forecast_yield
    result = forecast_yield("Tomato", "Tamil Nadu", 2)

    assert result["crop"]          == "Tomato"
    assert len(result["forecasts"]) == 2
    assert result["avg_yield"]      > 0
    for fc in result["forecasts"]:
        assert "year"      in fc
        assert "predicted" in fc
        assert fc["predicted"] > 0


def test_yield_forecast_from_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    from data.seed_data import seed_yield_history
    seed_yield_history()

    monkeypatch.setattr("app.agents.yield_market_agent.YIELD_CSV", tmp_path / "data" / "yield_history.csv")
    from app.agents.yield_market_agent import forecast_yield
    result = forecast_yield("Rice", "Punjab", 3)
    assert len(result["forecasts"]) == 3


# ── Market Agent ──────────────────────────────────────────────
def test_market_stats_synthetic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    from app.agents.yield_market_agent import get_price_stats
    stats = get_price_stats("Tomato")
    assert stats["current_price_inr"]  > 0
    assert stats["price_signal"] in ("BUY", "HOLD", "SELL")


def test_market_stats_from_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    from data.seed_data import seed_market_prices
    seed_market_prices()
    monkeypatch.setattr("app.agents.yield_market_agent.MARKET_CSV", tmp_path / "data" / "market_prices.csv")

    from app.agents.yield_market_agent import get_price_stats
    stats = get_price_stats("Wheat")
    assert stats["avg_30d"]   > 0
    assert stats["avg_1y"]    > 0
    assert stats["min_30d"]   <= stats["max_30d"]


# ── AIOps ─────────────────────────────────────────────────────
def test_anomaly_detector_no_recent_data(db_session):
    """With no recent sensor data, anomaly detector should return empty list."""
    from app.aiops.anomaly_detector import AnomalyDetector
    detector   = AnomalyDetector()
    anomalies  = detector.detect_anomalies(db_session, "empty_field")
    assert anomalies == []


def test_anomaly_detector_detects_low_moisture(db_session):
    from app.database import SensorData
    from app.aiops.anomaly_detector import AnomalyDetector

    db_session.add(SensorData(
        field_id="anomaly_test", temperature=25.0, soil_moisture=18.0,  # BELOW 30 threshold
        ph_level=6.8, nitrogen=70.0, phosphorus=30.0, potassium=90.0,
        humidity=65.0, rainfall_mm=0.0,
        timestamp=datetime.utcnow(),
    ))
    db_session.commit()

    detector  = AnomalyDetector()
    anomalies = detector.detect_anomalies(db_session, "anomaly_test")
    types     = [a["type"] for a in anomalies]
    assert "low_moisture" in types
    low_m = next(a for a in anomalies if a["type"] == "low_moisture")
    assert low_m["severity"] == "critical"


def test_anomaly_creates_alerts(db_session):
    from app.database import SensorData
    from app.aiops.anomaly_detector import AnomalyDetector

    db_session.add(SensorData(
        field_id="alert_test", temperature=42.0,  # ABOVE 35 threshold
        soil_moisture=55.0, ph_level=6.8,
        nitrogen=70.0, phosphorus=30.0, potassium=90.0,
        humidity=65.0, rainfall_mm=0.0,
        timestamp=datetime.utcnow(),
    ))
    db_session.commit()

    detector = AnomalyDetector()
    result   = detector.monitor_and_alert(db_session, "alert_test")
    assert result["anomalies_detected"] > 0
    assert result["alerts_created"]     > 0


# ── Auto Remediation ──────────────────────────────────────────
def test_auto_remediation_should_remediate(db_session):
    from app.database import Alert
    from app.aiops.auto_remediation import AutoRemediationEngine

    alert = Alert(
        field_id="rem_test", severity="critical",
        alert_type="low_moisture", message="Test low moisture",
    )
    db_session.add(alert); db_session.commit()

    engine = AutoRemediationEngine()
    assert engine.should_remediate(alert) is True


def test_auto_remediation_execute(db_session):
    from app.database import Alert
    from app.aiops.auto_remediation import AutoRemediationEngine

    alert = Alert(
        field_id="exec_test", severity="high",
        alert_type="temperature_spike", message="Temp spike test",
    )
    db_session.add(alert); db_session.commit()

    engine = AutoRemediationEngine()
    result = engine.execute_remediation(db_session, alert)
    assert result["executed"]    is True
    assert result["cost_inr"]    >= 0


# ── LLM Client ────────────────────────────────────────────────
def test_llm_client_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import app.llm_client as lc
    lc._client = None
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        lc.get_client()


# ── FastAPI endpoints (no LLM calls) ─────────────────────────
def test_health_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_list_fields_empty():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    r = client.get("/api/sensors/fields")
    assert r.status_code == 200
    assert "fields" in r.json()


def test_dashboard_stats():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    r = client.get("/api/dashboard/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_fields"   in data
    assert "active_alerts"  in data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
