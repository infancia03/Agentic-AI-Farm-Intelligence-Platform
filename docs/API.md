# 📡 AgriFarm API Reference

Base URL: `http://localhost:8000` (local) or `http://YOUR_EC2_IP` (deployed)

Interactive docs: `GET /docs` (Swagger UI)

---

## Authentication
No authentication required for local/demo use. For production, add an API key header.

---

## Sensors

### POST `/api/sensors/data`
Submit a new IoT sensor reading. Triggers background AIOps anomaly detection automatically.

**Request body:**
```json
{
  "field_id":      "field_A1",
  "temperature":   28.5,
  "soil_moisture": 45.2,
  "ph_level":      6.8,
  "nitrogen":      75.0,
  "phosphorus":    32.0,
  "potassium":     88.0,
  "humidity":      68.0,
  "rainfall_mm":   2.5
}
```

**Response:** `201` — created sensor record with `id` and `timestamp`

---

### GET `/api/sensors/data/{field_id}?hours=24`
Historical sensor readings for a field.

**Parameters:**
- `field_id` — e.g. `field_A1`
- `hours` — integer, default 24

**Response:** Array of sensor records ordered by timestamp descending.

---

### GET `/api/sensors/latest/{field_id}`
Most recent sensor reading for a field.

**Response:** Single sensor record or `404` if no data.

---

### GET `/api/sensors/fields`
List all fields that have sensor data.

**Response:**
```json
{ "fields": ["field_A1", "field_A2", "field_B1"], "total": 3 }
```

---

## Agentic AI

### POST `/api/agent/query`
Main agentic endpoint. The Orchestrator automatically classifies intent and routes to the correct agent.

**Request body:**
```json
{
  "query":        "What is the soil health of field_A1?",
  "field_id":     "field_A1",
  "farm_context": {
    "crop":  "Tomato",
    "state": "Tamil Nadu"
  }
}
```

**Intent routing:**
- Keywords like *disease, blight, leaf, infected, scan* → Disease Agent
- Keywords like *yield, forecast, harvest, production* → Yield Agent
- Keywords like *price, market, sell, mandi* → Market Agent
- Everything else → Diagnostic + Action Agent (your original 2-phase pipeline)

**Response:**
```json
{
  "orchestrator_summary": "**Diagnostic Summary:** ...",
  "diagnostic_phase": {
    "response":       "Soil nitrogen is deficient at 28 ppm...",
    "tools_used":     ["analyze_soil_health", "get_latest_sensor_data"],
    "execution_time": 4.2,
    "iterations":     3
  },
  "action_phase": {
    "response":       "Applied 25kg nitrogen fertiliser...",
    "actions_taken":  [{"action": "apply_fertilizer", "result": {...}}],
    "total_actions":  1,
    "total_cost_inr": 1125.0
  },
  "total_execution_time": 8.1,
  "intent":   "diagnostic",
  "field_id": "field_A1"
}
```

---

### GET `/api/agent/recommendations/{field_id}`
Comprehensive field health report without auto-remediation.

**Response:** Same format as `/api/agent/query` but with `auto_remediate=false`.

---

## Disease Detection

### POST `/api/disease/detect`
Upload a leaf image. ResNet18 classifies among 38 PlantVillage disease classes. LLM generates treatment plan.

**Request:** `multipart/form-data`
- `file` — image file (jpg, png, webp)
- `crop` — string, e.g. "Tomato" (optional hint)
- `field_id` — string (optional)
- `location` — string, e.g. "Tamil Nadu, India"

**Response:**
```json
{
  "crop":       "Tomato",
  "disease":    "Early blight",
  "confidence": 0.8732,
  "is_healthy": false,
  "top3": [
    {"crop": "Tomato", "disease": "Early blight",  "confidence": 0.8732},
    {"crop": "Tomato", "disease": "Late blight",   "confidence": 0.0891},
    {"crop": "Tomato", "disease": "Target Spot",   "confidence": 0.0214}
  ],
  "llm_advice": "**SEVERITY: High**\n\n**IMMEDIATE ACTIONS:**\nApply copper-based...",
  "agent":      "disease_agent"
}
```

---

## Yield & Market

### GET `/api/yield/forecast?crop=Tomato&state=Tamil+Nadu&years=3`
Facebook Prophet time-series yield forecast with LLM narrative.

**Parameters:**
- `crop`  — e.g. Tomato, Potato, Rice, Wheat, Corn, Soybean, Cotton
- `state` — Indian state name
- `years` — forecast horizon (default 3)

**Response:**
```json
{
  "crop":      "Tomato",
  "state":     "Tamil Nadu",
  "historical": [
    {"year": 2015, "yield_tonnes_ha": 23.4},
    ...
  ],
  "forecasts": [
    {"year": 2025, "predicted": 28.1, "lower": 24.7, "upper": 31.5},
    ...
  ],
  "avg_yield":       25.3,
  "trend_pct":       12.4,
  "best_year":       2023,
  "next_year_pred":  28.1,
  "method":          "Prophet",
  "narrative":       "Tomato yield in Tamil Nadu shows a consistent upward trend...",
  "agent":           "yield_agent"
}
```

---

### GET `/api/market/advisory?crop=Rice&quantity_quintals=20`
RAG-enhanced market price advisory.

**Parameters:**
- `crop` — crop name
- `quantity_quintals` — how much the farmer wants to sell (default 10)

**Response:**
```json
{
  "crop": "Rice",
  "stats": {
    "current_price_inr": 38.5,
    "avg_30d":           36.2,
    "avg_1y":            34.8,
    "min_30d":           32.1,
    "max_30d":           42.3,
    "trend_7d":          2.1,
    "price_signal":      "SELL",
    "monthly":           [{"month": "2024-01", "avg_price": 33.2}, ...]
  },
  "advisory":  "**RECOMMENDATION: SELL NOW**\nRice is trading 10.6% above...",
  "rag_used":  true,
  "agent":     "market_agent"
}
```

---

## AIOps

### POST `/api/aiops/monitor`
Trigger rule-based anomaly detection and auto-remediation.

**Request body (optional):**
```json
{ "field_id": "field_A1" }
```
Omit `field_id` to monitor all fields.

**Response:**
```json
{
  "anomalies_detected": 3,
  "alerts_created":     2,
  "anomalies": [
    {
      "field_id":  "field_A1",
      "type":      "low_moisture",
      "severity":  "critical",
      "value":     22.3,
      "threshold": 30.0,
      "message":   "Low soil moisture: 22.3% (threshold 30%)"
    }
  ],
  "auto_remediation": {
    "remediations_executed": 1,
    "total_cost_inr":        15.0
  }
}
```

---

### POST `/api/aiops/llm-analyse`
LLM root-cause analysis of current anomalies. Returns structured JSON with severity and actionable steps.

**Response:**
```json
{
  "status": "anomaly_detected",
  "analysis": {
    "severity":                    "high",
    "root_cause":                  "field_A1 soil moisture critically low — likely irrigation pump failure or blocked drip lines",
    "actions":                     [
      "Check drip irrigation valves for field_A1",
      "Manually irrigate 300L immediately",
      "Inspect pump pressure gauge"
    ],
    "prevention":                  "Install soil moisture alerts at 40% threshold (above current 30%)",
    "estimated_resolution_mins":   20
  },
  "actions":       ["Check drip irrigation valves...", ...],
  "raw_anomalies": [...]
}
```

---

### GET `/api/aiops/trends/{field_id}?hours=24`
Predictive trend analysis with irrigation/heat stress predictions.

**Response:**
```json
{
  "field_id":              "field_A1",
  "period_hours":          24,
  "temperature_trend":     "increasing",
  "moisture_trend":        "decreasing",
  "current_temp":          31.2,
  "current_moisture":      38.4,
  "predictions": [
    {
      "type":                    "irrigation_needed",
      "estimated_time_hours":    4.5,
      "confidence":              "high"
    }
  ],
  "data_points_analyzed":  96
}
```

---

### GET `/api/aiops/status`
System CPU/RAM health snapshot.

**Response:**
```json
{
  "status":     "healthy",
  "cpu_pct":    23.4,
  "memory_pct": 51.2,
  "memory_mb":  512,
  "timestamp":  "2025-01-15T10:30:00"
}
```

---

## Alerts

### GET `/api/alerts?field_id=field_A1&resolved=false&severity=critical`
List alerts with optional filters.

### PATCH `/api/alerts/{alert_id}/resolve`
Mark an alert as resolved.

---

## Remediation

### POST `/api/remediation/execute`
Manually trigger auto-remediation for a specific alert.

**Request body:**
```json
{ "alert_id": 5, "action_type": "trigger_irrigation", "manual_override": false }
```

### GET `/api/remediation/history?field_id=field_A1&hours=24`
Get history of auto-remediation actions.

**Response:**
```json
{
  "total_remediations": 8,
  "total_cost_inr":     342.50,
  "action_breakdown":   {"trigger_irrigation": 5, "apply_fertilizer": 2, "send_farmer_alert": 1},
  "recent_actions":     [...]
}
```

---

## Dashboard

### GET `/api/dashboard/stats`
Summary statistics for the dashboard overview.

---

## System

### GET `/health`
Basic health check.

```json
{
  "status":    "healthy",
  "service":   "AgriFarm Intelligence Platform v2.0",
  "timestamp": "2025-01-15T10:30:00"
}
```
