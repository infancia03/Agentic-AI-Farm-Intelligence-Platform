"""
AgriFarm — Seed Data Generator
Run ONCE before starting the app:  python data/seed_data.py

Generates:
  • SQLite sensor data (YOUR original 5-field simulation)
  • market_prices.csv
  • yield_history.csv
  • agri_knowledge.json  (for RAG ChromaDB)
"""

from __future__ import annotations
import os, sys, json, random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

random.seed(42)
np.random.seed(42)

DATA_DIR = Path(__file__).parent
os.makedirs(DATA_DIR, exist_ok=True)

CROPS  = ["Tomato", "Potato", "Corn", "Wheat", "Rice", "Soybean", "Cotton"]
STATES = ["Tamil Nadu","Maharashtra","Karnataka","Andhra Pradesh",
          "Uttar Pradesh","Punjab","Haryana","Gujarat","Rajasthan"]
MARKETS = ["Chennai APMC","Koyambedu","Madurai","Coimbatore APMC",
           "Delhi Azadpur","Mumbai APMC","Bengaluru APMC","Hyderabad"]

TREATMENTS = {
    "Early Blight":        "Apply copper-based fungicide every 7–10 days. Remove infected lower leaves. Improve air circulation.",
    "Late Blight":         "Apply mancozeb or chlorothalonil IMMEDIATELY. Destroy severely infected plants. Switch to drip irrigation.",
    "Leaf Mold":           "Reduce humidity below 85%. Apply chlorothalonil. Ensure 24-hour dry periods between irrigations.",
    "Septoria Leaf Spot":  "Apply mancozeb or chlorothalonil. Avoid working in wet fields. Rotate crops next season.",
    "Gray Leaf Spot":      "Apply strobilurin or triazole fungicide. Rotate crops annually. Use resistant hybrids.",
    "Common Rust":         "Apply triazole fungicide early. Plant resistant varieties next season. Scout bi-weekly.",
    "Northern Leaf Blight":"Apply propiconazole or azoxystrobin. Improve drainage. Till under crop residue.",
    "Yellow Rust":         "Apply propiconazole IMMEDIATELY (spreads fast in cool weather). Scout fields bi-weekly.",
    "Brown Rust":          "Apply tebuconazole. Monitor flag leaf closely. Maintain balanced N fertilisation.",
    "Blast":               "Apply tricyclazole or azoxystrobin. Avoid excess nitrogen. Drain fields periodically.",
    "Bacterial Blight":    "No effective chemical. Remove infected plants. Use resistant varieties next season.",
    "Brown Spot":          "Apply mancozeb or iprodione. Ensure balanced potassium. Avoid water stress.",
    "Frogeye Leaf Spot":   "Apply azoxystrobin or pyraclostrobin. Rotate crops with non-host plants.",
    "Sudden Death Syndrome":"No curative treatment. Improve drainage. Apply seed treatments next season.",
    "Verticillium Wilt":   "No chemical cure. Remove infected plants. Solarise soil. Use resistant varieties.",
    "Healthy":             "No treatment needed. Scout every 5–7 days. Maintain balanced fertilisation.",
}


# ── 1. SQLite sensor data (YOUR original logic) ───────────────
def seed_sensor_data():
    from app.database import init_db, SessionLocal, SensorData
    init_db()
    db = SessionLocal()

    fields = ["field_A1","field_A2","field_B1","field_B2","field_C1"]
    interval = 15       # minutes
    hours_back = 48
    n_readings = (hours_back * 60) // interval
    print("🌱 Seeding sensor data …")

    for fid in fields:
        base_temp  = random.uniform(25, 30)
        base_moist = random.uniform(50, 70)
        base_ph    = random.uniform(6.5, 7.2)
        base_n     = random.uniform(60, 120)
        base_p     = random.uniform(25, 50)
        base_k     = random.uniform(60, 100)
        base_hum   = random.uniform(60, 80)

        for i in range(n_readings):
            ts = datetime.utcnow() - timedelta(minutes=interval * (n_readings - i - 1))
            h  = ts.hour

            temp     = base_temp  + 5 * np.sin((h - 6) * np.pi / 12) + random.uniform(-1, 1)
            moisture = max(20, base_moist - (i / n_readings) * 20 + random.uniform(-3, 3))
            ph       = base_ph + random.uniform(-0.2, 0.2)
            nitrogen = max(30, base_n - (i / n_readings) * 15 + random.uniform(-5, 5))
            phosphorus = max(15, base_p - (i / n_readings) * 6 + random.uniform(-2, 2))
            potassium  = max(40, base_k - (i / n_readings) * 8 + random.uniform(-3, 3))
            humidity   = base_hum - 5 * np.sin((h - 6) * np.pi / 12) + random.uniform(-2, 2)

            # Inject anomalies in last 6 hours for testing
            if i > n_readings - 24:
                if fid == "field_A1":  moisture = min(moisture, 25)
                if fid == "field_B1":  temp     = max(temp, 38)
                if fid == "field_C1":  nitrogen = min(nitrogen, 25)

            db.add(SensorData(
                field_id=fid, timestamp=ts,
                temperature=round(temp, 2),
                soil_moisture=round(float(np.clip(moisture, 0, 100)), 2),
                ph_level=round(float(np.clip(ph, 0, 14)), 2),
                nitrogen=round(max(0, nitrogen), 2),
                phosphorus=round(max(0, phosphorus), 2),
                potassium=round(max(0, potassium), 2),
                humidity=round(float(np.clip(humidity, 0, 100)), 2),
                rainfall_mm=round(max(0, random.gauss(1, 2)), 1),
            ))
        db.commit()
        print(f"  ✓ {fid}: {n_readings} readings")
    db.close()


# ── 2. Market prices CSV ──────────────────────────────────────
def seed_market_prices():
    print("📊 Generating market_prices.csv …")
    base = {"Tomato":45,"Potato":18,"Corn":22,"Wheat":28,"Rice":35,"Soybean":52,"Cotton":68}
    rows = []
    base_date = datetime(2023, 1, 1)
    for i in range(365):
        d = base_date + timedelta(days=i)
        for crop in CROPS:
            seasonal = np.sin(2 * np.pi * (i + 60) / 365) * 6
            price    = round(max(8, base[crop] + seasonal + np.random.normal(0, 1.5)), 2)
            rows.append({
                "date":                  d.strftime("%Y-%m-%d"),
                "crop":                  crop,
                "price_per_quintal_inr": price,
                "market":                random.choice(MARKETS),
                "state":                 random.choice(STATES),
                "arrivals_tonnes":       round(random.uniform(50, 5000), 1),
            })
    pd.DataFrame(rows).to_csv(DATA_DIR / "market_prices.csv", index=False)
    print(f"  ✓ {len(rows)} market price records")


# ── 3. Yield history CSV ──────────────────────────────────────
def seed_yield_history():
    print("🌾 Generating yield_history.csv …")
    base_yields = {"Tomato":25,"Potato":20,"Corn":6.5,"Wheat":4.2,"Rice":5.8,"Soybean":2.1,"Cotton":1.8}
    rows = []
    for year in range(2015, 2025):
        for state in STATES:
            for crop in CROPS:
                trend = (year - 2015) * 0.05 * base_yields[crop]
                rows.append({
                    "year":            year,
                    "state":           state,
                    "crop":            crop,
                    "yield_tonnes_ha": round(max(0.5, base_yields[crop] + trend + np.random.normal(0, base_yields[crop]*0.08)), 2),
                    "area_ha":         round(random.uniform(1000, 50000), 0),
                    "rainfall_mm":     round(random.uniform(400, 1800), 0),
                    "avg_temp_c":      round(random.uniform(22, 35), 1),
                })
    pd.DataFrame(rows).to_csv(DATA_DIR / "yield_history.csv", index=False)
    print(f"  ✓ {len(rows)} yield history records")


# ── 4. Agri knowledge base (for RAG) ─────────────────────────
def seed_knowledge_base():
    print("📚 Generating agri_knowledge.json …")
    docs = []

    # Disease treatments
    for disease, treatment in TREATMENTS.items():
        docs.append({
            "id":      f"treatment_{disease.lower().replace(' ','_')}",
            "type":    "disease_treatment",
            "title":   f"Treatment Guide: {disease}",
            "content": (
                f"Disease: {disease}\n"
                f"Treatment Protocol: {treatment}\n"
                f"Severity: {'High' if any(w in treatment for w in ['IMMEDIATELY','URGENT']) else 'Medium'}\n"
                f"Prevention: Rotate crops, use certified seeds, maintain field hygiene, scout regularly.\n"
                f"Economic Impact: Untreated {disease} can cause 20–80% yield loss.\n"
                f"Organic Option: Neem oil (5ml/L) + Trichoderma bio-agent as preventive spray.\n"
            ),
            "crop": "General",
        })

    # Irrigation guides
    irrigation = {
        "Tomato":  ("every 2–3 days","25–30mm/week","flowering and fruit set"),
        "Potato":  ("every 3–4 days","20–25mm/week","tuber initiation"),
        "Rice":    ("continuous flood","1200–2000mm/season","tillering and heading"),
        "Wheat":   ("every 7–10 days","450–650mm/season","jointing and grain fill"),
        "Corn":    ("every 4–5 days","500–800mm/season","silking and grain fill"),
        "Soybean": ("every 5–7 days","450–700mm/season","pod fill"),
        "Cotton":  ("every 7–10 days","700–1300mm/season","boll development"),
    }
    for crop, (freq, amount, critical) in irrigation.items():
        docs.append({
            "id":   f"irrigation_{crop.lower()}",
            "type": "irrigation_guide",
            "title": f"Irrigation Guide: {crop}",
            "content": (
                f"{crop} Irrigation Management:\n"
                f"Frequency: {freq}\nWater requirement: {amount}\n"
                f"Critical growth stage: {critical}\n"
                f"Deficit irrigation warning: Soil moisture below 40% field capacity triggers stress.\n"
                f"Excess water warning: Waterlogging >48h causes root hypoxia and disease susceptibility.\n"
                f"Drip irrigation saves 30–40% water and improves yield 15–25% vs flood irrigation.\n"
            ),
            "crop": crop,
        })

    # Market advisory
    docs += [
        {
            "id": "market_timing_tomato",
            "type": "market_advisory",
            "title": "Tomato Market Timing Guide",
            "content": (
                "Tomato prices peak December–February (winter demand) and May–June (summer scarcity).\n"
                "Avoid selling October–November when harvest glut drops prices 40–60%.\n"
                "Cold storage extends shelf life 3–4 weeks; use to shift selling window.\n"
                "Direct farm-to-hotel channel bypasses APMC 8% commission.\n"
                "MSP reference: ₹600–800/quintal for processing grade.\n"
            ),
            "crop": "Tomato",
        },
        {
            "id": "market_general_strategy",
            "type": "market_advisory",
            "title": "General Crop Market Strategy",
            "content": (
                "Forward contracts with food processors lock in price 3–6 months ahead.\n"
                "Farmer Producer Organisations (FPOs) aggregate volume for better mandi bargaining.\n"
                "eNAM (National Agriculture Market) gives pan-India price discovery — register free.\n"
                "Value addition: Tomato→puree 3–5x, Potato→chips 8–10x, Cotton→handloom 2–3x.\n"
                "Export windows: Basmati rice Aug–Nov. Soybean exports March–June.\n"
            ),
            "crop": "General",
        },
    ]

    with open(DATA_DIR / "agri_knowledge.json", "w") as f:
        json.dump(docs, f, indent=2)
    print(f"  ✓ {len(docs)} knowledge base documents")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🌱 AgriFarm — Seed Data Generator")
    print("=" * 42)
    seed_sensor_data()
    seed_market_prices()
    seed_yield_history()
    seed_knowledge_base()
    print("\n✅ All seed data ready. Next: python rag/retriever.py\n")
