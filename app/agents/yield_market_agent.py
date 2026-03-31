"""
AgriFarm — Yield & Market Agent  (from AgriGenie)
Prophet time-series forecasting for yield prediction.
ChromaDB RAG + LLM for market price advisory.
"""

from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

from app.llm_client import quick_ask, detailed_ask

YIELD_CSV  = Path("./data/yield_history.csv")
MARKET_CSV = Path("./data/market_prices.csv")

CROPS = ["Tomato", "Potato", "Corn", "Wheat", "Rice", "Soybean", "Cotton"]


# ──────────────────────────────────────────────────────────────
# YIELD FORECASTING
# ──────────────────────────────────────────────────────────────
def forecast_yield(crop: str, state: str = "Tamil Nadu", forecast_years: int = 3) -> dict:
    """Facebook Prophet time-series forecast; falls back to linear regression."""
    if YIELD_CSV.exists():
        df = pd.read_csv(YIELD_CSV)
        sub = df[(df["crop"] == crop) & (df["state"] == state)].sort_values("year")
    else:
        sub = pd.DataFrame()

    if sub.empty:
        base = {"Tomato":25,"Potato":20,"Corn":6.5,"Wheat":4.2,
                "Rice":5.8,"Soybean":2.1,"Cotton":1.8}.get(crop, 5.0)
        years  = list(range(2015, 2025))
        yields = [round(base + i*0.05 + np.random.normal(0, 0.2), 2) for i in range(10)]
        sub    = pd.DataFrame({"year": years, "yield_tonnes_ha": yields})

    historical = sub[["year","yield_tonnes_ha"]].to_dict("records")
    years_list = [r["year"]            for r in historical]
    vals       = [r["yield_tonnes_ha"] for r in historical]

    try:
        from prophet import Prophet   # type: ignore
        pf  = pd.DataFrame({"ds": pd.to_datetime([str(y) for y in years_list]), "y": vals})
        m   = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                      daily_seasonality=False, interval_width=0.80)
        m.fit(pf)
        fut = m.make_future_dataframe(periods=forecast_years, freq="YE")
        fc  = m.predict(fut).tail(forecast_years)[["ds","yhat","yhat_lower","yhat_upper"]]
        forecasts = [
            {"year": int(r.ds.year), "predicted": round(r.yhat,2),
             "lower": round(r.yhat_lower,2), "upper": round(r.yhat_upper,2)}
            for _, r in fc.iterrows()
        ]
        method = "Prophet"
    except Exception:
        logger.warning("Prophet unavailable — linear regression fallback")
        coeffs = np.polyfit(years_list, vals, 1)
        poly   = np.poly1d(coeffs)
        last   = max(years_list)
        forecasts = [
            {"year": last+i+1, "predicted": round(float(poly(last+i+1)),2),
             "lower": round(float(poly(last+i+1))*0.88,2),
             "upper": round(float(poly(last+i+1))*1.12,2)}
            for i in range(forecast_years)
        ]
        method = "Linear Regression"

    return {
        "crop":          crop,
        "state":         state,
        "historical":    historical,
        "forecasts":     forecasts,
        "avg_yield":     round(np.mean(vals), 2),
        "trend_pct":     round((vals[-1]-vals[0])/vals[0]*100, 1) if vals[0] else 0,
        "best_year":     years_list[int(np.argmax(vals))],
        "next_year_pred": forecasts[0]["predicted"],
        "method":        method,
    }


def get_yield_narrative(data: dict) -> str:
    preds = data["forecasts"]
    pred_str = "\n".join(
        f"  {p['year']}: {p['predicted']} t/ha (range {p['lower']}–{p['upper']})"
        for p in preds
    )
    prompt = (
        f"Crop Yield Analysis — {data['crop']} in {data['state']}:\n"
        f"Historical avg: {data['avg_yield']} t/ha | Trend: {data['trend_pct']:+.1f}%\n"
        f"Forecast:\n{pred_str}\n\n"
        "Provide: 1) Plain-language outlook 2) Top 3 risks 3) Yield improvement tips "
        "4) Rough INR/hectare profitability 5) State average comparison. Under 250 words."
    )
    return detailed_ask(
        prompt,
        system="You are an expert agricultural economist specialising in Indian crop production.",
    )


def run_yield(crop: str, state: str = "Tamil Nadu") -> dict:
    data      = forecast_yield(crop, state)
    narrative = get_yield_narrative(data)
    return {**data, "narrative": narrative, "agent": "yield_agent"}


# ──────────────────────────────────────────────────────────────
# MARKET ADVISORY
# ──────────────────────────────────────────────────────────────
def get_price_stats(crop: str, days: int = 30) -> dict:
    if not MARKET_CSV.exists():
        base = {"Tomato":45,"Potato":18,"Corn":22,"Wheat":28,
                "Rice":35,"Soybean":52,"Cotton":68}.get(crop, 30)
        return {
            "crop": crop, "current_price_inr": base,
            "avg_30d": base, "avg_1y": base*0.95,
            "min_30d": base*0.85, "max_30d": base*1.15,
            "trend_7d": 1.2, "monthly": [],
            "price_signal": "HOLD",
        }
    df = pd.read_csv(MARKET_CSV)
    df["date"] = pd.to_datetime(df["date"])
    sub = df[df["crop"] == crop].sort_values("date")
    if sub.empty:
        return get_price_stats.__wrapped__(crop) if hasattr(get_price_stats, "__wrapped__") else {}  # fallback

    recent   = sub.tail(days)
    current  = round(float(recent["price_per_quintal_inr"].iloc[-1]), 2)
    avg_30d  = round(float(recent["price_per_quintal_inr"].mean()), 2)
    avg_1y   = round(float(sub["price_per_quintal_inr"].mean()), 2)
    trend_7d = round(
        float(recent.tail(7)["price_per_quintal_inr"].mean()) -
        float(recent.head(7)["price_per_quintal_inr"].mean()), 2
    )
    monthly = (
        sub.set_index("date")
        .resample("ME")["price_per_quintal_inr"].mean().round(2)
        .reset_index().rename(columns={"date":"month","price_per_quintal_inr":"avg_price"})
        .tail(12)
    )
    monthly["month"] = monthly["month"].dt.strftime("%Y-%m")

    return {
        "crop": crop, "current_price_inr": current,
        "avg_30d": avg_30d, "avg_1y": avg_1y,
        "min_30d": round(float(recent["price_per_quintal_inr"].min()), 2),
        "max_30d": round(float(recent["price_per_quintal_inr"].max()), 2),
        "trend_7d": trend_7d,
        "monthly": monthly.to_dict("records"),
        "price_signal": "BUY" if current < avg_1y*0.92 else "SELL" if current > avg_1y*1.10 else "HOLD",
    }


def run_market(crop: str, quantity_quintals: float = 10) -> dict:
    stats = get_price_stats(crop)

    # RAG retrieval
    rag_ctx = ""
    try:
        from rag.retriever import query_knowledge_base
        docs = query_knowledge_base(f"best time sell {crop} market price India", n_results=2)
        if docs:
            rag_ctx = "\nKnowledge base:\n" + "\n---\n".join(d["content"][:350] for d in docs)
    except Exception:
        pass

    revenue = round(stats["current_price_inr"] * quantity_quintals, 0)
    prompt  = (
        f"Market analysis — {crop}:\n"
        f"Current ₹{stats['current_price_inr']}/qtl | 30d avg ₹{stats['avg_30d']} | "
        f"1y avg ₹{stats['avg_1y']} | 7d trend {stats['trend_7d']:+.1f} | "
        f"Signal: {stats['price_signal']}\n"
        f"Quantity: {quantity_quintals} qtl | Est. revenue: ₹{revenue:,.0f}"
        f"{rag_ctx}\n\n"
        "Provide: 1) SELL/HOLD/WAIT decision 2) Target price (if holding) "
        "3) 30-day risk 4) Storage viability 5) Direct-market options 6) One quick win. "
        "Max 250 words."
    )
    advisory = detailed_ask(
        prompt,
        system=(
            "You are an expert agricultural commodity trader and farmer advisor with deep knowledge "
            "of APMC mandis, eNAM, FPOs, and agri-commodity trading across India."
        ),
    )
    return {
        "crop": crop, "stats": stats, "advisory": advisory,
        "rag_used": bool(rag_ctx), "agent": "market_agent",
    }
