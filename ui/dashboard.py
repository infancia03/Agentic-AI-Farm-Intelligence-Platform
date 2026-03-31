"""
AgriFarm Intelligence Platform — Streamlit Dashboard
Multi-page UI replacing the original Gradio dashboard.
Run: streamlit run ui/dashboard.py
"""

from __future__ import annotations
import sys, os, base64, json
from pathlib import Path

import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AgriFarm Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0d2b1a; }
[data-testid="stSidebar"] * { color: #c8e6c9 !important; }
.metric-card { background:#f0fdf4; border:1px solid #86efac; border-radius:10px; padding:1rem; text-align:center; }
.alert-critical { background:#fef2f2; border-left:4px solid #ef4444; padding:.75rem 1rem; border-radius:4px; margin:.5rem 0; }
.alert-high     { background:#fff7ed; border-left:4px solid #f97316; padding:.75rem 1rem; border-radius:4px; margin:.5rem 0; }
.alert-medium   { background:#fefce8; border-left:4px solid #eab308; padding:.75rem 1rem; border-radius:4px; margin:.5rem 0; }
.badge-healthy  { background:#dcfce7; color:#166534; padding:2px 10px; border-radius:20px; font-size:.8rem; font-weight:600; }
.badge-disease  { background:#fef2f2; color:#991b1b; padding:2px 10px; border-radius:20px; font-size:.8rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

CROPS  = ["Tomato","Potato","Corn","Wheat","Rice","Soybean","Cotton"]
STATES = ["Tamil Nadu","Maharashtra","Karnataka","Andhra Pradesh","Uttar Pradesh","Punjab","Haryana","Gujarat","Rajasthan"]
FIELDS = ["field_A1","field_A2","field_B1","field_B2","field_C1"]


# ── helpers ───────────────────────────────────────────────────
def _get(ep, params=None, timeout=15):
    try:
        r = requests.get(f"{API}{ep}", params=params, timeout=timeout)
        return r.json() if r.ok else None
    except Exception:
        return None

def _post(ep, data=None, files=None, timeout=60):
    try:
        if files:
            r = requests.post(f"{API}{ep}", data=data, files=files, timeout=timeout)
        else:
            r = requests.post(f"{API}{ep}", json=data, timeout=timeout)
        return r.json() if r.ok else None
    except Exception:
        return None


# ── sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 AgriFarm")
    st.markdown("*Intelligence Platform v2.0*")
    st.divider()

    page = st.radio("Navigate", [
        "🏠 Overview",
        "🤖 AI Agent Chat",
        "🔬 Disease Scanner",
        "📈 Yield Forecast",
        "📊 Market Advisory",
        "📡 Sensor Monitor",
        "⚙️ AIOps & Alerts",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("**Farm Profile**")
    farm_field = st.selectbox("Field", FIELDS, key="sidebar_field")
    farm_crop  = st.selectbox("Crop",  CROPS,  key="sidebar_crop")
    farm_state = st.selectbox("State", STATES, key="sidebar_state")

    st.divider()
    api_ok = _get("/health", timeout=3)
    if api_ok:
        st.success("API ✓ Online")
    else:
        st.error("API ✗ Offline")
    st.caption("OpenRouter :free models\nAWS Free Tier · $0/month")


# ════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🌾 AgriFarm Intelligence Platform")
    st.caption("Agentic AI · ResNet18 Vision · RAG · Prophet Forecasting · AIOps · Auto-Remediation")

    stats = _get("/api/dashboard/stats")
    if stats:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Fields Monitored", stats.get("total_fields",0))
        c2.metric("Active Alerts",    stats.get("active_alerts",0),
                  delta=f"{stats.get('critical_alerts',0)} critical",
                  delta_color="inverse")
        c3.metric("Sensor Readings/h", stats.get("recent_readings_1h",0))
        c4.metric("Auto-Actions (24h)", stats.get("remediation_summary",{}).get("total_24h",0))

    st.divider()
    st.subheader("What this platform does")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**🤖 Multi-Agent System (your original)**
- Diagnostic Agent: real LLM tool-calling loop (≤6 iterations)
- Action Agent: LLM picks irrigation/fertiliser/cooling actions
- Orchestrator: routes intent to right specialist

**🔬 Disease Detection (new)**
- ResNet18 vision model, 38 PlantVillage classes
- Upload leaf photo → instant diagnosis
- LLM treatment plan with Indian product names
""")
    with col2:
        st.markdown("""
**📈 Yield & Market Intelligence (new)**
- Facebook Prophet time-series yield forecasting
- ChromaDB RAG retrieval over agri knowledge base
- Sell / Hold / Wait market price advisory

**⚙️ AIOps (enhanced)**
- Rule-based anomaly detection (your original)
- LLM root-cause analysis layer added
- CloudWatch metrics + auto-remediation engine
""")


# ════════════════════════════════════════════════════════════
# PAGE: AI AGENT CHAT
# ════════════════════════════════════════════════════════════
elif page == "🤖 AI Agent Chat":
    st.header("🤖 Multi-Agent AI Chat")
    st.caption("Ask anything — the orchestrator routes to the right specialist agent automatically")

    if "history" not in st.session_state: st.session_state.history = []

    for m in st.session_state.history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("meta"):
                st.caption(m["meta"])

    suggestions = [
        "What is the soil health status of field_A1?",
        "field_B1 has high temperature — what should I do?",
        "Which fields need irrigation right now?",
        "Forecast tomato yield for Tamil Nadu next 3 years",
        "Should I sell my 20 quintals of rice now or wait?",
        "What fields have nitrogen deficiency?",
    ]
    st.markdown("**Quick questions:**")
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        if cols[i % 3].button(s, key=f"s{i}", use_container_width=True):
            st.session_state._inject = s
            st.rerun()

    if prompt := st.chat_input("Ask about your farm …") or st.session_state.pop("_inject", None):
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent reasoning …"):
                result = _post("/api/agent/query", {
                    "query": prompt,
                    "field_id": farm_field,
                    "farm_context": {"crop": farm_crop, "state": farm_state},
                })
            if result:
                answer = result.get("orchestrator_summary") or result.get("llm_advice") or result.get("narrative") or "No response"
                intent = result.get("intent","diagnostic")
                tools  = result.get("diagnostic_phase",{}).get("tools_used",[])
                meta   = f"Agent: {intent} | Tools used: {', '.join(tools) if tools else 'none'} | Time: {result.get('total_execution_time',0):.1f}s"
                st.markdown(answer)
                st.caption(meta)
                st.session_state.history.append({"role":"assistant","content":answer,"meta":meta})

                if result.get("action_phase",{}).get("actions_taken"):
                    with st.expander(f"⚡ {result['action_phase']['total_actions']} auto-remediation actions executed"):
                        for a in result["action_phase"]["actions_taken"]:
                            st.write(f"• **{a['action']}** — {a['result'].get('action','')} | ₹{a['result'].get('cost_inr',0)}")
            else:
                st.error("API error. Make sure FastAPI is running.")


# ════════════════════════════════════════════════════════════
# PAGE: DISEASE SCANNER
# ════════════════════════════════════════════════════════════
elif page == "🔬 Disease Scanner":
    st.header("🔬 Crop Disease Scanner")
    st.caption("Upload a leaf photo → ResNet18 (38-class PlantVillage) → LLM treatment plan")

    c1, c2 = st.columns([1,1])
    with c1:
        uploaded = st.file_uploader("Upload leaf image", type=["jpg","jpeg","png","webp"])
        crop_hint = st.selectbox("Crop (helps context)", ["Auto-detect"] + CROPS, key="disease_crop_hint")
        location  = st.text_input("Location", value=f"{farm_state}, India", key="disease_location")
        run_btn   = st.button("🔬 Analyse Disease", type="primary", disabled=not uploaded)

    with c2:
        if uploaded:
            st.image(uploaded, caption="Uploaded leaf", use_column_width=True)

    if run_btn and uploaded:
        with st.spinner("Running ResNet18 + LLM …"):
            files  = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            result = _post("/api/disease/detect",
                           data={"crop": crop_hint if crop_hint != "Auto-detect" else "Unknown",
                                 "field_id": farm_field, "location": location},
                           files=files)
        if result:
            vr   = result.get("vision_result", result)
            conf = vr.get("confidence", 0)

            # Warning banner — shown when model not fine-tuned
            if result.get("warning"):
                st.warning(result["warning"])

            c1,c2,c3 = st.columns(3)
            c1.metric("Detected Crop", vr.get("crop","—"))
            c2.metric("Disease",       vr.get("disease","—"))
            c3.metric("Confidence",    f"{conf:.1%}",
                      delta="Fine-tuned" if vr.get("is_finetuned") else "Not fine-tuned",
                      delta_color="normal" if vr.get("is_finetuned") else "inverse")

            if result.get("low_confidence") and not vr.get("is_finetuned"):
                st.info("ℹ️ Confidence too low for reliable crop/disease ID from image. "
                        "Treatment advice below is based on your selected crop type.")
            elif vr.get("is_healthy"):
                st.success("✅ Plant appears HEALTHY")
            else:
                st.error(f"⚠️ Disease detected: **{vr.get('disease')}** in **{vr.get('crop')}**")

            st.subheader("🩺 Treatment Advisory")
            st.markdown(result.get("llm_advice",""))

            with st.expander("Top 3 Vision Predictions (for reference)"):
                if not vr.get("is_finetuned"):
                    st.caption("These predictions are from a non-fine-tuned model and may be inaccurate.")
                for item in vr.get("top3",[]):
                    st.write(f"  {item['confidence']:.1%} — **{item['crop']}**: {item['disease']}")


# ════════════════════════════════════════════════════════════
# PAGE: YIELD FORECAST
# ════════════════════════════════════════════════════════════
elif page == "📈 Yield Forecast":
    st.header("📈 Crop Yield Forecast")
    st.caption("Facebook Prophet time-series model + LLM narrative analysis")

    c1,c2 = st.columns(2)
    with c1: crop  = st.selectbox("Crop",  CROPS, index=CROPS.index(farm_crop),  key="yield_crop")
    with c2: state = st.selectbox("State", STATES, index=STATES.index(farm_state), key="yield_state")

    if st.button("📈 Generate Forecast", type="primary"):
        with st.spinner("Running Prophet forecast + LLM …"):
            result = _get("/api/yield/forecast", {"crop": crop, "state": state})
        if result:
            hist = pd.DataFrame(result.get("historical",[]))
            fc   = pd.DataFrame(result.get("forecasts",[]))

            if not hist.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist["year"], y=hist["yield_tonnes_ha"],
                    name="Historical", mode="lines+markers", line=dict(color="#16a34a", width=2)))
                if not fc.empty:
                    fig.add_trace(go.Scatter(x=fc["year"], y=fc["predicted"],
                        name="Forecast", mode="lines+markers",
                        line=dict(color="#f59e0b", width=2, dash="dash")))
                    fig.add_trace(go.Scatter(
                        x=pd.concat([fc["year"], fc["year"][::-1]]),
                        y=pd.concat([fc["upper"], fc["lower"][::-1]]),
                        fill="toself", fillcolor="rgba(245,158,11,0.12)",
                        line=dict(color="rgba(0,0,0,0)"), name="80% CI"))
                fig.update_layout(title=f"{crop} Yield — {state}", xaxis_title="Year",
                                  yaxis_title="t/ha", height=380)
                st.plotly_chart(fig, use_container_width=True)

            c1,c2,c3 = st.columns(3)
            c1.metric("Avg Historical", f"{result.get('avg_yield',0)} t/ha")
            c2.metric("10-Year Trend",  f"{result.get('trend_pct',0):+.1f}%")
            c3.metric("Next Year",      f"{result.get('next_year_pred',0)} t/ha")

            st.subheader("AI Yield Narrative")
            st.markdown(result.get("narrative",""))
            st.caption(f"Method: {result.get('method','Prophet')}")


# ════════════════════════════════════════════════════════════
# PAGE: MARKET ADVISORY
# ════════════════════════════════════════════════════════════
elif page == "📊 Market Advisory":
    st.header("📊 Crop Market Price Advisory")
    st.caption("Price trend analysis + ChromaDB RAG + LLM sell/hold/wait recommendation")

    c1,c2 = st.columns(2)
    with c1: crop = st.selectbox("Crop", CROPS, index=CROPS.index(farm_crop), key="market_crop")
    with c2: qty  = st.number_input("Quantity (quintals)", value=10.0, min_value=1.0, step=5.0)

    if st.button("📊 Get Advisory", type="primary"):
        with st.spinner("Analysing prices + querying knowledge base …"):
            result = _get("/api/market/advisory", {"crop": crop, "quantity_quintals": qty})
        if result:
            stats  = result.get("stats",{})
            signal = stats.get("price_signal","HOLD")
            icons  = {"SELL":"🟢","HOLD":"🟡","BUY":"🔵"}
            revenue = stats.get("current_price_inr",0) * qty

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Current Price", f"₹{stats.get('current_price_inr',0)}/qtl")
            c2.metric("30-day Avg",    f"₹{stats.get('avg_30d',0)}/qtl")
            c3.metric("Signal",        f"{icons.get(signal,'')} {signal}")
            c4.metric("Est. Revenue",  f"₹{revenue:,.0f}")

            monthly = stats.get("monthly",[])
            if monthly:
                df_m = pd.DataFrame(monthly)
                fig  = px.area(df_m, x="month", y="avg_price",
                               title=f"{crop} Monthly Price (₹/quintal)",
                               color_discrete_sequence=["#16a34a"])
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("AI Market Advisory")
            if result.get("rag_used"):
                st.caption("📚 Enhanced with knowledge base retrieval (RAG)")
            st.markdown(result.get("advisory",""))


# ════════════════════════════════════════════════════════════
# PAGE: SENSOR MONITOR
# ════════════════════════════════════════════════════════════
elif page == "📡 Sensor Monitor":
    st.header("📡 Field Sensor Monitor")
    c1,c2 = st.columns(2)
    with c1: sel_field = st.selectbox("Field", FIELDS, index=FIELDS.index(farm_field), key="sensor_field")
    with c2: hours     = st.slider("History (hours)", 6, 48, 24)

    data = _get(f"/api/sensors/data/{sel_field}", {"hours": hours})
    if data:
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        if not df.empty:
            latest = df.iloc[-1]
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Temperature",   f"{latest.temperature}°C")
            c2.metric("Soil Moisture", f"{latest.soil_moisture}%")
            c3.metric("pH",            f"{latest.ph_level}")
            c4.metric("Nitrogen",      f"{latest.nitrogen} ppm")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df["temperature"],   name="Temp °C",    line=dict(color="#ef4444")))
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df["soil_moisture"], name="Moisture %", line=dict(color="#3b82f6")))
            fig.add_hline(y=35,  line_dash="dash", line_color="#ef4444", annotation_text="Temp threshold")
            fig.add_hline(y=30,  line_dash="dash", line_color="#3b82f6", annotation_text="Moisture threshold")
            fig.update_layout(title=f"{sel_field} — Temperature & Moisture", height=350)
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df["timestamp"], y=df["nitrogen"],   name="N ppm",  line=dict(color="#8b5cf6")))
            fig2.add_trace(go.Scatter(x=df["timestamp"], y=df["phosphorus"], name="P ppm",  line=dict(color="#f59e0b")))
            fig2.add_trace(go.Scatter(x=df["timestamp"], y=df["potassium"],  name="K ppm",  line=dict(color="#10b981")))
            fig2.update_layout(title=f"{sel_field} — NPK Nutrient Levels", height=300)
            st.plotly_chart(fig2, use_container_width=True)

        trend = _get(f"/api/aiops/trends/{sel_field}", {"hours": hours})
        if trend and "error" not in trend:
            st.subheader("Trend Analysis")
            c1,c2 = st.columns(2)
            c1.metric("Temperature Trend", trend["temperature_trend"].title())
            c2.metric("Moisture Trend",    trend["moisture_trend"].title())
            if trend.get("predictions"):
                for p in trend["predictions"]:
                    st.warning(f"⚠️ {p['type'].replace('_',' ').title()} predicted in ~{p['estimated_time_hours']}h (confidence: {p['confidence']})")


# ════════════════════════════════════════════════════════════
# PAGE: AIOPS & ALERTS
# ════════════════════════════════════════════════════════════
elif page == "⚙️ AIOps & Alerts":
    st.header("⚙️ AIOps & Alerts")
    st.caption("Rule-based detection (your original) + LLM root-cause analysis (new)")

    # System status
    status = _get("/api/aiops/status")
    if status:
        col = "green" if status["status"] == "healthy" else "orange"
        st.markdown(f"**System:** :{col}[{status['status'].upper()}]  |  "
                    f"CPU {status['cpu_pct']}%  |  RAM {status['memory_pct']}%")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Rule-Based Monitor")
        field_filter = st.selectbox("Filter field", ["All"] + FIELDS, key="aiops_field_filter")
        fid = None if field_filter == "All" else field_filter

        if st.button("🔍 Run AIOps Monitor", type="primary"):
            with st.spinner("Detecting anomalies …"):
                r = _post("/api/aiops/monitor", {"field_id": fid} if fid else {})
            if r:
                an = r.get("anomalies_detected", 0)
                al = r.get("alerts_created", 0)
                if an == 0:
                    st.success("✅ No anomalies detected")
                else:
                    st.warning(f"⚠️ {an} anomalies | {al} new alerts")
                    for a in r.get("anomalies", [])[:5]:
                        sev = a["severity"]
                        cls = {"critical":"alert-critical","high":"alert-high","medium":"alert-medium"}.get(sev,"")
                        st.markdown(f'<div class="{cls}"><b>[{sev.upper()}]</b> {a["field_id"]}: {a["message"]}</div>', unsafe_allow_html=True)
                if r.get("auto_remediation"):
                    ar = r["auto_remediation"]
                    st.info(f"⚡ Auto-remediation: {ar.get('remediations_executed',0)} actions | ₹{ar.get('total_cost_inr',0)}")

    with c2:
        st.subheader("LLM Root-Cause Analysis")
        if st.button("🧠 LLM Anomaly Analysis"):
            with st.spinner("LLM analysing anomalies …"):
                r = _post("/api/aiops/llm-analyse", {"field_id": fid} if fid else {})
            if r:
                if r.get("status") == "healthy":
                    st.success(r.get("analysis","System healthy"))
                else:
                    a = r.get("analysis", {})
                    if isinstance(a, dict):
                        sev = a.get("severity","?")
                        col = "red" if sev in ("critical","high") else "orange"
                        st.markdown(f"**Severity:** :{col}[{sev.upper()}]")
                        st.markdown(f"**Root Cause:** {a.get('root_cause','')}")
                        st.markdown("**Recommended Actions:**")
                        for act in a.get("actions",[]):
                            st.code(act)
                        st.caption(f"Est. resolution: {a.get('estimated_resolution_mins','?')} min")
                    else:
                        st.warning(str(a))

    st.divider()
    st.subheader("Active Alerts")
    alerts = _get("/api/alerts", {"resolved": False})
    if alerts:
        for a in alerts[:10]:
            sev = a.get("severity","medium")
            cls = {"critical":"alert-critical","high":"alert-high","medium":"alert-medium"}.get(sev,"")
            col1, col2 = st.columns([5,1])
            with col1:
                st.markdown(f'<div class="{cls}"><b>[{sev.upper()}] {a["field_id"]}</b> — {a["message"]}<br>'
                            f'<small>{a["timestamp"][:16]} | Remediated: {a["auto_remediation_applied"]}</small></div>',
                            unsafe_allow_html=True)
            with col2:
                if st.button("Resolve", key=f"res_{a['id']}"):
                    requests.patch(f"{API}/api/alerts/{a['id']}/resolve")
                    st.rerun()
    elif alerts is not None:
        st.success("✅ No active alerts")

    st.divider()
    st.subheader("Remediation History (24h)")
    hist = _get("/api/remediation/history")
    if hist and hist.get("total_remediations", 0) > 0:
        c1,c2 = st.columns(2)
        c1.metric("Total Actions", hist["total_remediations"])
        c2.metric("Total Cost", f"₹{hist['total_cost_inr']:.2f}")
        df_h = pd.DataFrame(hist.get("recent_actions",[]))
        if not df_h.empty:
            st.dataframe(df_h[["timestamp","field_id","action_type","success","cost_inr"]], use_container_width=True)
