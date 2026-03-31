# 🌾 AgriFarm Intelligence Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple?style=flat-square)
![OpenRouter](https://img.shields.io/badge/OpenRouter-Free_Tier-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

<<<<<<< HEAD
**Production-grade Agentic AI + AIOps platform for precision agriculture**
=======
** Agentic AI + AIOps platform for precision agriculture**
>>>>>>> ab87c161e923e5ef0e79b4006a985329c80268b7

*Multi-agent tool-calling · ResNet18 disease vision · Prophet forecasting · ChromaDB RAG · LLM-powered AIOps*

</div>

---

<<<<<<< HEAD
## 🏆 Resume Headline

> *"Built a production-grade multi-agent AI platform for precision agriculture combining LLM tool-calling agents, computer vision disease detection (ResNet18 / PlantVillage), time-series yield forecasting (Prophet), RAG-powered market advisory (ChromaDB), and an AIOps monitoring layer with LLM root-cause analysis — deployed on AWS EC2 free tier at zero monthly cost."*

---

## ✨ What Makes This Stand Out
=======

---

>>>>>>> ab87c161e923e5ef0e79b4006a985329c80268b7

| Feature | Technology | Resume Signal |
|---|---|---|
| **Real agentic tool-calling loop** | LLM iteratively calls 7 tools (≤6 rounds) | "I know how actual agents work, not just chatbots" |
| **Computer vision pipeline** | ResNet18 fine-tuned on PlantVillage (38 classes) | ML model, not just prompting |
| **RAG pipeline** | ChromaDB + sentence-transformers + semantic search | Vector databases, embeddings |
| **Time-series forecasting** | Facebook Prophet + linear regression fallback | ML beyond LLMs |
| **AIOps with LLM analysis** | Rule engine + LLM root-cause + CloudWatch | SRE / MLOps skills |
| **Auto-remediation engine** | Rules → actions → cost tracking | Production thinking |
| **$0/month infrastructure** | AWS free tier + OpenRouter :free models | Cost-aware engineering |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Streamlit Dashboard (ui/)                        │
│    Overview · AI Chat · Disease Scanner · Yield · Market · AIOps    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────────────────┐
│                    FastAPI Backend (app/main.py)                      │
│   /api/agent/*  /api/disease/*  /api/yield/*  /api/market/*         │
│   /api/sensors/*  /api/aiops/*  /api/alerts/*  /api/remediation/*   │
└──────┬──────────────┬──────────────┬──────────────┬─────────────────┘
       │              │              │              │
┌──────▼──────┐ ┌─────▼──────┐ ┌────▼─────┐ ┌─────▼──────┐
│ Orchestrator│ │  Disease   │ │  Yield & │ │   AIOps    │
│   Agent     │ │  Agent     │ │  Market  │ │  Engine    │
│             │ │            │ │  Agent   │ │            │
│ ┌─────────┐ │ │ ResNet18   │ │ Prophet  │ │ Anomaly    │
│ │Diagnos- │ │ │ (38-class) │ │ ChromaDB │ │ Detector   │
│ │tic Agent│ │ │ + LLM      │ │ RAG      │ │ + LLM RCA  │
│ │(7 tools)│ │ │ treatment  │ │ + LLM    │ │            │
│ └────┬────┘ │ └────────────┘ │ advisory │ │ Auto-      │
│      │      │                └──────────┘ │ Remediation│
│ ┌────▼────┐ │                             └────────────┘
│ │ Action  │ │
│ │ Agent   │ │
│ │(5 tools)│ │
│ └─────────┘ │
└─────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────────┐
│                     Data Layer                                        │
│  SQLite (sensor_data · alerts · remediation_logs · agent_logs)       │
│  ChromaDB (agri knowledge base · disease treatments · irrigation)    │
│  CSV files (market_prices · yield_history · agri_knowledge)          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
agrifarm/
├── app/
│   ├── agents/
│   │   ├── diagnostic_agent.py    # LLM tool-calling loop (7 tools, ≤6 rounds)
│   │   ├── action_agent.py        # LLM picks remediation actions (5 tools)
│   │   ├── orchestrator.py        # Intent routing + 2-phase pipeline
│   │   ├── disease_agent.py       # ResNet18 vision + LLM treatment
│   │   └── yield_market_agent.py  # Prophet forecast + RAG market advisory
│   ├── aiops/
│   │   ├── anomaly_detector.py    # Rule engine + LLM root-cause analysis
│   │   └── auto_remediation.py    # Rules → actions → cost tracking
│   ├── tools/
│   │   └── sensor_tools.py        # 7 callable tools (sensor, soil, pest, RAG)
│   ├── database.py                # SQLAlchemy models (6 tables)
│   ├── models.py                  # Pydantic schemas
│   ├── llm_client.py              # OpenRouter client + free-model fallback
│   └── main.py                    # FastAPI (20+ endpoints)
├── rag/
│   └── retriever.py               # ChromaDB ingest + semantic search
├── data/
│   └── seed_data.py               # All synthetic data generator
├── ui/
│   └── dashboard.py               # Streamlit multi-page dashboard (7 pages)
├── tests/
│   └── test_platform.py           # Pytest test suite
├── deploy/
│   ├── ec2_setup.sh               # One-command AWS EC2 deploy
│   └── github_actions.yml         # CI/CD pipeline
├── docs/
│   ├── API.md                     # Full API reference
│   └── ARCHITECTURE.md            # Deep-dive architecture doc
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.11+
- [OpenRouter API key](https://openrouter.ai) (free, no credit card required for :free models)

### 1 — Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/agrifarm.git
cd agrifarm
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Configure environment

```bash
cp .env.example .env
# Open .env and add your OpenRouter API key:
# OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### 3 — Generate seed data + init RAG

```bash
python data/seed_data.py        # ~30 seconds
python rag/retriever.py         # downloads all-MiniLM-L6-v2 (~22 MB, once)
```

### 4 — Start the API

```bash
python app/main.py
# API running at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 5 — Launch the dashboard

```bash
# New terminal
streamlit run ui/dashboard.py
# Dashboard at http://localhost:8501
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🔑 Free API Keys Setup

### OpenRouter (required)
1. Go to [openrouter.ai](https://openrouter.ai)
2. Sign up (no credit card needed)
3. Dashboard → API Keys → Create Key
4. Add to `.env`: `OPENROUTER_API_KEY=sk-or-v1-...`

Free models used (auto-fallback order):
```
nvidia/nemotron-3-super-120b-a12b:free  →  meta-llama/llama-3.1-405b-instruct:free  →  mistralai/mistral-7b-instruct:free
```

### AWS Free Tier (optional — for CloudWatch AIOps)
1. Create account at [aws.amazon.com](https://aws.amazon.com) (free)
2. IAM → Users → Create user → Programmatic access
3. Add to `.env`: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
4. Set `CLOUDWATCH_ENABLED=true`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/agent/query` | Main agentic chat — auto-routes to specialist |
| `GET`  | `/api/agent/recommendations/{field_id}` | Full field health report |
| `POST` | `/api/disease/detect` | Upload leaf image → disease + treatment |
| `GET`  | `/api/yield/forecast` | Prophet yield forecast + LLM narrative |
| `GET`  | `/api/market/advisory` | RAG + LLM price advisory |
| `POST` | `/api/sensors/data` | Submit IoT sensor reading |
| `GET`  | `/api/sensors/data/{field_id}` | Historical sensor data |
| `GET`  | `/api/sensors/latest/{field_id}` | Latest sensor reading |
| `POST` | `/api/aiops/monitor` | Trigger rule-based anomaly detection |
| `POST` | `/api/aiops/llm-analyse` | LLM root-cause analysis of anomalies |
| `GET`  | `/api/aiops/trends/{field_id}` | Predictive trend analysis |
| `GET`  | `/api/aiops/status` | System CPU/RAM health |
| `GET`  | `/api/alerts` | List alerts (filterable) |
| `PATCH`| `/api/alerts/{id}/resolve` | Resolve an alert |
| `POST` | `/api/remediation/execute` | Manual remediation trigger |
| `GET`  | `/api/remediation/history` | Auto-remediation log |
| `GET`  | `/api/dashboard/stats` | Dashboard summary statistics |
| `GET`  | `/health` | Health check |

Full documentation: [docs/API.md](docs/API.md)

---

<<<<<<< HEAD
## 🎤 Interview Talking Points

**"Walk me through your agentic AI implementation"**
> The Diagnostic Agent uses a `while iteration < 6` loop where the LLM decides which of 7 tools to call next — sensor data, soil analysis, pest detection, irrigation check, field history, alerts, and a RAG knowledge base search. It's a genuine ReAct-style loop, not a canned pipeline. The Action Agent does the same for remediation: irrigation triggers, fertilizer scheduling, pH adjustment, cooling systems, and farmer SMS alerts.

**"How does your RAG pipeline work?"**
> Documents (disease treatments, irrigation guides, market strategies) are chunked and embedded using `all-MiniLM-L6-v2` locally — zero API cost. They're stored in ChromaDB with cosine similarity. At query time, the market agent and the diagnostic agent's knowledge-base tool both query it semantically. The retrieved context is injected into the LLM prompt.

**"What's the AIOps component?"**
> Two layers: a rule engine that checks sensor streams against configurable thresholds every time data arrives (your original), plus an LLM layer that periodically receives a summary of current anomalies and system metrics and returns structured JSON with severity, root cause, and remediation steps. The auto-remediation engine maps alert types to actions using a rules table and executes them automatically.

**"How did you handle the ML model?"**
> ResNet18 pretrained on ImageNet, with the final FC layer replaced for 38-class PlantVillage output. The model loads locally — no inference API, zero cost. For production I'd fine-tune it on the actual PlantVillage dataset (87K images, available free on HuggingFace), which takes ~30 min on a T4 GPU.

**"Why not LangGraph here?"**
> The original project's manual tool-calling loop is actually more transparent and debuggable for an interview demo. LangGraph adds value at scale (parallel branches, checkpointing, streaming); for a 2-agent system a clean while-loop is clearer. I'd add LangGraph if I needed parallel agent execution or persistent checkpointing across sessions.

---

## 💰 Cost Breakdown

| Component | Service | Monthly Cost |
|---|---|---|
| API server | AWS EC2 t2.micro | **$0** (750h free) |
| Storage | AWS S3 5GB | **$0** (free tier) |
| Monitoring | AWS CloudWatch 10 metrics | **$0** (free tier) |
| LLM calls | OpenRouter :free models | **$0** |
| Vector DB | ChromaDB (local) | **$0** |
| Dashboard | Streamlit Cloud | **$0** |
| Weather API | Open-Meteo | **$0** |
| **Total** | | **$0 / month** |

---
=======
>>>>>>> ab87c161e923e5ef0e79b4006a985329c80268b7

## 🗺️ Roadmap / Extensions

- [ ] Fine-tune ResNet18 on full PlantVillage (87K images) — 30 min on free Colab GPU
- [ ] Add real ESP32 IoT sensor integration via MQTT
- [ ] Twilio/Instaalerts WhatsApp notifications for critical alerts
- [ ] Docker Compose for one-command local setup
- [ ] Swap SQLite → PostgreSQL RDS for production scale
- [ ] LangGraph for parallel agent execution

---

<<<<<<< HEAD
## 📝 License

MIT — free to use for portfolios, interviews, and production projects.
=======

>>>>>>> ab87c161e923e5ef0e79b4006a985329c80268b7
