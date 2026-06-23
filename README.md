# 🌾 AgriFarm Intelligence Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple?style=flat-square)
![OpenRouter](https://img.shields.io/badge/OpenRouter-Free_Tier-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

*Multi-agent tool-calling · ResNet18 disease vision · Prophet forecasting · ChromaDB RAG · LLM-powered AIOps*

</div>

---

## 🎯 Overview

AgriFarm Intelligence Platform is a comprehensive agricultural management system that combines multiple AI technologies to provide real-time field monitoring, disease detection, yield forecasting, and automated remediation. The platform operates at zero infrastructure cost using AWS free tier and open-source models.

| Feature | Technology |
|---|---|
| **Multi-agent agentic architecture** | LLM iterative tool-calling (7 tools, ≤6 rounds) |
| **Computer vision pipeline** | ResNet18 fine-tuned on PlantVillage (38 classes) |
| **Semantic search & RAG** | ChromaDB + sentence-transformers |
| **Time-series forecasting** | Facebook Prophet + linear regression fallback |
| **Intelligent anomaly detection** | Rule engine + LLM root-cause analysis |
| **Auto-remediation engine** | Rules-based action execution with cost tracking |
| **Zero-cost infrastructure** | AWS free tier + OpenRouter free models |

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
│ └────┬────┘ │ │ synthesis  │ │ advisory │ │ Auto-      │
│      │      │ └────────────┘ │ synthesis│ │ Remediation│
│ ┌────▼────┐ │                └──────────┘ │            │
│ │ Action  │ │                             └────────────┘
│ │ Agent   │ │
│ │(5 tools)│ │
│ └─────────┘ │
└─────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────────┐
│                     Data Layer                                        │
│  SQLite (sensor_data · alerts · remediation_logs · agent_logs)       │
│  ChromaDB (agri knowledge base · disease treatments · irrigation)    │
│  CSV files (market_prices · yield_history · agricultural_knowledge) │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
agrifarm/
├── app/
│   ├── agents/
│   │   ├── diagnostic_agent.py    # Tool-calling loop (7 tools, ≤6 rounds)
│   │   ├── action_agent.py        # Remediation action selection (5 tools)
│   │   ├── orchestrator.py        # Intent routing + 2-phase pipeline
│   │   ├── disease_agent.py       # ResNet18 vision + LLM synthesis
│   │   └── yield_market_agent.py  # Prophet forecast + RAG advisory
│   ├── aiops/
│   │   ├── anomaly_detector.py    # Rule engine + LLM analysis
│   │   └── auto_remediation.py    # Rules → actions → tracking
│   ├── tools/
│   │   └── sensor_tools.py        # 7 callable tools
│   ├── database.py                # SQLAlchemy models (6 tables)
│   ├── models.py                  # Pydantic schemas
│   ├── llm_client.py              # OpenRouter client + fallback
│   └── main.py                    # FastAPI (20+ endpoints)
├── rag/
│   └── retriever.py               # ChromaDB + semantic search
├── data/
│   └── seed_data.py               # Synthetic data generator
├── ui/
│   └── dashboard.py               # Streamlit multi-page dashboard
├── tests/
│   └── test_platform.py           # Pytest test suite
├── deploy/
│   ├── ec2_setup.sh               # AWS EC2 deployment script
│   └── github_actions.yml         # CI/CD pipeline
├── docs/
│   ├── API.md                     # Full API reference
│   └── ARCHITECTURE.md            # Architecture deep-dive
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.11+
- [OpenRouter API key](https://openrouter.ai) (free tier available)

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

### 3 — Generate seed data and initialize RAG

```bash
python data/seed_data.py        # ~30 seconds
python rag/retriever.py         # downloads all-MiniLM-L6-v2 (~22 MB, one-time)
```

### 4 — Start the API server

```bash
python app/main.py
# API running at http://localhost:8000
# Swagger documentation at http://localhost:8000/docs
```

### 5 — Launch the dashboard

```bash
# In a new terminal
streamlit run ui/dashboard.py
# Dashboard available at http://localhost:8501
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🔑 API Keys Setup

### OpenRouter (Required)
1. Visit [openrouter.ai](https://openrouter.ai)
2. Create account (no credit card required for free tier)
3. Dashboard → API Keys → Create Key
4. Add to `.env`: `OPENROUTER_API_KEY=sk-or-v1-...`

**Free models** (auto-fallback order):
```
nvidia/nemotron-3-super-120b-a12b:free
→ meta-llama/llama-3.1-405b-instruct:free
→ mistralai/mistral-7b-instruct:free
```

### AWS Free Tier (Optional — for CloudWatch monitoring)
1. Create account at [aws.amazon.com](https://aws.amazon.com)
2. IAM → Users → Create user → Programmatic access
3. Add credentials to `.env`:
   ```
   AWS_ACCESS_KEY_ID=your_key
   AWS_SECRET_ACCESS_KEY=your_secret
   CLOUDWATCH_ENABLED=true
   ```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/agent/query` | Multi-agent query routing |
| `GET`  | `/api/agent/recommendations/{field_id}` | Comprehensive field health assessment |
| `POST` | `/api/disease/detect` | Leaf disease detection via image |
| `GET`  | `/api/yield/forecast` | Yield forecast with narrative analysis |
| `GET`  | `/api/market/advisory` | Market price advisory with reasoning |
| `POST` | `/api/sensors/data` | Submit sensor reading |
| `GET`  | `/api/sensors/data/{field_id}` | Historical sensor data |
| `GET`  | `/api/sensors/latest/{field_id}` | Latest sensor reading |
| `POST` | `/api/aiops/monitor` | Trigger anomaly detection |
| `POST` | `/api/aiops/llm-analyse` | LLM root-cause analysis |
| `GET`  | `/api/aiops/trends/{field_id}` | Trend analysis |
| `GET`  | `/api/aiops/status` | System health metrics |
| `GET`  | `/api/alerts` | List all alerts (filterable) |
| `PATCH`| `/api/alerts/{id}/resolve` | Resolve alert |
| `POST` | `/api/remediation/execute` | Execute remediation action |
| `GET`  | `/api/remediation/history` | Auto-remediation log |
| `GET`  | `/api/dashboard/stats` | Dashboard statistics |
| `GET`  | `/health` | Health check |

Full API documentation: [docs/API.md](docs/API.md)

---

## 🛠️ System Components

### Multi-Agent Architecture
The platform employs multiple specialized agents that operate within a shared tool ecosystem:

- **Diagnostic Agent**: Analyzes field conditions through iterative tool calls, evaluating sensor data, soil conditions, pest presence, and historical patterns
- **Disease Agent**: Processes leaf images through ResNet18 computer vision and synthesizes treatment recommendations via LLM
- **Yield & Market Agent**: Forecasts production using Prophet and retrieves market insights through semantic search
- **Action Agent**: Determines and executes remediation actions based on detected issues
- **Orchestrator Agent**: Routes incoming queries to appropriate specialist agents

### Computer Vision
ResNet18 model fine-tuned on PlantVillage dataset provides disease classification across 38 plant disease categories. The model operates locally without requiring external inference APIs.

### Knowledge Retrieval
ChromaDB stores agricultural domain knowledge, including disease treatments, irrigation schedules, and market strategies. Queries are resolved through semantic similarity using locally-embedded vectors.

### Anomaly Detection & Remediation
A two-tier detection system combines rule-based threshold monitoring with LLM-driven root cause analysis. Detected anomalies trigger automatic remediation actions tracked in the audit log.

### Time-Series Forecasting
Facebook Prophet generates yield forecasts from historical production data with automatic seasonality detection and trend analysis.

---

## 💰 Cost Analysis

| Component | Service | Cost |
|---|---|---|
| API server | AWS EC2 t2.micro | $0 (750h free tier) |
| Storage | AWS S3 5GB | $0 (free tier) |
| Monitoring | AWS CloudWatch | $0 (free tier) |
| LLM inference | OpenRouter free models | $0 |
| Vector database | ChromaDB (local) | $0 |
| Dashboard hosting | Streamlit Cloud | $0 |
| Weather data | Open-Meteo | $0 |
| **Total** | | **$0/month** |

---

## 🗺️ Roadmap

- [ ] Fine-tune ResNet18 on complete PlantVillage dataset (87K images)
- [ ] Integrate ESP32 IoT sensors via MQTT protocol
- [ ] Multi-channel alerts (WhatsApp, SMS, email)
- [ ] Docker Compose for local deployment
- [ ] PostgreSQL RDS for production scaling
- [ ] LangGraph for parallel agent workflows

---

## 📝 License

MIT License — See LICENSE file for details

---

## 📚 Documentation

- [API Reference](docs/API.md) — Complete endpoint documentation
- [Architecture Deep-Dive](docs/ARCHITECTURE.md) — System design and data flows
