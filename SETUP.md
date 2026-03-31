# ⚡ AgriFarm — Step-by-Step Setup Guide

Complete guide to run the project locally, then deploy to AWS EC2 free tier.

---

## Part 1 — Local Development (30 minutes)

### Step 1: Python Environment

```bash
# Verify Python 3.11+
python --version

# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows PowerShell

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏱️ First install takes 5–10 minutes (downloads PyTorch, sentence-transformers, etc.)

**If PyTorch install fails (Windows/older Mac):**
```bash
# CPU-only PyTorch (smaller, faster to install)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**If Prophet install fails:**
```bash
pip install pystan==3.9.1
pip install prophet==1.1.5
```

### Step 3: Set Up Environment Variables

```bash
# Copy template
cp .env.example .env
```

Open `.env` in any editor and fill in:

```env
# REQUIRED — get free key at https://openrouter.ai
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here

# OPTIONAL — only if you have AWS account
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
```

**How to get your free OpenRouter key:**
1. Go to [openrouter.ai](https://openrouter.ai)
2. Click "Sign Up" (email only, no credit card)
3. Dashboard → API Keys → "Create Key"
4. Copy the key (starts with `sk-or-v1-`)

### Step 4: Generate All Seed Data

```bash
python data/seed_data.py
```

**Expected output:**
```
🌱 AgriFarm — Seed Data Generator
==========================================
🌱 Seeding sensor data …
  ✓ field_A1: 192 readings
  ✓ field_A2: 192 readings
  ✓ field_B1: 192 readings
  ✓ field_B2: 192 readings
  ✓ field_C1: 192 readings
📊 Generating market_prices.csv …
  ✓ 2555 market price records
🌾 Generating yield_history.csv …
  ✓ 630 yield history records
📚 Generating agri_knowledge.json …
  ✓ 24 knowledge base documents
✅ All seed data ready.
```

### Step 5: Initialise RAG (ChromaDB)

```bash
python rag/retriever.py
```

**Expected output:**
```
# First run downloads the embedding model (~22 MB, one time only):
Downloading all-MiniLM-L6-v2 ...
✓ ChromaDB: 24 documents
```

### Step 6: Start the FastAPI Backend

```bash
python app/main.py
```

**Expected output:**
```
✅ Database initialised
✓ ChromaDB: 24 documents (already ingested)
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Verify it's working:**
```bash
# In a new terminal
curl http://localhost:8000/health
# {"status":"healthy","service":"AgriFarm Intelligence Platform v2.0",...}

curl "http://localhost:8000/api/sensors/fields"
# {"fields":["field_A1","field_A2","field_B1","field_B2","field_C1"],"total":5}
```

**Swagger UI:** Open [http://localhost:8000/docs](http://localhost:8000/docs) in browser

### Step 7: Launch the Streamlit Dashboard

```bash
# New terminal (keep API running in the other one)
streamlit run ui/dashboard.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

---

## Part 2 — Testing the Features

### Test 1: AI Agent Chat
1. Open dashboard → **🤖 AI Agent Chat**
2. Click "What is the soil health status of field_A1?"
3. Watch the agent call tools and return diagnosis
4. Try: "field_B1 has high temperature — what should I do?"

### Test 2: AIOps Monitor
1. Go to **⚙️ AIOps & Alerts**
2. Click "Run AIOps Monitor"
3. Should detect anomalies in field_A1 (low moisture) and field_B1 (high temp)
4. Click "LLM Anomaly Analysis" for root-cause report

### Test 3: Disease Scanner
1. Go to **🔬 Disease Scanner**
2. Upload any plant leaf image (jpg/png)
3. Click "Analyse Disease"
4. See ResNet18 classification + LLM treatment plan

### Test 4: Yield Forecast
1. Go to **📈 Yield Forecast**
2. Select Tomato / Tamil Nadu
3. Click "Generate Forecast"
4. See Prophet chart + LLM narrative

### Test 5: Market Advisory
1. Go to **📊 Market Advisory**
2. Select Rice, quantity 20 quintals
3. Click "Get Advisory"
4. See price chart + RAG-enhanced LLM recommendation

### Test 6: API directly
```bash
# Agent query
curl -X POST http://localhost:8000/api/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What fields need irrigation?", "field_id": "field_A1"}'

# Yield forecast
curl "http://localhost:8000/api/yield/forecast?crop=Tomato&state=Tamil+Nadu"

# Market advisory
curl "http://localhost:8000/api/market/advisory?crop=Rice&quantity_quintals=20"

# AIOps LLM analysis
curl -X POST http://localhost:8000/api/aiops/llm-analyse

# Trend analysis
curl "http://localhost:8000/api/aiops/trends/field_A1?hours=24"
```

---

## Part 3 — AWS EC2 Free Tier Deployment

### Prerequisites
- AWS account (free at aws.amazon.com)
- EC2 key pair (.pem file)

### Step 1: Launch EC2 Instance
1. AWS Console → EC2 → Launch Instance
2. **Name:** agrifarm-api
3. **AMI:** Amazon Linux 2023 (free tier eligible)
4. **Instance type:** t2.micro (free tier — 750 hrs/month)
5. **Key pair:** Create new or use existing → download .pem
6. **Security Group:** Allow inbound TCP 22 (SSH), 80 (HTTP), 8000 (API)
7. **Storage:** 20 GB gp3 (free tier allows 30 GB)
8. Launch instance

### Step 2: Connect and Deploy

```bash
# Make key file secure
chmod 400 your-key.pem

# SSH into instance
ssh -i your-key.pem ec2-user@YOUR_EC2_PUBLIC_IP

# Upload project (from your local machine, in a new terminal)
scp -i your-key.pem -r ./agrifarm ec2-user@YOUR_EC2_IP:/home/ec2-user/
```

### Step 3: Run Setup Script

```bash
# On the EC2 instance
cd /home/ec2-user/agrifarm
chmod +x deploy/ec2_setup.sh
./deploy/ec2_setup.sh
```

### Step 4: Configure Environment

```bash
# On EC2
nano .env
# Add your OPENROUTER_API_KEY
# Save: Ctrl+O, Enter, Ctrl+X

# Restart service
sudo systemctl restart agrifarm
sudo systemctl status agrifarm   # should show "active (running)"
```

### Step 5: Verify

```bash
curl http://YOUR_EC2_IP/health
# {"status":"healthy",...}
```

### Step 6: Deploy Dashboard to Streamlit Cloud (free)

1. Push code to GitHub: `git push origin main`
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → select your repo → branch: main → file: `ui/dashboard.py`
4. Advanced settings → Secrets:
```toml
API_URL = "http://YOUR_EC2_PUBLIC_IP"
```
5. Deploy → get free URL: `https://agrifarm-xxx.streamlit.app`

---

## Troubleshooting

### "OPENROUTER_API_KEY not set"
```bash
cat .env   # verify key is there
source .env  # reload environment
```

### "No data for field X"
```bash
python data/seed_data.py   # regenerate seed data
```

### "ChromaDB collection empty"
```bash
python rag/retriever.py   # reinitialise RAG
```

### "Port 8000 already in use"
```bash
lsof -i :8000              # find the process
kill -9 <PID>              # kill it
python app/main.py         # restart
```

### Dashboard shows "API ✗ Offline"
```bash
# Make sure FastAPI is running
python app/main.py
# Check the API_URL in dashboard (default: http://localhost:8000)
```

### Prophet install fails
```bash
conda install -c conda-forge prophet   # if using conda
# OR
pip install pystan==3.9.1 && pip install prophet==1.1.5
```

### PyTorch too large / slow to install
```bash
# Use CPU-only version (sufficient for ResNet18 inference)
pip install torch==2.3.1+cpu torchvision==0.18.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu
```
