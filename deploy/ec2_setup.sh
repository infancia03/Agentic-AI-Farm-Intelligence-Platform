#!/bin/bash
# ================================================================
#  AgriFarm — AWS EC2 Free Tier Deployment
#  Run on a fresh Amazon Linux 2023 t2.micro instance
#  Usage: chmod +x deploy/ec2_setup.sh && ./deploy/ec2_setup.sh
# ================================================================
set -e

echo "🌾 AgriFarm EC2 Setup"
echo "====================="

# 1. System
echo "[1/7] System update..."
sudo dnf update -y -q
sudo dnf install -y python3.11 python3.11-pip git nginx -q

# 2. App directory
echo "[2/7] App directory..."
sudo mkdir -p /opt/agrifarm
sudo chown ec2-user:ec2-user /opt/agrifarm
cp -r . /opt/agrifarm/ 2>/dev/null || true
cd /opt/agrifarm

# 3. Python venv
echo "[3/7] Python environment..."
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 4. Environment
echo "[4/7] Environment..."
[ ! -f .env ] && cp .env.example .env
echo "⚠️  IMPORTANT: Edit /opt/agrifarm/.env and add your OPENROUTER_API_KEY"

# 5. Seed data + RAG
echo "[5/7] Seeding data + RAG..."
python data/seed_data.py
python rag/retriever.py

# 6. Systemd service
echo "[6/7] Creating systemd service..."
sudo tee /etc/systemd/system/agrifarm.service > /dev/null << 'EOF'
[Unit]
Description=AgriFarm Intelligence Platform
After=network.target

[Service]
Type=exec
User=ec2-user
WorkingDirectory=/opt/agrifarm
EnvironmentFile=/opt/agrifarm/.env
ExecStart=/opt/agrifarm/venv/bin/python app/main.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable agrifarm
sudo systemctl start agrifarm

# 7. Nginx
echo "[7/7] Nginx reverse proxy..."
sudo tee /etc/nginx/conf.d/agrifarm.conf > /dev/null << 'EOF'
server {
    listen 80;
    server_name _;
    client_max_body_size 10M;
    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
}
EOF

sudo nginx -t && sudo systemctl restart nginx

PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "YOUR_IP")
echo ""
echo "✅ AgriFarm deployed!"
echo "   API:   http://$PUBLIC_IP/health"
echo "   Docs:  http://$PUBLIC_IP/docs"
echo ""
echo "⚠️  Next steps:"
echo "   1. nano /opt/agrifarm/.env   → add OPENROUTER_API_KEY"
echo "   2. sudo systemctl restart agrifarm"
echo "   3. Deploy dashboard: streamlit run ui/dashboard.py"
