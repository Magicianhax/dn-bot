# Deploying Ethereal Trading Bot on Google Cloud

This guide walks you through deploying the Ethereal Points Farming Bot on Google Cloud Platform (GCP) so you can control it from anywhere.

---

## Prerequisites

- Google Cloud account with billing enabled
- Basic terminal/SSH knowledge
- Your `.env` file with private keys ready

---

## Step 1: Create a VM Instance

### Via Google Cloud Console

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **Compute Engine** → **VM Instances**
3. Click **Create Instance**

### Recommended Settings

| Setting | Value |
|---------|-------|
| **Name** | `ethereal-bot` |
| **Region** | Choose closest to you (e.g., `us-central1`) |
| **Machine type** | `e2-small` (2 vCPU, 2GB RAM) - $13/month |
| **Boot disk** | Ubuntu 22.04 LTS, 20GB SSD |
| **Firewall** | ✅ Allow HTTP, ✅ Allow HTTPS |

### Or use gcloud CLI:

```bash
gcloud compute instances create ethereal-bot \
  --zone=us-central1-a \
  --machine-type=e2-small \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=20GB \
  --tags=http-server,https-server
```

---

## Step 2: Configure Firewall

Allow port 3000 for the dashboard:

```bash
gcloud compute firewall-rules create allow-dashboard \
  --allow tcp:3000 \
  --target-tags=http-server \
  --description="Allow dashboard access"
```

---

## Step 3: Connect to Your VM

```bash
gcloud compute ssh ethereal-bot --zone=us-central1-a
```

Or use the **SSH** button in the Cloud Console.

---

## Step 4: Install Dependencies

Run these commands on your VM:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11+
sudo apt install -y python3 python3-pip python3-venv git

# Verify Python version (should be 3.10+)
python3 --version
```

---

## Step 5: Clone and Setup the Bot

```bash
# Clone the repository
cd ~
git clone https://github.com/Magicianhax/dn-bot.git
cd dn-bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 6: Configure Environment

Create your `.env` file:

```bash
nano .env
```

Paste your configuration:

```env
# Account 1 (Linked Signer)
ACCOUNT1_PRIVATE_KEY=0x...your_linked_signer_private_key...
ACCOUNT1_WALLET_ADDRESS=0x...your_eoa_wallet_address...

# Account 2 (Linked Signer)
ACCOUNT2_PRIVATE_KEY=0x...your_linked_signer_private_key...
ACCOUNT2_WALLET_ADDRESS=0x...your_eoa_wallet_address...

# Trading Settings
TRADING_PAIRS=BTCUSD,ETHUSD
LEVERAGE=10
USE_FULL_BALANCE=true
MIN_BALANCE_THRESHOLD=10

# Risk Settings
STOP_LOSS_PERCENT=0.05
TAKE_PROFIT_PERCENT=0.05

# Time Settings
MIN_HOLD_TIME_MINUTES=30
MAX_HOLD_TIME_MINUTES=120
MIN_TRADE_DELAY_SECONDS=60
MAX_TRADE_DELAY_SECONDS=300
MAX_DAILY_TRADES=100

# API
ETHEREAL_API_URL=https://api.ethereal.trade
ETHEREAL_RPC_URL=https://rpc.ethereal.trade
```

Press `Ctrl+X`, then `Y`, then `Enter` to save.

---

## Step 7: Test the Bot

```bash
# Activate virtual environment
source venv/bin/activate

# Test the dashboard starts
python main.py dashboard --port 3000
```

Access the dashboard at: `http://YOUR_VM_EXTERNAL_IP:3000`

Find your external IP in the Cloud Console or run:
```bash
curl ifconfig.me
```

---

## Step 8: Run as a System Service

Create a systemd service to run the bot 24/7:

```bash
sudo nano /etc/systemd/system/ethereal-bot.service
```

Paste this content:

```ini
[Unit]
Description=Ethereal Trading Bot Dashboard
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/dn-bot
Environment=PATH=/home/YOUR_USERNAME/dn-bot/venv/bin
ExecStart=/home/YOUR_USERNAME/dn-bot/venv/bin/python main.py dashboard --host 0.0.0.0 --port 3000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Replace `YOUR_USERNAME` with your actual username (run `whoami` to check).

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ethereal-bot
sudo systemctl start ethereal-bot

# Check status
sudo systemctl status ethereal-bot

# View logs
sudo journalctl -u ethereal-bot -f
```

---

## Step 9: Set Up Nginx Reverse Proxy (Recommended)

Use Nginx as a reverse proxy to access via port 80 (standard HTTP):

### Install Nginx

```bash
sudo apt update
sudo apt install -y nginx
```

### Configure Nginx

```bash
sudo nano /etc/nginx/sites-available/ethereal
```

Paste this configuration:

```nginx
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

### Enable the Site

```bash
# Enable config
sudo ln -s /etc/nginx/sites-available/ethereal /etc/nginx/sites-enabled/

# Remove default site
sudo rm /etc/nginx/sites-enabled/default

# Test config
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

### Update Firewall

Make sure HTTP is allowed:

```bash
gcloud compute firewall-rules create allow-http \
  --allow tcp:80 \
  --target-tags=http-server
```

Now access your dashboard at: `http://YOUR_VM_IP` (no port needed!)

### Optional: Add HTTPS with Domain

If you have a domain pointing to your VM:

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d YOUR_DOMAIN.com
```

Access securely at: `https://YOUR_DOMAIN.com`

---

## Step 10: Security Hardening

### Option A: IP Whitelist (Recommended)

Only allow your IP to access the dashboard:

```bash
# Get your current IP
curl ifconfig.me

# Create firewall rule
gcloud compute firewall-rules create allow-dashboard-restricted \
  --allow tcp:3000 \
  --source-ranges=YOUR_IP/32 \
  --target-tags=http-server

# Delete the open rule
gcloud compute firewall-rules delete allow-dashboard
```

### Option B: Basic Authentication

Add password protection to Nginx:

```bash
# Create password file
sudo apt install -y apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd admin

# Update nginx config - add these lines inside location block:
# auth_basic "Restricted";
# auth_basic_user_file /etc/nginx/.htpasswd;

sudo systemctl restart nginx
```

---

## Quick Commands Reference

```bash
# SSH into VM
gcloud compute ssh ethereal-bot --zone=us-central1-a

# View bot status
sudo systemctl status ethereal-bot

# View live logs
sudo journalctl -u ethereal-bot -f

# Restart bot
sudo systemctl restart ethereal-bot

# Stop bot
sudo systemctl stop ethereal-bot

# Start bot
sudo systemctl start ethereal-bot

# Update bot code
cd ~/dn-bot
git pull
sudo systemctl restart ethereal-bot

# Edit environment
nano ~/dn-bot/.env
sudo systemctl restart ethereal-bot
```

---

## Cost Estimate

| Resource | Monthly Cost |
|----------|-------------|
| e2-small VM (2 vCPU, 2GB) | ~$13 |
| 20GB SSD | ~$2 |
| Network egress | ~$1 |
| **Total** | **~$16/month** |

💡 **Tip**: Use `e2-micro` for ~$6/month if you want cheaper option (may be slower).

---

## Troubleshooting

### Bot won't start
```bash
# Check logs
sudo journalctl -u ethereal-bot -n 50

# Test manually
cd ~/dn-bot
source venv/bin/activate
python main.py dashboard --port 3000
```

### Can't access dashboard
```bash
# Check if service is running
sudo systemctl status ethereal-bot

# Check if port is listening
sudo netstat -tlnp | grep 3000

# Check firewall
gcloud compute firewall-rules list
```

### WebSocket disconnecting
Make sure Nginx is configured with proper WebSocket headers (proxy_set_header Upgrade/Connection).

---

## Mobile Access

The dashboard is responsive and works on mobile browsers. Simply:

1. Open your browser on your phone
2. Navigate to `http://YOUR_VM_IP` (with Nginx) or `http://YOUR_VM_IP:3000` (without Nginx)
3. Control the bot from anywhere!

---

## Backup Your Keys

⚠️ **Important**: Your `.env` file contains private keys. Back it up securely:

```bash
# Download to your local machine
gcloud compute scp ethereal-bot:~/dn-bot/.env ~/dn-bot-backup.env --zone=us-central1-a
```

Store this backup in a secure location (encrypted USB, password manager, etc.)

---

## Alternative: Deploy with Docker

If you prefer Docker:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 3000

CMD ["python", "main.py", "dashboard", "--host", "0.0.0.0", "--port", "3000"]
```

Build and run:
```bash
docker build -t ethereal-bot .
docker run -d --name ethereal -p 3000:3000 --env-file .env ethereal-bot
```

---

Happy Trading! 🚀
