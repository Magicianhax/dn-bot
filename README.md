# Ethereal Points Farmer

A delta-neutral trading bot for farming points on [Ethereal DEX](https://ethereal.trade). Opens matched long/short positions across two accounts to earn trading volume and points while minimizing directional risk.

## Features

- **Delta-Neutral Trading**: Opens long on one account, short on another
- **Anti-Sybil Protection**: Randomly assigns which account goes long/short
- **Full Balance Mode**: Automatically trades with the smaller account's full balance
- **Random Hold Times**: Each trade pair gets a random hold time between min/max
- **Random Delay**: Configurable random delay between opening new trades
- **Daily Trade Limits**: Max trades per day (resets at midnight)
- **Web Dashboard**: Control the bot via a sleek web interface
- **Real-time Monitoring**: WebSocket updates for live PnL tracking
- **Live Logs**: See bot activity in real-time on the dashboard
- **Trade History**: SQLite database records all trades with volume stats
- **Risk Management**: Configurable stop-loss, take-profit, and hold times
- **Cloud Ready**: Deploy on Google Cloud and control from anywhere

## Quick Start

```bash
# Clone and install
git clone https://github.com/Magicianhax/dn-bot.git
cd dn-bot
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your keys

# Run dashboard
python main.py dashboard --port 3000
```

Open http://localhost:3000 and click **Start Bot**!

## Prerequisites

1. **Two Ethereal accounts** with funds deposited
2. **Linked Signers** set up for each account (see setup below)
3. Python 3.10+

## Installation

```bash
# Clone the repository
git clone https://github.com/Magicianhax/dn-bot.git
cd dn-bot

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
```

## Setting Up Linked Signers

The bot uses **linked signers** instead of your main wallet private keys for security.

1. Go to [Ethereal Trade](https://ethereal.trade)
2. Connect your wallet
3. Navigate to **Settings → Linked Signers**
4. Generate a new signer key or import one
5. Copy the **signer private key** (not your wallet key!)
6. Repeat for your second account

## Configuration

Edit `.env` with your settings:

```env
# Account 1 - Linked signer private key (NOT your wallet private key!)
ACCOUNT1_PRIVATE_KEY=0x...
# Account 1 - Your EOA wallet address
ACCOUNT1_WALLET_ADDRESS=0x...

# Account 2 - Same as above for second account
ACCOUNT2_PRIVATE_KEY=0x...
ACCOUNT2_WALLET_ADDRESS=0x...

# API Configuration
ETHEREAL_API_URL=https://api.ethereal.trade
ETHEREAL_RPC_URL=https://rpc.ethereal.trade

# Trading Settings
TRADING_PAIRS=BTCUSD,ETHUSD       # Comma-separated pairs
USE_FULL_BALANCE=true              # Use full balance of smaller account
POSITION_SIZE=50                   # Fixed USD per trade (if USE_FULL_BALANCE=false)
MIN_BALANCE_THRESHOLD=10           # Stop trading when balance below this
LEVERAGE=10                        # 1-20x

# Daily Limits
MAX_DAILY_TRADES=100               # Max trades per day (resets at midnight)

# Delay Settings (random between min/max)
MIN_TRADE_DELAY_SECONDS=60         # Minimum delay between trades
MAX_TRADE_DELAY_SECONDS=300        # Maximum delay between trades

# Risk Management
STOP_LOSS_PERCENT=0.05             # 5% stop loss
TAKE_PROFIT_PERCENT=0.05           # 5% take profit

# Hold Time (random between min/max)
MIN_HOLD_TIME_MINUTES=30           # Minimum hold time
MAX_HOLD_TIME_MINUTES=120          # Maximum hold time
```

## Usage

### Web Dashboard (Recommended)

Launch the web dashboard for full control:

```bash
python main.py dashboard --port 3000
```

Then open http://localhost:3000 in your browser.

**Dashboard Features:**
- Start/Stop bot
- View account balances
- Monitor active trades with live PnL
- View trade history
- Edit settings in real-time
- Track buy/sell volume

### Command Line

```bash
# Start farming bot directly
python main.py farm

# Check account status
python main.py status

# View account balances
python main.py balance

# Close all positions
python main.py close-all

# List available products
python main.py products

# View funding rates
python main.py funding

# Check risk metrics
python main.py risk

# Show current config
python main.py config
```

## How It Works

1. **Trade Pair Opens**:
   - Randomly selects a trading pair (BTCUSD, ETHUSD, etc.)
   - Randomly decides which account goes long vs short (anti-sybil)
   - Opens LONG on Account 1 or 2
   - Opens SHORT on the other account
   - Sets a random hold time between min/max

2. **Monitoring**:
   - Checks PnL every 5 seconds
   - Logs status every 30 seconds

3. **Trade Pair Closes** when any condition is met:
   - Stop Loss hit (e.g., -5%)
   - Take Profit hit (e.g., +5%)
   - Random hold time reached
   - Both positions close **simultaneously**

4. **Next Trade**:
   - Waits for delay period
   - Opens new pair with fresh random settings
   - Repeats until max trades reached

## Volume Tracking

The bot tracks:
- **Buy Volume**: Total notional of long positions
- **Sell Volume**: Total notional of short positions
- **Total Volume**: Both sides combined (for points farming)

Volume is recorded per trade and aggregated daily in the SQLite database.

## API Endpoints

The dashboard exposes a REST API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Bot status, balances, stats |
| `/api/bot/start` | POST | Start the bot |
| `/api/bot/stop` | POST | Stop the bot |
| `/api/bot/close-all` | POST | Close all positions |
| `/api/trades` | GET | Trade history |
| `/api/trades/active` | GET | Active trades |
| `/api/positions` | GET | Current positions |
| `/api/prices` | GET | Market prices |
| `/api/volume` | GET | Volume statistics |
| `/api/settings` | GET/POST | Get/update settings |
| `/ws` | WebSocket | Real-time updates |

## Project Structure

```
ethereal/
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── .env                    # Configuration (create from .env.example)
├── static/                 # Dashboard frontend
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── data/
│   └── trades.db          # SQLite database
└── src/
    ├── config.py          # Settings management
    ├── client.py          # Ethereal client wrapper
    ├── api/
    │   └── server.py      # FastAPI dashboard backend
    ├── database/
    │   └── db.py          # Trade history database
    ├── trading/
    │   ├── points_strategy.py  # Main farming strategy
    │   ├── orders.py
    │   ├── positions.py
    │   └── risk.py
    ├── market/
    │   ├── data.py
    │   └── websocket.py
    └── utils/
        └── logger.py
```

## Troubleshooting

### "No subaccounts found"
- Make sure you've deposited funds to your Ethereal account
- The subaccount is created automatically on first deposit

### "401 Unauthorized"
- Verify your linked signer is properly set up
- Check that the signer hasn't expired
- Ensure `ACCOUNT1_PRIVATE_KEY` is the **signer key**, not your wallet key
- Ensure `ACCOUNT1_WALLET_ADDRESS` is your **EOA wallet address**

### "Signer not linked"
- Go to https://ethereal.trade → Settings → Linked Signers
- Link the signer address derived from your private key

## Cloud Deployment

Want to run the bot 24/7 and control it from anywhere?

See **[DEPLOY.md](DEPLOY.md)** for a complete guide to deploying on Google Cloud:

- Create a $16/month VM
- Run bot as a background service
- Set up HTTPS with custom domain
- Secure with IP whitelist or password
- Access from your phone or anywhere

Quick deploy:
```bash
# On your Google Cloud VM
git clone https://github.com/Magicianhax/dn-bot.git
cd dn-bot
pip install -r requirements.txt
# Configure .env
python main.py dashboard --host 0.0.0.0 --port 3000
```

## Disclaimer

This software is for educational purposes only. Trading cryptocurrency derivatives involves substantial risk of loss. Use at your own risk. The authors are not responsible for any financial losses incurred.

## License

MIT License - See LICENSE file for details.
