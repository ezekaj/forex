# Fixed Trading System - Setup Guide

## What Was Fixed

| Issue | Problem | Solution |
|-------|---------|----------|
| **#1 Data Source** | yfinance API blocked → using fake simulated data | Multi-source fallback: MT5 → Twelve Data → Alpha Vantage → Yahoo → Cache |
| **#2 News Integration** | EnhancedSignals import fails | Properly imports from `forex_system/services/enhanced_forex_signals.py` |
| **#3 Min Confidence** | 20% threshold → opens garbage trades | Increased to **35%** minimum confidence |
| **#4 Broker Connection** | No real trade execution | **MetaTrader 5** API integration |

---

## Quick Start

### 1. Install Dependencies

```bash
cd /path/to/forex
pip install -r requirements.txt
```

### 2. Get Free API Keys (RECOMMENDED)

For reliable data, get free API keys:

| Service | Free Tier | Sign Up |
|---------|-----------|---------|
| **Twelve Data** | 800 requests/day | https://twelvedata.com/ |
| **Alpha Vantage** | 25 requests/day | https://www.alphavantage.co/support/#api-key |

Set them as environment variables:
```bash
export TWELVE_DATA_KEY="your_key"
export ALPHA_VANTAGE_KEY="your_key"
```

Or add directly in the code:
```python
config = Config(
    TWELVE_DATA_KEY="your_key",
    ALPHA_VANTAGE_KEY="your_key",
)
```

### 3. Run Paper Trading (Test Mode)

```bash
python FIXED_TRADING_SYSTEM.py
```

This runs in **paper trading mode** by default - no real money involved.

---

## Live Trading Setup

### Step 1: Install MetaTrader 5

1. Download MT5 from your broker or https://www.metatrader5.com/
2. Install and login to your broker account

### Step 2: Configure Credentials

Edit `FIXED_TRADING_SYSTEM.py`:

```python
config = Config(
    PAPER_TRADING=False,  # IMPORTANT: Set to False for live trading

    MT5_LOGIN=12345678,
    MT5_PASSWORD="your_password",
    MT5_SERVER="YourBroker-Server",

    # Recommended: Get data from multiple sources
    TWELVE_DATA_KEY="your_key",
    ALPHA_VANTAGE_KEY="your_key",

    # Optional: Telegram notifications
    TELEGRAM_TOKEN="your_bot_token",
    TELEGRAM_CHAT_ID="your_chat_id",
)
```

### Step 3: Run Live

```python
import asyncio

config = Config(
    PAPER_TRADING=False,
    # ... other settings
)

bot = FixedTradingBot(config)
asyncio.run(bot.run(interval_minutes=60))  # Check every hour
```

---

## Configuration Options

```python
@dataclass
class Config:
    # Capital & Risk
    STARTING_CAPITAL: float = 30000     # Your trading capital
    RISK_PER_TRADE: float = 0.015       # 1.5% risk per trade
    MAX_POSITIONS: int = 3              # Maximum concurrent positions
    MAX_DAILY_LOSS: float = 0.05        # 5% max daily drawdown

    # Signal Quality (FIX #3)
    MIN_CONFIDENCE: float = 0.35        # Minimum 35% confidence to trade

    # Risk:Reward
    DEFAULT_RR: float = 2.5             # Risk:Reward ratio (1:2.5)

    # Trading Pairs
    PAIRS: List[str] = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]

    # Execution
    PAPER_TRADING: bool = True          # True = no real trades
```

---

## Telegram Notifications

1. Create a bot via @BotFather on Telegram
2. Get your chat ID via @userinfobot
3. Add to config:

```python
config = Config(
    TELEGRAM_TOKEN="123456:ABC-your-bot-token",
    TELEGRAM_CHAT_ID="your_chat_id",
)
```

You'll receive notifications for:
- 🟢 Trade Opened
- 💰 Trade Closed (profit)
- 📉 Trade Closed (loss)
- ⚠️ Drawdown Warning

---

## Data Source Priority

The system tries data sources in this order:

1. **MetaTrader 5** - Best for forex, real-time quotes (requires MT5 connection)
2. **Twelve Data** - Free tier: 800 requests/day
3. **Alpha Vantage** - Free tier: 25 requests/day
4. **Yahoo Finance** - Often blocked, but tries
5. **Cached Data** - Falls back to saved data
6. **Synthetic Data** - Only for testing, NOT for live trading

---

## Important Notes

1. **MetaTrader 5 only works on Windows**
   - On Mac/Linux, use paper trading mode + API data sources

2. **Always paper trade first** for at least 1 month

3. **Backtest ≠ Live trading** - expect 50-70% of backtest performance

4. **The system WILL have losing months** - this is normal

5. **Never risk more than you can afford to lose**

---

## Files

| File | Purpose |
|------|---------|
| `FIXED_TRADING_SYSTEM.py` | **Main bot with all fixes** |
| `UPGRADED_FEATURES.py` | 37 ML features |
| `requirements.txt` | Dependencies |
| `forex_system/services/enhanced_forex_signals.py` | News/VIX integration |

---

## Troubleshooting

### "MT5 not available"
- MetaTrader 5 only works on Windows
- Make sure MT5 is installed and running
- Try: `pip install MetaTrader5`

### "Symbol not found"
- Your broker may use different symbol names
- Try: EURUSD, EURUSD.m, EURUSDm, EURUSD.raw

### "Below min confidence"
- This is GOOD - the bot is rejecting weak signals
- You can lower MIN_CONFIDENCE to 0.30 for more trades (but lower quality)

### "No data available"
- Get free API keys from Twelve Data and/or Alpha Vantage
- Check your internet connection
