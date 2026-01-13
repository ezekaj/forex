# Quick Start Guide - Unified Trading Platform

Get up and running in 5 minutes!

## Prerequisites

Install these first:
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (recommended)
- OR: Python 3.10+, Node.js 18+, PostgreSQL 14+, Redis 7+

## Option 1: Docker (Easiest)

### 1. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API keys
# At minimum, you need:
# - GEMINI_API_KEY (get from https://ai.google.dev/)
```

### 2. Start Everything

```bash
# Start all services
docker-compose up

# That's it! 🎉
```

### 3. Access Dashboard

Open browser to: **http://localhost:5173**

Services running:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Database UI: http://localhost:8080 (Adminer)
- Redis UI: http://localhost:8081

## Option 2: Manual Setup

### 1. Start Database Services

```bash
# Start PostgreSQL
# Windows: Use installer from postgresql.org
# Mac: brew install postgresql && brew services start postgresql
# Linux: sudo apt install postgresql && sudo systemctl start postgresql

# Start Redis
# Windows: Download from github.com/microsoftarchive/redis/releases
# Mac: brew install redis && brew services start redis
# Linux: sudo apt install redis && sudo systemctl start redis

# Create database
psql -U postgres -c "CREATE DATABASE forex_trading;"
psql -U postgres -c "CREATE USER trader WITH PASSWORD 'trader_password';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE forex_trading TO trader;"
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp ../.env.example .env
# Edit .env with your API keys

# Run migrations (if any)
alembic upgrade head

# Start backend
uvicorn api.main:app --reload --port 8000
```

### 3. Frontend Setup

```bash
# In a new terminal
cd frontend

# Install dependencies
npm install

# Configure environment
cp ../.env.example .env.local
# Edit .env.local with API URL and keys

# Start frontend
npm run dev
```

### 4. Access Dashboard

Open browser to: **http://localhost:5173**

## First Trade Simulation

### 1. Initial Setup Screen

When you first open the dashboard:
- Set **Starting Capital**: e.g., 10,000 EUR
- Set **Target**: e.g., 100,000 EUR
- Click **Initialize**

### 2. Paper Trading Mode

- System starts in **Paper Trading** (risk-free simulation)
- Real market data, fake money
- Perfect for learning and testing

### 3. Watch the AI Work

1. Select an asset (BTC, ETH, etc.) from the ticker bar
2. Click **Run** (or enable Auto mode)
3. AI analyzes market every 4 seconds:
   - Calculates technical indicators
   - Detects patterns
   - Generates trading decision
   - Shows detailed reasoning
4. If confidence > 50%, trade executes automatically
5. Watch your portfolio grow (or shrink!) in real-time

### 4. View AI Reasoning

Click **AI Brain** tab to see:
- Current market phase (Trending/Ranging)
- Confidence score
- Key factors
- Strategy used
- Reasoning

### 5. Monitor Portfolio

Click **Assets** tab to see:
- Current positions
- Unrealized P&L
- Entry prices
- Stop loss / Take profit levels

## Next Steps

### Enable More Assets

Edit `backend/data/aggregator.py` to add more symbols:
```python
SYMBOLS = {
    "BTCUSDT": {"type": "crypto", "exchange": "binance"},
    "ETHUSDT": {"type": "crypto", "exchange": "binance"},
    "EURUSD": {"type": "forex", "exchange": "oanda"},
    "AAPL": {"type": "stock", "exchange": "yahoo"},
    # Add more here
}
```

### Connect Real Broker (Demo Account)

1. Get demo account from OANDA or MT5
2. Add credentials to `.env`:
   ```bash
   OANDA_API_KEY=your_demo_api_key
   OANDA_ACCOUNT_ID=your_demo_account_id
   OANDA_PRACTICE=true
   ```
3. Switch to "Demo" mode in dashboard

### Customize AI Strategy

Edit `backend/ai/gemini_service.py` to modify:
- Analysis prompt
- Indicator weights
- Strategy selection
- Risk parameters

### Add Custom Indicators

Create new indicator in `backend/indicators/basic.py`:
```python
@staticmethod
def calculate_your_indicator(candles: List[Candle]) -> float:
    # Your calculation here
    return result
```

Add to `get_all_indicators()` method.

## Troubleshooting

### "Connection refused" error

Backend not running. Check:
```bash
cd backend
uvicorn api.main:app --reload --port 8000
```

### "Database connection failed"

PostgreSQL not running. Check:
```bash
# Windows
services.msc → PostgreSQL service → Start
# Mac
brew services start postgresql
# Linux
sudo systemctl start postgresql
```

### "API key invalid"

Check `.env` file has valid API keys:
```bash
GEMINI_API_KEY=AIza...  # Get from https://ai.google.dev/
```

### Frontend shows no data

WebSocket not connected. Check:
1. Backend is running (port 8000)
2. VITE_WS_URL in `frontend/.env.local` is correct
3. Browser console for errors (F12)

### Docker issues

```bash
# Stop all containers
docker-compose down

# Remove volumes (fresh start)
docker-compose down -v

# Rebuild everything
docker-compose up --build
```

## Common Questions

**Q: Can I use real money?**
A: Yes, but only after extensive testing in paper/demo modes. Set `ENABLE_LIVE_TRADING=true` and configure live broker credentials. Use at your own risk!

**Q: Which brokers are supported?**
A: Currently: Binance (crypto), MT5 (forex/CFDs), OANDA (forex). More coming soon.

**Q: How do I backtest strategies?**
A: Use the backtrader integration (coming in next release). For now, paper trade for several weeks to validate.

**Q: Can I add more AI models?**
A: Yes! The system supports multiple AI models. Edit `backend/ai/orchestrator.py` to add more agents and weight their votes.

**Q: Is my data secure?**
A: API keys are stored in environment variables (never in code). WebSocket connections can be authenticated. For production, use proper secrets management.

**Q: What's the cost?**
A: Software is free (MIT license). Costs:
- Gemini API: ~$0.01 per analysis (if using thinking mode)
- Hosting: $5-20/month (or free on local machine)
- Broker fees: Varies by broker

**Q: Can I modify the UI?**
A: Absolutely! Frontend is in `frontend/src`. Built with React and TypeScript. Hot reload enabled.

**Q: How accurate is the AI?**
A: No trading system is 100% accurate. Always use stop losses, never risk more than you can afford to lose, and thoroughly test before live trading.

## Help & Support

- **Issues**: Open a GitHub issue
- **Questions**: Start a discussion on GitHub
- **Documentation**: See `/docs` folder
- **API Docs**: http://localhost:8000/docs (when backend running)

---

## What's Next?

Now that you're up and running:

1. ✅ **Watch paper trading for a few days** - Observe AI decisions
2. ✅ **Analyze performance** - Check win rate, P&L, drawdown
3. ✅ **Adjust risk settings** - Modify position sizing, stop loss
4. ✅ **Add more assets** - Diversify across crypto, forex, stocks
5. ✅ **Test custom strategies** - Implement your own indicators
6. ✅ **Connect demo broker** - Test with real market execution
7. ⚠️ **Go live** (optional, advanced) - Only after thorough validation

**Remember**: Trading involves risk. Start small, test thoroughly, never risk money you can't afford to lose.

---

**Happy Trading! 📈🚀**
