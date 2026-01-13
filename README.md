# Unified Trading Platform

**AlphaThink + Forex System = Professional Algorithmic Trading Platform**

A cutting-edge algorithmic trading platform combining AlphaThink's beautiful real-time UI with a production-ready Python trading backend. Supports crypto, forex, and stocks with AI-powered decision making.

![Platform Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![TypeScript](https://img.shields.io/badge/typescript-5.0+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Features

### Multi-Asset Support
- **Cryptocurrencies**: Binance WebSocket integration (BTC, ETH, and 10+ pairs)
- **Forex**: MT5 and OANDA broker connections (50+ currency pairs)
- **Stocks**: Yahoo Finance integration (US equities)

### AI-Powered Decision Making
- **Gemini 3.0 Pro**: "Thinking" mode with 1024-token deep reasoning
- **Multi-Agent System**: 6+ specialized agents (chart analysis, news, risk, trend)
- **Ensemble Voting**: Consensus-based decisions from multiple AI models
- **Claude API**: Advanced market analysis and pattern recognition

### Professional Trading Features
- **Multiple Trading Modes**:
  - Paper Trading (risk-free simulation)
  - Demo Trading (broker simulation accounts)
  - Live Trading (real money with safety guardrails)
  - Backtesting (historical validation)

- **Advanced Risk Management**:
  - Kelly Criterion position sizing
  - Multi-tier risk controls
  - Portfolio heat monitoring
  - Correlation-aware exposure limits
  - Dynamic stop loss / take profit

- **Technical Analysis**:
  - 90+ indicators via ta-lib
  - Custom indicators (ATR, RSI, Bollinger Bands, ADX, MACD, Pivots)
  - Pattern recognition (engulfing, doji, hammer, etc.)
  - Support/resistance detection
  - Market regime classification

### Real-Time Dashboard
- Live price charts with multiple timeframes
- AI decision transparency (see reasoning)
- Portfolio analytics and P&L tracking
- Trade history with performance metrics
- Risk exposure monitoring
- Mobile-responsive design

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│   TypeScript Frontend (React)       │
│   - Real-time charts (Recharts)     │
│   - WebSocket client                │
│   - Modern terminal UI              │
└────────────┬────────────────────────┘
             │ WebSocket + REST API
┌────────────▼────────────────────────┐
│   Python Backend (FastAPI)          │
│                                     │
│   ┌─ AI Council ─────────────┐     │
│   │ • Gemini 3.0 Pro         │     │
│   │ • Multi-Agent Ensemble   │     │
│   │ • Claude API             │     │
│   └──────────────────────────┘     │
│                                     │
│   ┌─ Trading Engine ─────────┐     │
│   │ • Strategy execution     │     │
│   │ • Position management    │     │
│   │ • Risk management        │     │
│   └──────────────────────────┘     │
│                                     │
│   ┌─ Brokers ────────────────┐     │
│   │ • Binance (crypto)       │     │
│   │ • MT5 (forex/CFDs)       │     │
│   │ • OANDA (forex)          │     │
│   └──────────────────────────┘     │
└─────────────────────────────────────┘
         │            │
         ▼            ▼
    PostgreSQL    Redis Cache
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL 14+
- Redis 7+

### Installation

1. **Clone the repository**
```bash
git clone <repo-url>
cd forex
```

2. **Backend Setup**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Frontend Setup**
```bash
cd frontend
npm install
```

4. **Environment Configuration**

Create `.env` files:

`backend/.env`:
```bash
DATABASE_URL=postgresql://user:password@localhost:5432/forex_trading
REDIS_URL=redis://localhost:6379
GEMINI_API_KEY=your_gemini_api_key
ANTHROPIC_API_KEY=your_claude_api_key
OPENAI_API_KEY=your_openai_api_key  # optional
```

`frontend/.env.local`:
```bash
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
API_KEY=your_gemini_api_key
```

5. **Start Services**

Terminal 1 - Backend:
```bash
cd backend
uvicorn api.main:app --reload --port 8000
```

Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

6. **Access Dashboard**
Open browser to `http://localhost:5173`

## 📊 Usage

### Paper Trading (Recommended Start)
1. Set initial capital and goal on setup screen
2. Click "Initialize" to start paper trading mode
3. System will analyze markets every 4 seconds
4. Watch AI decisions and execute trades automatically

### Demo Trading (Broker Simulation)
1. Configure broker demo account credentials in settings
2. Switch mode to "Demo" in dashboard
3. Trades execute on broker's demo server (real market conditions, fake money)

### Live Trading (Real Money)
⚠️ **Only use after thorough testing in paper/demo modes**

1. Configure live broker credentials (encrypted storage)
2. Set risk limits and safety guardrails
3. Enable live mode (requires confirmation)
4. Start with small positions

## 🧠 AI Decision System

### How It Works

1. **Market Data Collection**: Real-time price data from WebSocket feeds
2. **Technical Analysis**: Calculate 15+ indicators
3. **Pattern Recognition**: Detect chart patterns and market regimes
4. **AI Analysis**: Gemini/Claude analyze indicators + patterns
5. **Agent Consensus**: Multiple agents vote on decision
6. **Risk Check**: Position sizing and risk limits applied
7. **Execution**: Trade executed if confidence threshold met

### AI Agents

- **Chart Analysis Agent**: Technical patterns and indicators
- **News Sentiment Agent**: Market news and social sentiment
- **Risk Management Agent**: Portfolio risk and exposure
- **Trend Identification Agent**: Market regime classification
- **Economic Factors Agent**: Macro events and correlations
- **Gemini Orchestrator**: Deep reasoning with "thinking" mode

### Decision Transparency

Every trade includes:
- AI confidence score (0-100%)
- Detailed reasoning
- Key factors considered
- Strategy used
- Market phase
- Risk parameters (stop loss, take profit)

## 📈 Performance Metrics

Dashboard tracks:
- Total P&L (realized + unrealized)
- Win rate percentage
- Average profit per trade
- Max drawdown
- Sharpe ratio
- Profit factor
- Risk-adjusted returns

## 🛠️ Development

### Project Structure

```
forex/
├── backend/              # Python backend
│   ├── api/             # FastAPI routes and WebSocket
│   ├── brokers/         # Broker connectors (Binance, MT5, OANDA)
│   ├── ai/              # AI decision engine
│   │   ├── agents/      # Specialized agents
│   │   ├── gemini_service.py
│   │   └── orchestrator.py
│   ├── indicators/      # Technical indicators
│   ├── trading/         # Trading engine
│   ├── data/            # Data aggregation
│   ├── models/          # Pydantic models
│   └── tests/           # Backend tests
├── frontend/            # TypeScript/React frontend
│   ├── src/
│   │   ├── components/  # UI components
│   │   ├── services/    # API clients
│   │   ├── utils/       # Utilities
│   │   └── types.ts     # TypeScript types
│   └── public/
├── docs/                # Documentation
├── tests/               # Integration tests
└── docker-compose.yml   # Development environment
```

### Adding New Indicators

1. Add calculation to `backend/indicators/basic.py`
2. Update `get_all_indicators()` method
3. Expose via API in `backend/api/routes/analysis.py`
4. Display in frontend charts

### Adding New Brokers

1. Create connector in `backend/brokers/new_broker.py`
2. Extend `BaseBroker` class
3. Implement required methods (connect, get_price, execute_order)
4. Register in `backend/brokers/__init__.py`

### Testing

Backend:
```bash
cd backend
pytest
```

Frontend:
```bash
cd frontend
npm test
```

Integration:
```bash
pytest tests/integration/
```

## 🔒 Security

- API keys encrypted at rest
- WebSocket authentication
- Rate limiting on all endpoints
- No sensitive data in logs
- Broker credentials never exposed to frontend
- Regular security audits

## 📝 Configuration

### Risk Profiles

`backend/config/risk_profiles.json`:
```json
{
  "conservative": {
    "max_position_size": 0.05,
    "max_portfolio_heat": 0.10,
    "max_drawdown": 0.05
  },
  "moderate": {
    "max_position_size": 0.15,
    "max_portfolio_heat": 0.25,
    "max_drawdown": 0.15
  },
  "aggressive": {
    "max_position_size": 0.40,
    "max_portfolio_heat": 0.50,
    "max_drawdown": 0.30
  }
}
```

### Trading Strategies

Create custom strategies in `backend/trading/strategies/`:
```python
from .base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def analyze(self, asset, indicators):
        # Your logic here
        return TradeDecision(...)
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see LICENSE file

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Trading involves significant risk of loss
- Past performance does not guarantee future results
- Never trade with money you cannot afford to lose
- Always test strategies thoroughly before live trading
- Authors are not responsible for any financial losses

## 📞 Support

- GitHub Issues: Report bugs and request features
- Documentation: `/docs` folder
- Discord: [Join our community](#)

## 🎯 Roadmap

- [ ] Multi-exchange arbitrage
- [ ] Machine learning strategy optimization
- [ ] Mobile app (React Native)
- [ ] Social trading features
- [ ] Advanced backtesting engine
- [ ] Strategy marketplace
- [ ] Telegram bot integration
- [ ] Voice alerts

## 🙏 Acknowledgments

- AlphaThink for the beautiful UI foundation
- OpenAI, Anthropic, Google for AI APIs
- Binance, MT5, OANDA for market data
- Open source community

---

**Built with ❤️ by algorithmic traders, for algorithmic traders**

**Version:** 1.0.0
**Status:** Beta (Active Development)
**Last Updated:** January 2026
