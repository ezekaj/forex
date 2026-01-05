# Stock Sentiment Trading Bot

A sentiment-driven stock trading bot that uses Reddit/news sentiment with Quarter-Kelly position sizing and hard circuit breakers.

## Philosophy

**The system that survives longest wins through compounding.**

This bot is designed to:
1. **Not bust** - Circuit breakers prevent catastrophic losses
2. **Maximize profits** - Trade high-conviction sentiment signals only
3. **Run autonomously** - Set it and forget it operation

## Quick Start

### 1. Install Dependencies

```bash
cd stock_sentiment_bot
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required API keys:
- **Alpaca** (required): Get at https://app.alpaca.markets/paper/dashboard/overview
- **Reddit** (required): Create app at https://www.reddit.com/prefs/apps
- **Telegram** (optional): Create bot with @BotFather

### 3. Run in Paper Trading Mode

```bash
python main.py --dry-run
```

### 4. Run Tests

```bash
pytest tests/ -v
```

## Risk Management

### Position Sizing (Quarter-Kelly)

| Signal Score | Confidence | Risk % | Example ($2000 account) |
|--------------|------------|--------|-------------------------|
| 90-100 | Very High | 5% | $100 risk |
| 80-89 | High | 4% | $80 risk |
| 70-79 | Medium | 3% | $60 risk |
| 60-69 | Low | 2% | $40 risk |
| <60 | No Trade | 0% | Skip |

### Circuit Breakers (CANNOT be overridden)

```
Daily loss limit:     -10% → Stop trading for the day
Daily trade limit:    5 trades → No more entries
Drawdown halt:        -25% → Full trading halt
Consecutive losses:   7 in a row → Stop and review
```

### Drawdown Recovery Protocol

```
0% to -10%    → Normal trading (4% risk)
-10% to -15%  → Reduced trading (2% risk)
-15% to -20%  → Minimal trading (1% risk, high conviction only)
-20% to -25%  → HALT new trades
-25%+         → FULL STOP, manual review required
```

## Project Structure

```
stock_sentiment_bot/
├── config/
│   ├── settings.py       # Configuration management
│   └── constants.py      # All magic numbers centralized
├── data/
│   ├── alpaca_client.py  # Price data + execution
│   └── reddit_scraper.py # r/wallstreetbets, r/stocks
├── analysis/
│   └── sentiment_scorer.py # Bullish/bearish scoring
├── trading/
│   ├── signal_generator.py # Combine all signals
│   ├── position_sizer.py   # Kelly-based sizing
│   └── risk_manager.py     # Circuit breakers
├── database/
│   └── models.py         # SQLite models
├── monitoring/           # Telegram alerts (TODO)
├── backtest/             # Backtesting engine (TODO)
├── tests/                # Test suite
├── main.py               # Entry point
└── requirements.txt
```

## Signal Scoring System

Signals are scored 0-100. Only trade if score >= 60.

### Sentiment Momentum (0-40 points)
- Mention spike (>3x average): 20 points
- Bullish sentiment (>0.6): 20 points

### Technical Confirmation (0-30 points)
- RSI oversold (<40): 15 points
- Volume spike (>2x average): 15 points

### Quality Filters (0-30 points)
- Unique authors (>20): 10 points (filters bots)
- Has DD post: 10 points
- High engagement: 10 points

## Paper Trading Requirements

Before going live, complete 30+ days of paper trading with:

| Metric | Minimum | Target |
|--------|---------|--------|
| Win Rate | >50% | >55% |
| Profit Factor | >1.3 | >1.8 |
| Max Drawdown | <20% | <15% |
| System Uptime | >95% | >99% |

## License

MIT

## Disclaimer

This software is for educational purposes only. Trading stocks involves substantial risk of loss. Past performance does not guarantee future results. Never risk money you cannot afford to lose.
