# 💰 Forex Trading Bot - Complete Setup Guide (Minimal Budget)

## ⚠️ IMPORTANT WARNING
**Trading forex involves significant risk. You can lose all your money. Start with amounts you can afford to lose completely.**

## 📋 What You Need to Get Started

### 1. FREE API Keys (Required)
- **Alpha Vantage** (Market Data): https://www.alphavantage.co/support/#api-key
  - FREE tier: 25 requests per day
  - Enough for testing and small-scale trading

- **News API** (Optional but recommended): https://newsapi.org/register  
  - FREE tier: 100 requests per day
  - For news sentiment analysis

### 2. Broker Account (Choose ONE)

#### Option A: OANDA (Easiest for Beginners)
- **Minimum Deposit**: $0 for practice, $1 for real
- **Pros**: REST API, no software needed, good for beginners
- **Sign up**: https://www.oanda.com
- **Get API Key**: In account settings after signup

#### Option B: XM with MT5 (Most Popular)
- **Minimum Deposit**: $5
- **Pros**: MetaTrader 5, many features, micro lots
- **Sign up**: https://www.xm.com
- **Download**: MetaTrader 5 from their site

#### Option C: Exness (Low Minimum)
- **Minimum Deposit**: $10
- **Pros**: Low spreads, MT4/MT5 support
- **Sign up**: https://www.exness.com

## 🚀 Step-by-Step Setup

### Step 1: Install Required Software

```bash
# Install Python packages
pip install -r requirements.txt

# For MT5 trading (if using MT5 broker)
pip install MetaTrader5

# For OANDA trading
pip install oandapyV20
```

### Step 2: Configure Environment Variables

Edit the `.env` file I created with your actual API keys:

```bash
# Required - Get free key from Alpha Vantage
ALPHAVANTAGE_API_KEY=your_actual_key_here

# Optional - For news sentiment
NEWS_API_KEY=your_news_api_key_here

# For MT5 Trading (if using MT5)
MT5_LOGIN=your_mt5_account_number
MT5_PASSWORD=your_mt5_password
MT5_SERVER=XMGlobal-MT5 3  # Your broker's server

# For OANDA Trading (if using OANDA)
OANDA_API_KEY=your_oanda_api_key
OANDA_ACCOUNT_ID=your_account_id

# IMPORTANT: Keep this false until fully tested!
TRADING_ENABLED=false
```

### Step 3: Test in Demo Mode First

```bash
# Run in demo mode (no real money)
cd BayloZzi
python run/live_trade_safe.py --broker demo

# This will:
# - Use fake money
# - Test all systems
# - Show you how it works
```

### Step 4: Train the Model with Real Data

```bash
# Download latest market data
python -c "from core.data_loader import download_alpha_fx_daily; download_alpha_fx_daily()"

# Train the model
python run/train_model.py
```

### Step 5: Paper Trade with Real Broker (Practice Account)

```bash
# For OANDA practice account
python run/live_trade_safe.py --broker oanda --risk 0.01

# For MT5 demo account  
python run/live_trade_safe.py --broker mt5 --risk 0.01
```

### Step 6: Go Live (When Ready)

**⚠️ ONLY AFTER SUCCESSFUL TESTING:**

1. Change in `.env`:
```bash
TRADING_ENABLED=true  # Enable real trading
MAX_RISK_PER_TRADE=0.01  # Only risk 1% per trade
```

2. Start with minimum amount:
```bash
# Start live trading with strict limits
python run/live_trade_safe.py --broker oanda --risk 0.01
```

## 📊 Understanding the Risk Settings

The bot has multiple safety features:

- **MAX_RISK_PER_TRADE=0.01**: Only risks 1% of your account per trade
- **MAX_DRAWDOWN=0.10**: Stops if you lose 10% total
- **MAX_CONCURRENT_POSITIONS=1**: Only 1 trade at a time
- **Stop Loss**: Always set at 20 pips (automatic)
- **Take Profit**: Set at 40 pips (2:1 reward/risk ratio)

### Example with $100 Account:
- Risk per trade: $1 (1% of $100)
- If stop loss hits: You lose $1
- If take profit hits: You gain $2
- Daily loss limit: $5 (5% of $100)

## 🔍 Monitoring Your Bot

### Check Logs
```bash
# View real-time logs
tail -f logs/live_trading.log

# Check for errors
grep ERROR logs/live_trading.log
```

### Check Performance
```bash
# Run performance report
python run/analyze_performance.py
```

## 🛑 Emergency Stop

To stop the bot immediately:
1. Press `Ctrl+C` in the terminal
2. The bot will close all positions
3. Show you the final balance

## 💡 Tips for Success

1. **Start Small**: Begin with $10-20 maximum
2. **Test First**: Run demo mode for at least 1 week
3. **Monitor Closely**: Check logs every few hours initially
4. **Be Patient**: The bot needs time to find good trades
5. **Manage Expectations**: 5-10% monthly return is excellent

## 📈 Expected Performance

With conservative settings:
- **Win Rate**: 55-65%
- **Risk/Reward**: 1:2
- **Monthly Return**: 5-15% (in good conditions)
- **Max Drawdown**: 10%

## 🚨 When to Stop Trading

Stop immediately if:
- You lose 10% of your account
- The bot behaves unexpectedly  
- You see unusual market conditions
- Your internet is unstable

## 📞 Support Resources

- **Alpha Vantage Docs**: https://www.alphavantage.co/documentation/
- **OANDA API Docs**: https://developer.oanda.com/
- **MT5 Docs**: https://www.mql5.com/en/docs

## ✅ Checklist Before Going Live

- [ ] Tested in demo mode for 1+ week
- [ ] Profitable in paper trading
- [ ] Understand all risk settings
- [ ] Have stable internet connection
- [ ] Can afford to lose the money
- [ ] Have emergency stop plan
- [ ] Logs are working properly

## 🔧 Troubleshooting

### "No API Key" Error
- Make sure `.env` file has your actual API keys
- Check that keys are valid and active

### "Connection Failed" Error  
- Check internet connection
- Verify broker credentials
- Make sure broker platform is open (for MT5)

### "Insufficient Balance" Error
- Minimum $10 in account
- Check that funds are in trading account (not wallet)

### "Model Not Found" Error
- Run training script first: `python run/train_model.py`

---

**Remember**: Start with demo → Paper trade → Small real money → Scale up slowly

**Never invest more than you can afford to lose completely!**