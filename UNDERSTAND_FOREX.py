#!/usr/bin/env python
"""
UNDERSTAND YOUR FOREX SYSTEM - Plain English Explanation
"""

import time
import random

print("\n" + "="*70)
print("YOUR FOREX TRADING BOT - EXPLAINED SIMPLY")
print("="*70)

def explain():
    print("""
WHAT IS THIS SYSTEM?
--------------------
It's a computer program that trades currencies automatically.
Like a robot that buys and sells money 24/7 to make profit.

HOW FOREX WORKS:
----------------
EUR/USD = 1.1850 means:
- 1 Euro = 1.1850 US Dollars
- If it goes UP to 1.1900 = Euro got stronger = You make money if you bought
- If it goes DOWN to 1.1800 = Euro got weaker = You lose money if you bought

EXAMPLE TRADE:
--------------
1. Bot sees EUR/USD at 1.1850
2. Bot thinks price will go UP (using math/indicators)
3. Bot BUYS EUR/USD
4. Price goes to 1.1900
5. Bot SELLS and makes profit
6. Profit = 50 pips = $50 on $1000 trade
""")
    
    input("\nPress Enter to see HOW THE BOT DECIDES...")
    
    print("""
HOW YOUR BOT DECIDES TO TRADE:
-------------------------------
The bot checks 5 things every few minutes:

1. TREND - Is price going up or down?
   - Last 20 prices average vs Last 50 prices average
   - If 20 > 50 = Uptrend = Consider buying

2. RSI (Strength Indicator) - Is it too high or too low?
   - Number from 0-100
   - Below 30 = Oversold (too cheap) = Good to BUY
   - Above 70 = Overbought (too expensive) = Good to SELL

3. MOMENTUM - How fast is price moving?
   - Fast movement = Strong signal
   - Slow movement = Weak signal

4. SUPPORT/RESISTANCE - Important price levels
   - Support = Price floor (tends to bounce up)
   - Resistance = Price ceiling (tends to bounce down)

5. VOLUME - How many people are trading?
   - More volume = Stronger move
   - Less volume = Weaker move

CONFIDENCE CALCULATION:
-----------------------
Bot combines all 5 indicators:
- All 5 agree = 80-85% confidence
- 3-4 agree = 65-75% confidence  
- Less than 3 = NO TRADE (too risky)
""")
    
    input("\nPress Enter to see a LIVE EXAMPLE...")
    
    # Live simulation
    simulate_trade()

def simulate_trade():
    print("\n" + "="*70)
    print("LIVE TRADING SIMULATION")
    print("="*70)
    print("\nStarting Balance: $1,000")
    print("\n[SCANNING MARKET...]")
    time.sleep(1)
    
    # Simulate market analysis
    price = 1.1850
    print(f"\nEUR/USD Current Price: {price}")
    time.sleep(1)
    
    print("\n[ANALYZING INDICATORS...]")
    time.sleep(1)
    
    # Show each indicator
    indicators = []
    
    print("1. TREND: Upward (20-day average > 50-day average) -> BUY SIGNAL")
    indicators.append("BUY")
    time.sleep(0.5)
    
    print("2. RSI: 32 (Oversold, below 30) -> BUY SIGNAL")
    indicators.append("BUY")
    time.sleep(0.5)
    
    print("3. MOMENTUM: Positive (MACD above signal) -> BUY SIGNAL")
    indicators.append("BUY")
    time.sleep(0.5)
    
    print("4. SUPPORT: Price near support level -> BUY SIGNAL")
    indicators.append("BUY")
    time.sleep(0.5)
    
    print("5. VOLUME: Above average -> CONFIRMS SIGNAL")
    
    confidence = 75  # 4 out of 5 indicators agree
    
    print(f"\n[DECISION]")
    print(f"Signal: BUY")
    print(f"Confidence: {confidence}% (4 of 5 indicators agree)")
    print(f"Action: EXECUTE TRADE")
    
    input("\nPress Enter to execute trade...")
    
    # Execute trade
    print("\n[EXECUTING TRADE]")
    print(f"Buying EUR/USD at {price}")
    print(f"Position Size: $1,000 (0.01 lots)")
    print(f"Stop Loss: {price - 0.0020:.4f} (-20 pips = -$20 max loss)")
    print(f"Take Profit: {price + 0.0040:.4f} (+40 pips = +$40 target)")
    print(f"Risk/Reward: 1:2 (Risk $20 to make $40)")
    
    input("\nPress Enter to see what happens...")
    
    # Simulate outcome
    win = random.random() < 0.6  # 60% win probability
    
    print("\n[WAITING FOR PRICE MOVEMENT...]")
    time.sleep(2)
    
    if win:
        print("\n[RESULT: WIN!]")
        print(f"Price hit Take Profit: {price + 0.0040:.4f}")
        print(f"Profit: +40 pips = +$40")
        print(f"New Balance: $1,040 (+4%)")
    else:
        print("\n[RESULT: LOSS]")
        print(f"Price hit Stop Loss: {price - 0.0020:.4f}")
        print(f"Loss: -20 pips = -$20")
        print(f"New Balance: $980 (-2%)")
    
    print("""
KEY POINTS:
-----------
- Risk only 2% per trade ($20 on $1000)
- Target 2x your risk ($40 profit)
- With 60% win rate: Win $40 x 6 = $240, Lose $20 x 4 = $80
- Net profit = $160 per 10 trades
- This is how you profit even losing 40% of trades!
""")

def show_commands():
    print("\n" + "="*70)
    print("YOUR SYSTEM COMMANDS")
    print("="*70)
    print("""
TO TEST WITHOUT MONEY:
----------------------
python paper_trader.py
  - Uses fake $1000
  - Real strategy
  - See if it works

python REAL_SYSTEM.py
  - Tests on past data
  - Shows actual performance
  - Includes all costs

TO TRADE WITH REAL MONEY:
--------------------------
1. Get broker account (OANDA, eToro, etc.)
2. Add API keys to config
3. Start with $100-500
4. python start_live_24_7.py

CURRENT SYSTEM STATUS:
----------------------
[OK] Strategy: Fixed and working (70-80% confidence)
[OK] Risk Management: Proper (1-2% per trade)
[OK] Backtesting: Realistic (includes spread/commission)
[!!] Broker: Not connected (need account to trade real money)

THE TRUTH:
----------
- Most traders lose money (90%)
- Good traders make 10-20% per YEAR
- This is not get-rich-quick
- Practice for months before real money
- Psychology is harder than strategy
""")

# Run explanation
explain()
show_commands()

print("""
BOTTOM LINE:
------------
Your bot watches EUR/USD 24/7 and trades when it sees good opportunities.
It's like having a tireless trader that never sleeps, never gets emotional,
and follows rules perfectly. But it's not magic - it can still lose money.

Start with paper trading to see if it actually works!
""")

input("\nPress Enter to exit...")