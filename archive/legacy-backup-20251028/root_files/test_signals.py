"""Test if signals are being generated"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# Initialize
mt5.initialize()
mt5.login(95948709, 'To-4KyLg', 'MetaQuotes-Demo')

symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']

print("TESTING SIGNAL GENERATION")
print("="*60)

for symbol in symbols:
    print(f"\n{symbol}:")
    
    # Check if symbol is available
    info = mt5.symbol_info(symbol)
    if not info:
        print(f"  ERROR: Symbol not found")
        continue
    
    if not info.visible:
        mt5.symbol_select(symbol, True)
    
    # Get tick
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print(f"  ERROR: No tick data")
        continue
    
    # Calculate spread
    spread = (tick.ask - tick.bid) / info.point / 10
    print(f"  Bid: {tick.bid:.5f} | Ask: {tick.ask:.5f} | Spread: {spread:.1f} pips")
    
    # Get rates
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 20)
    if rates is None or len(rates) < 20:
        print(f"  ERROR: Not enough data (got {len(rates) if rates else 0} bars)")
        continue
    
    df = pd.DataFrame(rates)
    
    # Calculate MAs
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    
    current = df['close'].iloc[-1]
    ma5 = df['ma5'].iloc[-1]
    ma10 = df['ma10'].iloc[-1]
    
    print(f"  Current: {current:.5f}")
    print(f"  MA5: {ma5:.5f} | MA10: {ma10:.5f}")
    
    # Check for signals
    if ma5 > ma10:
        diff = (ma5 - ma10) / ma10 * 100
        print(f"  -> BULLISH: MA5 > MA10 by {diff:.3f}%")
    else:
        diff = (ma10 - ma5) / ma10 * 100
        print(f"  -> BEARISH: MA10 > MA5 by {diff:.3f}%")
    
    # Check last crossover
    prev_ma5 = df['ma5'].iloc[-2]
    prev_ma10 = df['ma10'].iloc[-2]
    
    if prev_ma5 <= prev_ma10 and ma5 > ma10:
        print(f"  *** CROSSOVER UP DETECTED! ***")
    elif prev_ma5 >= prev_ma10 and ma5 < ma10:
        print(f"  *** CROSSOVER DOWN DETECTED! ***")

print("\n" + "="*60)
print("SUMMARY:")
account = mt5.account_info()
print(f"Account Balance: ${account.balance:.2f}")
print(f"Market is {'OPEN' if tick.time else 'CLOSED'}")

# Check if it's weekend
from datetime import datetime
now = datetime.now()
print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Day of week: {now.strftime('%A')}")

if now.weekday() >= 5:
    print("WARNING: It's weekend - forex market is closed!")

mt5.shutdown()