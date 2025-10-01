import MetaTrader5 as mt5
from datetime import datetime
import pytz

# Initialize
mt5.initialize()
mt5.login(95948709, 'To-4KyLg', 'MetaQuotes-Demo')

print("="*60)
print("MARKET STATUS CHECK")
print("="*60)

# Current time
now = datetime.now()
print(f"Your Local Time: {now.strftime('%Y-%m-%d %H:%M:%S')} (Albania)")
print(f"Day: {now.strftime('%A')}")

# Check if weekend
if now.weekday() >= 5:
    print("\n[!] WEEKEND - FOREX MARKET CLOSED!")
    print("Market reopens Sunday 22:00 GMT")
else:
    print("\n[OK] Weekday - Market should be open")

# Check server time
server_time = mt5.symbol_info('EURUSD')
if server_time:
    tick = mt5.symbol_info_tick('EURUSD')
    if tick:
        tick_time = datetime.fromtimestamp(tick.time)
        time_diff = (now - tick_time).total_seconds()
        print(f"\nLast tick time: {tick_time.strftime('%H:%M:%S')}")
        print(f"Time since last tick: {time_diff:.0f} seconds")
        
        if time_diff > 60:
            print("[!] No recent ticks - Market might be closed!")
        else:
            print("[OK] Recent ticks - Market is active!")

# Check account status
account = mt5.account_info()
print(f"\nAccount Balance: ${account.balance:.2f}")
print(f"Account Equity: ${account.equity:.2f}")
print(f"Current P&L: ${account.profit:.2f}")

# Check positions
positions = mt5.positions_get()
if positions:
    print(f"\nOpen Positions: {len(positions)}")
    print("-"*60)
    
    total_loss = 0
    for pos in positions:
        direction = "BUY" if pos.type == 0 else "SELL"
        total_loss += pos.profit
        print(f"{pos.symbol} | {direction} | Vol: {pos.volume} | P&L: ${pos.profit:.2f}")
    
    print("-"*60)
    print(f"TOTAL P&L: ${total_loss:.2f}")
    
    if total_loss < -100:
        print("\n[WARNING] Large loss detected!")
        print("The bot might be:")
        print("1. Trading during low liquidity hours")
        print("2. Opening too many positions")
        print("3. Not managing risk properly")
else:
    print("\nNo open positions")

# Check trading sessions
print("\n" + "="*60)
print("FOREX TRADING SESSIONS (GMT):")
print("="*60)
print("Sydney:   21:00 - 06:00")
print("Tokyo:    00:00 - 09:00") 
print("London:   08:00 - 17:00")
print("New York: 13:00 - 22:00")
print("")
print("BEST TRADING TIMES:")
print("• London/NY Overlap: 13:00-17:00 GMT (Most liquid)")
print("• Asian Session: 00:00-09:00 GMT (JPY pairs)")
print("• Avoid: 20:00-22:00 GMT (Low liquidity)")

# Calculate current GMT hour
gmt = pytz.timezone('GMT')
gmt_now = datetime.now(gmt)
print(f"\nCurrent GMT Time: {gmt_now.strftime('%H:%M')}")

hour = gmt_now.hour
if 13 <= hour <= 17:
    print("[EXCELLENT] London/NY overlap - Best liquidity!")
elif 8 <= hour <= 17:
    print("[GOOD] London session - Good liquidity")
elif 0 <= hour <= 9:
    print("[OK] Asian session - Good for JPY pairs")
elif 21 <= hour or hour <= 6:
    print("[OK] Sydney session - Lower liquidity")
else:
    print("[POOR] Between sessions - Low liquidity")

mt5.shutdown()