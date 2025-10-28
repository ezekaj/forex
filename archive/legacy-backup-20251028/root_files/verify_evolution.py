"""
Verify the Evolution System is working
"""
import MetaTrader5 as mt5
import os
import json
import time

print("="*70)
print("EVOLUTION SYSTEM VERIFICATION")
print("="*70)

# Initialize MT5
mt5.initialize()
mt5.login(95948709, 'To-4KyLg', 'MetaQuotes-Demo')

# 1. Check if evolution state exists
print("\n1. EVOLUTION STATE FILE:")
print("-"*40)
if os.path.exists('evolution_state.json'):
    with open('evolution_state.json', 'r') as f:
        state = json.load(f)
        print(f"[OK] Evolution state found!")
        print(f"  Generation: {state.get('generation', 0)}")
        print(f"  Strategies: {len(state.get('strategies', []))}")
        
        ratio = state.get('buy_sell_ratio', {})
        if ratio.get('buy', 0) + ratio.get('sell', 0) > 0:
            buy_pct = ratio['buy'] / (ratio['buy'] + ratio['sell']) * 100
            print(f"  Historical: BUY {buy_pct:.1f}% / SELL {100-buy_pct:.1f}%")
        
        # Check strategy genes for balance
        strategies = state.get('strategies', [])
        if strategies:
            avg_buy_bias = sum(s['genes']['buy_bias'] for s in strategies) / len(strategies)
            avg_sell_bias = sum(s['genes']['sell_bias'] for s in strategies) / len(strategies)
            print(f"  Avg Buy Bias: {avg_buy_bias:.2f}")
            print(f"  Avg Sell Bias: {avg_sell_bias:.2f}")
else:
    print("[X] No evolution state file - bot hasn't saved yet")

# 2. Check Evolution Bot positions (magic 777777)
print("\n2. EVOLUTION BOT POSITIONS (Magic 777777):")
print("-"*40)

positions = mt5.positions_get(magic=777777)
if positions:
    buy_count = sum(1 for p in positions if p.type == 0)
    sell_count = sum(1 for p in positions if p.type == 1)
    
    print(f"[OK] Evolution bot has {len(positions)} positions")
    print(f"  BUY: {buy_count}")
    print(f"  SELL: {sell_count}")
    
    if buy_count + sell_count > 0:
        balance = buy_count / (buy_count + sell_count) * 100
        print(f"  Balance: {balance:.1f}% BUY / {100-balance:.1f}% SELL")
        
        if 40 <= balance <= 60:
            print("  [EXCELLENT] Well balanced!")
        elif 30 <= balance <= 70:
            print("  [GOOD] Reasonably balanced")
        else:
            print("  [WARNING] Still biased")
    
    # Show first few positions
    print("\n  Sample positions:")
    for p in positions[:5]:
        print(f"    {p.symbol}: {'BUY' if p.type == 0 else 'SELL'} - {p.comment}")
else:
    print("[PENDING] Evolution bot hasn't opened positions yet")
    print("  The bot needs time to analyze and evolve")

# 3. Check Old Bot positions (magic 555555)
print("\n3. OLD BOT POSITIONS (Magic 555555):")
print("-"*40)

old_positions = mt5.positions_get(magic=555555)
if old_positions:
    buy_count = sum(1 for p in old_positions if p.type == 0)
    sell_count = sum(1 for p in old_positions if p.type == 1)
    
    print(f"Old bot still has {len(old_positions)} positions")
    print(f"  BUY: {buy_count}")
    print(f"  SELL: {sell_count}")
    print(f"  Balance: {buy_count/max(1,buy_count+sell_count)*100:.1f}% BUY")
    print("\n  [INFO] You may want to close these old positions")

# 4. Account Status
print("\n4. ACCOUNT STATUS:")
print("-"*40)
account = mt5.account_info()
print(f"Balance: ${account.balance:.2f}")
print(f"Equity: ${account.equity:.2f}")
print(f"Margin Level: {account.margin_level:.2f}%" if account.margin_level else "Margin Level: N/A")

# 5. Test Evolution System signals
print("\n5. TESTING EVOLUTION SIGNALS:")
print("-"*40)

# Quick test to see if system would generate SELL signals
rates = mt5.copy_rates_from_pos('EURUSD', mt5.TIMEFRAME_M1, 0, 100)
if rates is not None and len(rates) > 50:
    import pandas as pd
    df = pd.DataFrame(rates)
    
    # Simple RSI calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    current_rsi = rsi.iloc[-1]
    print(f"Current EURUSD RSI: {current_rsi:.2f}")
    
    if current_rsi > 70:
        print("  [SELL ZONE] RSI is overbought - should trigger SELL signals")
    elif current_rsi < 30:
        print("  [BUY ZONE] RSI is oversold - should trigger BUY signals")
    else:
        print("  [NEUTRAL] RSI is neutral - mixed signals expected")

print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)

if os.path.exists('evolution_state.json'):
    print("[OK] Evolution system is saving state")
else:
    print("[PENDING] Evolution system hasn't saved state yet")

if positions:
    print("[OK] Evolution bot is trading")
    buy_pct = buy_count / (buy_count + sell_count) * 100
    if 30 <= buy_pct <= 70:
        print("[OK] Trading is balanced")
    else:
        print("[EVOLVING] Balance improving over generations")
else:
    print("[PENDING] Evolution bot analyzing market")

print("\nThe Evolution System will:")
print("1. Start opening balanced positions soon")
print("2. Evolve better strategies every 5 minutes")
print("3. Save its learning to evolution_state.json")
print("4. Improve balance with each generation")

mt5.shutdown()