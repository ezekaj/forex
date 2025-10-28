"""
Check if bots have directional bias
"""
import MetaTrader5 as mt5

mt5.initialize()
mt5.login(95948709, 'To-4KyLg', 'MetaQuotes-Demo')

positions = mt5.positions_get()

if positions:
    buy_count = 0
    sell_count = 0
    
    for pos in positions:
        if pos.type == 0:  # Buy
            buy_count += 1
        else:  # Sell
            sell_count += 1
    
    print(f"BUY positions: {buy_count}")
    print(f"SELL positions: {sell_count}")
    print(f"Bias: {'BULLISH' if buy_count > sell_count else 'BEARISH' if sell_count > buy_count else 'NEUTRAL'}")
    
    # Check by bot
    bots = {}
    for pos in positions:
        if pos.magic not in bots:
            bots[pos.magic] = {'buy': 0, 'sell': 0}
        
        if pos.type == 0:
            bots[pos.magic]['buy'] += 1
        else:
            bots[pos.magic]['sell'] += 1
    
    print("\nBy Bot (Magic Number):")
    for magic, counts in bots.items():
        print(f"  Bot {magic}: {counts['buy']} BUY, {counts['sell']} SELL")

mt5.shutdown()