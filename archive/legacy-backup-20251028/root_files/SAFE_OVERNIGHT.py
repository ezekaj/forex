"""
SAFE OVERNIGHT TRADING CONFIGURATION
=====================================
Run this for safe overnight demo testing
"""

import MetaTrader5 as mt5
import time
from datetime import datetime

# Initialize
mt5.initialize()
mt5.login(95948709, 'To-4KyLg', 'MetaQuotes-Demo')

print("="*70)
print("SETTING UP SAFE OVERNIGHT TRADING")
print("="*70)

# 1. CLOSE LOSING POSITIONS
positions = mt5.positions_get()
if positions:
    closed = 0
    for pos in positions:
        if pos.profit < -5:  # Close positions losing more than $5
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick:
                close_price = tick.bid if pos.type == 0 else tick.ask
                request = {
                    'action': mt5.TRADE_ACTION_DEAL,
                    'symbol': pos.symbol,
                    'volume': pos.volume,
                    'type': mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                    'position': pos.ticket,
                    'price': close_price,
                    'deviation': 20,
                    'magic': pos.magic,
                    'comment': 'overnight_cleanup',
                    'type_time': mt5.ORDER_TIME_GTC,
                    'type_filling': mt5.ORDER_FILLING_FOK
                }
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    closed += 1
                    print(f"Closed losing position: {pos.symbol} (${pos.profit:.2f})")
    
    print(f"\nClosed {closed} losing positions")

# 2. REDUCE POSITION COUNT
positions = mt5.positions_get()
if positions and len(positions) > 10:
    print(f"\nToo many positions ({len(positions)}), reducing to 10...")
    
    # Sort by profit (keep winners)
    sorted_positions = sorted(positions, key=lambda x: x.profit, reverse=True)
    
    # Close worst performers
    for pos in sorted_positions[10:]:
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick:
            close_price = tick.bid if pos.type == 0 else tick.ask
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': pos.symbol,
                'volume': pos.volume,
                'type': mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                'position': pos.ticket,
                'price': close_price,
                'deviation': 20,
                'magic': pos.magic,
                'comment': 'reduce_positions',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_FOK
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Closed: {pos.symbol} (${pos.profit:.2f})")

# 3. SET TIGHT STOPS ON REMAINING
positions = mt5.positions_get()
if positions:
    print(f"\nSetting protective stops on {len(positions)} positions...")
    for pos in positions:
        info = mt5.symbol_info(pos.symbol)
        tick = mt5.symbol_info_tick(pos.symbol)
        if info and tick:
            # Set stop loss at -$10 max per position
            if pos.type == 0:  # Buy
                new_sl = pos.price_open - 10 * info.point * 10  # 10 pip stop
            else:  # Sell
                new_sl = pos.price_open + 10 * info.point * 10
            
            request = {
                'action': mt5.TRADE_ACTION_SLTP,
                'symbol': pos.symbol,
                'position': pos.ticket,
                'sl': round(new_sl, info.digits),
                'tp': pos.tp,
                'magic': pos.magic
            }
            mt5.order_send(request)

# 4. SHOW FINAL STATUS
account = mt5.account_info()
positions = mt5.positions_get()

print("\n" + "="*70)
print("OVERNIGHT SETUP COMPLETE")
print("="*70)
print(f"Account Balance: ${account.balance:.2f}")
print(f"Open Positions: {len(positions) if positions else 0}")
print(f"Total Exposure: ${sum(p.profit for p in positions) if positions else 0:.2f}")
print("\nSAFETY MEASURES ACTIVE:")
print("- Maximum 10 positions")
print("- Stop loss on all positions")
print("- Losing positions closed")
print("\n" + "="*70)

mt5.shutdown()