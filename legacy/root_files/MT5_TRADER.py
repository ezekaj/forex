"""
MT5 TRADER - A SIMPLE, WORKING FOREX TRADING BOT
=================================================
This bot:
1. Monitors prices every 100ms
2. Uses small positions (0.01-0.1 lots based on balance)
3. Checks spread before entry (max 2 pips)
4. Exits quickly (5-10 pip targets)
5. Manages risk properly
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Account settings
LOGIN = 95948709
PASSWORD = 'To-4KyLg'
SERVER = 'MetaQuotes-Demo'
MAGIC = 123456

# Trading parameters
SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']  # Most liquid pairs
MAX_SPREAD = 2.0  # Maximum spread in pips
MIN_VOLUME = 0.01  # Minimum lot size
MAX_VOLUME = 0.5   # Maximum lot size
RISK_PERCENT = 0.5  # Risk 0.5% per trade

# Trade management (in pips)
STOP_LOSS = 10      # Stop loss in pips
TAKE_PROFIT = 10    # Take profit in pips  
BREAKEVEN_TRIGGER = 5  # Move SL to breakeven after 5 pips
TRAILING_DISTANCE = 3   # Trail stop by 3 pips

# Position limits
MAX_POSITIONS = 3  # Maximum open positions
MAX_PER_SYMBOL = 1  # Max 1 position per symbol

# Timing
SCAN_INTERVAL_MS = 100  # Scan every 100ms
TRADE_COOLDOWN = 30  # Seconds between trades on same symbol

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def initialize_mt5():
    """Initialize MT5 connection"""
    if not mt5.initialize():
        print("[ERROR] MT5 initialization failed")
        return False
    
    if not mt5.login(LOGIN, PASSWORD, SERVER):
        print(f"[ERROR] Login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False
    
    account = mt5.account_info()
    if not account:
        print("[ERROR] Could not get account info")
        return False
    
    print(f"[OK] Connected to MT5")
    print(f"     Account: {account.login}")
    print(f"     Balance: ${account.balance:.2f}")
    print(f"     Leverage: 1:{account.leverage}")
    
    return True

def get_symbol_info(symbol):
    """Get symbol information"""
    info = mt5.symbol_info(symbol)
    if not info:
        return None
    
    if not info.visible:
        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)
    
    return info

def calculate_position_size(symbol, stop_loss_pips):
    """Calculate position size based on risk"""
    account = mt5.account_info()
    if not account:
        return MIN_VOLUME
    
    balance = account.balance
    risk_amount = balance * (RISK_PERCENT / 100)
    
    # Get symbol info
    symbol_info = get_symbol_info(symbol)
    if not symbol_info:
        return MIN_VOLUME
    
    # Calculate pip value
    tick_value = symbol_info.trade_tick_value
    tick_size = symbol_info.trade_tick_size
    
    if tick_value == 0 or tick_size == 0:
        return MIN_VOLUME
    
    # Pip value for 1 lot
    pip_value = (tick_value * 10 * tick_size) / tick_size
    
    # Calculate lots
    lots = risk_amount / (stop_loss_pips * pip_value)
    
    # Apply limits
    lots = max(MIN_VOLUME, min(lots, MAX_VOLUME))
    
    # Round to valid step
    step = symbol_info.volume_step
    lots = round(lots / step) * step
    
    return lots

def check_spread(symbol):
    """Check if spread is acceptable"""
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return False, 999
    
    symbol_info = get_symbol_info(symbol)
    if not symbol_info:
        return False, 999
    
    spread_points = tick.ask - tick.bid
    spread_pips = spread_points / symbol_info.point / 10
    
    return spread_pips <= MAX_SPREAD, spread_pips

def get_open_positions():
    """Get all open positions"""
    positions = mt5.positions_get()
    if positions is None:
        return []
    return list(positions)

def has_position_for_symbol(symbol):
    """Check if we already have a position for this symbol"""
    positions = get_open_positions()
    for pos in positions:
        if pos.symbol == symbol and pos.magic == MAGIC:
            return True
    return False

# ============================================================================
# SIGNAL DETECTION
# ============================================================================

def get_signal(symbol):
    """Get trading signal for symbol"""
    
    # Don't trade if we already have a position
    if has_position_for_symbol(symbol):
        return None
    
    # Check spread
    spread_ok, spread = check_spread(symbol)
    if not spread_ok:
        return None
    
    # Get recent price data (last 20 1-minute bars)
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 20)
    if rates is None or len(rates) < 20:
        return None
    
    df = pd.DataFrame(rates)
    
    # Calculate indicators
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    
    # Current values
    current_price = df['close'].iloc[-1]
    ma5 = df['ma5'].iloc[-1]
    ma10 = df['ma10'].iloc[-1]
    prev_ma5 = df['ma5'].iloc[-2]
    prev_ma10 = df['ma10'].iloc[-2]
    
    # Look for crossovers
    signal = None
    
    # Bullish crossover: MA5 crosses above MA10
    if prev_ma5 <= prev_ma10 and ma5 > ma10:
        momentum = (current_price - ma10) / ma10
        if momentum > 0.0001:  # 0.01% minimum momentum
            signal = {
                'type': 'BUY',
                'reason': 'MA crossover UP',
                'strength': min(momentum * 1000, 1.0)  # 0-1 scale
            }
    
    # Bearish crossover: MA5 crosses below MA10
    elif prev_ma5 >= prev_ma10 and ma5 < ma10:
        momentum = (ma10 - current_price) / ma10
        if momentum > 0.0001:
            signal = {
                'type': 'SELL', 
                'reason': 'MA crossover DOWN',
                'strength': min(momentum * 1000, 1.0)
            }
    
    # Strong trend following
    elif ma5 > ma10:
        momentum = (ma5 - ma10) / ma10
        if momentum > 0.0002:  # 0.02% difference
            signal = {
                'type': 'BUY',
                'reason': 'Trend UP',
                'strength': min(momentum * 500, 1.0)
            }
    elif ma5 < ma10:
        momentum = (ma10 - ma5) / ma10
        if momentum > 0.0002:
            signal = {
                'type': 'SELL',
                'reason': 'Trend DOWN', 
                'strength': min(momentum * 500, 1.0)
            }
    
    return signal

# ============================================================================
# TRADE EXECUTION
# ============================================================================

def open_trade(symbol, signal):
    """Open a trade based on signal"""
    
    # Get symbol info
    symbol_info = get_symbol_info(symbol)
    if not symbol_info:
        return False
    
    # Get current tick
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return False
    
    # Calculate position size
    volume = calculate_position_size(symbol, STOP_LOSS)
    
    # Determine trade parameters
    if signal['type'] == 'BUY':
        trade_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        sl = price - STOP_LOSS * symbol_info.point * 10
        tp = price + TAKE_PROFIT * symbol_info.point * 10
    else:
        trade_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        sl = price + STOP_LOSS * symbol_info.point * 10
        tp = price - TAKE_PROFIT * symbol_info.point * 10
    
    # Prepare trade request
    request = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': symbol,
        'volume': volume,
        'type': trade_type,
        'price': price,
        'sl': round(sl, symbol_info.digits),
        'tp': round(tp, symbol_info.digits),
        'deviation': 10,
        'magic': MAGIC,
        'comment': signal['reason'],
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_FOK
    }
    
    # Send trade request
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"\n[FAILED] {symbol} order failed: {result.comment}")
        return False
    
    # Log successful trade
    spread_ok, spread = check_spread(symbol)
    print(f"\n[OPENED] {signal['type']} {symbol} x{volume:.2f}")
    print(f"         Price: {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f}")
    print(f"         Spread: {spread:.1f} pips | Signal: {signal['strength']:.0%}")
    
    return True

# ============================================================================
# POSITION MANAGEMENT
# ============================================================================

def manage_position(position):
    """Manage an open position"""
    
    if position.magic != MAGIC:
        return
    
    symbol = position.symbol
    ticket = position.ticket
    
    # Get current tick
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return
    
    symbol_info = get_symbol_info(symbol)
    if not symbol_info:
        return
    
    # Calculate profit in pips
    if position.type == mt5.ORDER_TYPE_BUY:
        current_price = tick.bid
        pips = (current_price - position.price_open) / symbol_info.point / 10
    else:
        current_price = tick.ask
        pips = (position.price_open - current_price) / symbol_info.point / 10
    
    # Breakeven management
    if pips >= BREAKEVEN_TRIGGER:
        new_sl = position.price_open
        
        # Only move SL if it's better than current
        if position.type == mt5.ORDER_TYPE_BUY and new_sl > position.sl:
            modify_position(position, new_sl, position.tp)
            print(f"\n[BREAKEVEN] {symbol} SL moved to entry")
        elif position.type == mt5.ORDER_TYPE_SELL and new_sl < position.sl:
            modify_position(position, new_sl, position.tp)
            print(f"\n[BREAKEVEN] {symbol} SL moved to entry")
    
    # Trailing stop
    elif pips >= BREAKEVEN_TRIGGER + TRAILING_DISTANCE:
        if position.type == mt5.ORDER_TYPE_BUY:
            new_sl = current_price - TRAILING_DISTANCE * symbol_info.point * 10
            if new_sl > position.sl:
                modify_position(position, new_sl, position.tp)
                print(f"\n[TRAILING] {symbol} SL -> {new_sl:.5f}")
        else:
            new_sl = current_price + TRAILING_DISTANCE * symbol_info.point * 10
            if new_sl < position.sl:
                modify_position(position, new_sl, position.tp)
                print(f"\n[TRAILING] {symbol} SL -> {new_sl:.5f}")

def modify_position(position, new_sl, new_tp):
    """Modify position SL/TP"""
    
    symbol_info = get_symbol_info(position.symbol)
    if not symbol_info:
        return False
    
    request = {
        'action': mt5.TRADE_ACTION_SLTP,
        'symbol': position.symbol,
        'position': position.ticket,
        'sl': round(new_sl, symbol_info.digits),
        'tp': round(new_tp, symbol_info.digits),
        'magic': MAGIC
    }
    
    result = mt5.order_send(request)
    return result.retcode == mt5.TRADE_RETCODE_DONE

# ============================================================================
# MAIN TRADING LOOP
# ============================================================================

def main():
    """Main trading loop"""
    
    # Initialize MT5
    if not initialize_mt5():
        return
    
    print("\n" + "="*60)
    print("MT5 TRADER - SIMPLE & EFFECTIVE")
    print("="*60)
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Max Spread: {MAX_SPREAD} pips")
    print(f"Risk: {RISK_PERCENT}% per trade")
    print(f"Targets: {STOP_LOSS}/{TAKE_PROFIT} pips")
    print(f"Scan Rate: {SCAN_INTERVAL_MS}ms")
    print("="*60)
    print("\nPress Ctrl+C to stop\n")
    
    last_trade_time = {}
    cycle_count = 0
    
    try:
        while True:
            cycle_count += 1
            
            # Get account info
            account = mt5.account_info()
            if not account:
                continue
            
            # Get current positions
            positions = get_open_positions()
            open_count = len([p for p in positions if p.magic == MAGIC])
            
            # Display status (every 10 cycles = 1 second)
            if cycle_count % 10 == 0:
                total_profit = sum(p.profit for p in positions if p.magic == MAGIC)
                print(f"\r[STATUS] Balance: ${account.balance:.2f} | Equity: ${account.equity:.2f} | Open: {open_count}/{MAX_POSITIONS} | P&L: ${total_profit:.2f}     ", end='')
            
            # Manage existing positions
            for position in positions:
                if position.magic == MAGIC:
                    manage_position(position)
            
            # Look for new trades if we have room
            if open_count < MAX_POSITIONS:
                for symbol in SYMBOLS:
                    # Check cooldown
                    if symbol in last_trade_time:
                        if time.time() - last_trade_time[symbol] < TRADE_COOLDOWN:
                            continue
                    
                    # Get signal
                    signal = get_signal(symbol)
                    if signal and signal['strength'] >= 0.3:  # Minimum 30% strength
                        if open_trade(symbol, signal):
                            last_trade_time[symbol] = time.time()
                            break  # One trade at a time
            
            # Sleep for interval
            time.sleep(SCAN_INTERVAL_MS / 1000)
            
    except KeyboardInterrupt:
        print("\n\n[STOPPED] Trading stopped by user")
        
    finally:
        # Close MT5
        mt5.shutdown()
        print("[DISCONNECTED] MT5 connection closed")

if __name__ == '__main__':
    main()