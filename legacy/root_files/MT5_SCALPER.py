"""
MT5 SCALPER - AGGRESSIVE QUICK TRADES
======================================
Designed to catch small movements quickly
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Configuration
LOGIN = 95948709
PASSWORD = 'To-4KyLg'
SERVER = 'MetaQuotes-Demo'
MAGIC = 999999

# Aggressive scalping parameters
SYMBOLS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']
VOLUME = 0.1  # Fixed small volume for safety
MAX_SPREAD = 2.5  # Allow slightly higher spread
MIN_MOVEMENT = 0.00005  # Very small movement threshold (0.005%)

# Quick profits
TAKE_PROFIT_PIPS = 5   # Quick 5 pip profit
STOP_LOSS_PIPS = 10    # 10 pip stop loss
MAX_POSITIONS = 2      # Max 2 positions

# Timing
CHECK_INTERVAL_MS = 200  # Check every 200ms

class Scalper:
    def __init__(self):
        self.connected = False
        self.positions = {}
        self.last_check = {}
        self.initialize()
    
    def initialize(self):
        """Connect to MT5"""
        if not mt5.initialize():
            print("[ERROR] MT5 init failed")
            return
        
        if not mt5.login(LOGIN, PASSWORD, SERVER):
            print(f"[ERROR] Login failed")
            mt5.shutdown()
            return
        
        account = mt5.account_info()
        if account:
            self.connected = True
            print(f"[CONNECTED] Balance: ${account.balance:.2f}")
    
    def check_positions(self):
        """Count our positions"""
        positions = mt5.positions_get()
        if not positions:
            return 0
        return len([p for p in positions if p.magic == MAGIC])
    
    def look_for_trade(self, symbol):
        """Look for scalping opportunity"""
        
        # Skip if we have max positions
        if self.check_positions() >= MAX_POSITIONS:
            return None
        
        # Get tick
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return None
        
        # Check spread
        info = mt5.symbol_info(symbol)
        spread = (tick.ask - tick.bid) / info.point / 10
        if spread > MAX_SPREAD:
            return None
        
        # Get last 10 bars
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 10)
        if rates is None or len(rates) < 10:
            return None
        
        # Quick analysis
        closes = [r['close'] for r in rates]
        current = closes[-1]
        prev = closes[-2]
        avg_5 = np.mean(closes[-5:])
        
        # Simple momentum detection
        momentum = (current - prev) / prev
        trend = (current - avg_5) / avg_5
        
        # BUY signal
        if momentum > MIN_MOVEMENT and trend > 0:
            return {
                'type': 'BUY',
                'price': tick.ask,
                'momentum': momentum
            }
        
        # SELL signal
        elif momentum < -MIN_MOVEMENT and trend < 0:
            return {
                'type': 'SELL',
                'price': tick.bid,
                'momentum': abs(momentum)
            }
        
        return None
    
    def execute_trade(self, symbol, signal):
        """Execute the trade"""
        
        info = mt5.symbol_info(symbol)
        if not info:
            return False
        
        # Setup trade
        if signal['type'] == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            price = signal['price']
            sl = price - STOP_LOSS_PIPS * info.point * 10
            tp = price + TAKE_PROFIT_PIPS * info.point * 10
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = signal['price']
            sl = price + STOP_LOSS_PIPS * info.point * 10
            tp = price - TAKE_PROFIT_PIPS * info.point * 10
        
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': VOLUME,
            'type': order_type,
            'price': price,
            'sl': round(sl, info.digits),
            'tp': round(tp, info.digits),
            'deviation': 20,
            'magic': MAGIC,
            'comment': 'scalp',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"\n[TRADE] {signal['type']} {symbol} @ {price:.5f}")
            print(f"        Momentum: {signal['momentum']*10000:.1f} | SL: {sl:.5f} | TP: {tp:.5f}")
            return True
        else:
            if result.comment != "Market is closed":
                print(f"\n[FAIL] {symbol}: {result.comment}")
        
        return False
    
    def run(self):
        """Main scalping loop"""
        if not self.connected:
            return
        
        print("\n" + "="*60)
        print("SCALPER ACTIVE - QUICK IN & OUT")
        print("="*60)
        print(f"Targets: {TAKE_PROFIT_PIPS}/{STOP_LOSS_PIPS} pips")
        print(f"Symbols: {', '.join(SYMBOLS)}")
        print("="*60)
        
        cycle = 0
        
        try:
            while True:
                cycle += 1
                
                # Check each symbol
                for symbol in SYMBOLS:
                    signal = self.look_for_trade(symbol)
                    if signal:
                        self.execute_trade(symbol, signal)
                
                # Status update every 5 cycles (1 second)
                if cycle % 5 == 0:
                    account = mt5.account_info()
                    positions = mt5.positions_get()
                    
                    if account:
                        pnl = sum(p.profit for p in positions) if positions else 0
                        print(f"\r[SCAN #{cycle}] Bal: ${account.balance:.2f} | Open: {self.check_positions()} | P&L: ${pnl:.2f}      ", end='')
                
                time.sleep(CHECK_INTERVAL_MS / 1000)
                
        except KeyboardInterrupt:
            print("\n\n[STOPPED]")
        finally:
            mt5.shutdown()

if __name__ == '__main__':
    scalper = Scalper()
    if scalper.connected:
        scalper.run()