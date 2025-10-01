#!/usr/bin/env python
"""
MT5 CONNECTOR - Use your existing MetaTrader 5
No new broker account needed!
"""

import sys
import time
import subprocess
from datetime import datetime

print("\n" + "="*70)
print("MT5 INTEGRATION - USE YOUR EXISTING ACCOUNT")
print("="*70)

def setup_mt5():
    """Setup MT5 for Python trading"""
    print("\n[SETUP] Installing MT5 Python package...")
    
    # Install MetaTrader5 package
    subprocess.run([sys.executable, "-m", "pip", "install", "MetaTrader5", "pandas", "numpy"])
    
    print("\n[OK] MT5 package installed!")
    
    # Test connection
    try:
        import MetaTrader5 as mt5
        
        # Initialize MT5
        if not mt5.initialize():
            print("[ERROR] MT5 not found. Make sure MT5 is running!")
            return False
        
        # Get account info
        account_info = mt5.account_info()
        if account_info:
            print(f"\n[SUCCESS] Connected to MT5!")
            print(f"  Account: {account_info.login}")
            print(f"  Balance: {account_info.currency} {account_info.balance:.2f}")
            print(f"  Server: {account_info.server}")
            
            # Get symbol info
            symbols = mt5.symbols_get()
            forex_pairs = [s.name for s in symbols if 'USD' in s.name][:5]
            print(f"  Available Pairs: {', '.join(forex_pairs)}")
        
        mt5.shutdown()
        return True
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def create_mt5_trader():
    """Create MT5 trading script"""
    
    script = '''#!/usr/bin/env python
"""
MT5 AUTOMATED TRADER
Trades using your MetaTrader 5 account
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import sys

sys.path.append('03_CORE_ENGINE')

# Import enhanced components
from smart_position_sizer import SmartPositionSizer
from win_rate_optimizer import WinRateOptimizer
from market_timing_system import MarketTimingSystem

class MT5Trader:
    def __init__(self):
        if not mt5.initialize():
            print("[ERROR] Failed to initialize MT5")
            quit()
        
        self.account = mt5.account_info()
        self.position_sizer = SmartPositionSizer()
        self.win_optimizer = WinRateOptimizer()
        self.timing_system = MarketTimingSystem()
        
        print(f"[MT5] Connected: {self.account.login}")
        print(f"[MT5] Balance: {self.account.balance:.2f}")
    
    def get_data(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, bars=100):
        """Get market data from MT5"""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    def analyze(self, symbol):
        """Analyze and generate signals"""
        df = self.get_data(symbol)
        
        # Calculate RSI
        close = df['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Generate signal
        if rsi > 70:
            return 'SELL', rsi
        elif rsi < 30:
            return 'BUY', rsi
        else:
            return 'HOLD', rsi
    
    def execute_trade(self, symbol, signal, volume=0.01):
        """Execute trade on MT5"""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"[ERROR] {symbol} not found")
            return
        
        price = mt5.symbol_info_tick(symbol).ask if signal == 'BUY' else mt5.symbol_info_tick(symbol).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if signal == 'BUY' else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": price - 0.0020 if signal == 'BUY' else price + 0.0020,
            "tp": price + 0.0040 if signal == 'BUY' else price - 0.0040,
            "deviation": 10,
            "magic": 234000,
            "comment": "Python AI Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"[FAILED] {result.comment}")
        else:
            print(f"[SUCCESS] {signal} {volume} {symbol} at {price}")
        
        return result
    
    def run(self):
        """Run trading loop"""
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        
        while True:
            for symbol in symbols:
                signal, rsi = self.analyze(symbol)
                print(f"{symbol}: RSI={rsi:.1f} Signal={signal}")
                
                if signal != 'HOLD':
                    # Check market timing
                    should_trade, reason = self.timing_system.should_trade_now(symbol)
                    if should_trade:
                        self.execute_trade(symbol, signal)
            
            time.sleep(300)  # Wait 5 minutes

if __name__ == "__main__":
    trader = MT5Trader()
    trader.run()
'''
    
    with open("mt5_trader.py", "w") as f:
        f.write(script)
    
    print("\n[CREATED] mt5_trader.py")
    print("\nTo run: python mt5_trader.py")

def main():
    print("\nThis will set up your existing MT5 for automated trading")
    print("No new broker account needed!\n")
    
    print("Options:")
    print("1. Setup MT5 Connection")
    print("2. Create MT5 Trading Script")
    print("3. Both")
    
    choice = input("\nChoice (1-3): ")
    
    if choice in ["1", "3"]:
        setup_mt5()
    
    if choice in ["2", "3"]:
        create_mt5_trader()
    
    print("\n[COMPLETE] You can now trade with your MT5 account!")
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()