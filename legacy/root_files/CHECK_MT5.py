#!/usr/bin/env python
"""
CHECK MT5 STATUS
Checks if MetaTrader 5 is properly installed and can connect
"""

import sys
import os

print("\n" + "="*70)
print("METATRADER 5 CONNECTION CHECK")
print("="*70)

# Check if MT5 module is available
try:
    import MetaTrader5 as mt5
    print("[OK] MetaTrader5 Python module installed")
except ImportError:
    print("[ERROR] MetaTrader5 module not found")
    print("Install with: pip install MetaTrader5")
    sys.exit(1)

# Try to initialize MT5
print("\n[1] Attempting to connect to MetaTrader 5...")
print("    NOTE: MetaTrader 5 application must be running!")

if not mt5.initialize():
    print("\n[ERROR] Cannot connect to MetaTrader 5")
    print("\nPossible reasons:")
    print("1. MetaTrader 5 is not installed")
    print("2. MetaTrader 5 is not running") 
    print("3. Python cannot find MT5 installation")
    
    print("\nSOLUTION:")
    print("1. Download MetaTrader 5 from: https://www.metatrader5.com/")
    print("2. Install and open MetaTrader 5")
    print("3. Login with your demo account:")
    print("   Server: MetaQuotes-Demo")
    print("   Login: 95948709")
    print("   Password: To-4KyLg")
    print("4. Then run this script again")
    
    # Show last error if available
    error = mt5.last_error()
    if error:
        print(f"\nMT5 Error: {error}")
    
    sys.exit(1)

print("[OK] Connected to MetaTrader 5!")

# Get terminal info
terminal_info = mt5.terminal_info()
if terminal_info:
    print(f"\nTerminal Info:")
    print(f"  Company: {terminal_info.company}")
    print(f"  Name: {terminal_info.name}")
    print(f"  Path: {terminal_info.path}")
    print(f"  Data Path: {terminal_info.data_path}")
    print(f"  Connected: {terminal_info.connected}")

# Try to login to the account
print(f"\n[2] Logging into demo account...")
account_number = 95948709
password = "To-4KyLg"
server = "MetaQuotes-Demo"

authorized = mt5.login(login=account_number, password=password, server=server)

if authorized:
    print("[OK] Successfully logged in!")
    
    # Get account info
    account_info = mt5.account_info()
    if account_info:
        print(f"\nAccount Info:")
        print(f"  Login: {account_info.login}")
        print(f"  Server: {account_info.server}")
        print(f"  Name: {account_info.name}")
        print(f"  Currency: {account_info.currency}")
        print(f"  Balance: ${account_info.balance:.2f}")
        print(f"  Equity: ${account_info.equity:.2f}")
        print(f"  Leverage: 1:{account_info.leverage}")
        print(f"  Trade Allowed: {account_info.trade_allowed}")
        print(f"  Trade Expert: {account_info.trade_expert}")
    
    # Check available symbols
    print(f"\n[3] Checking available symbols...")
    symbols = mt5.symbols_get()
    if symbols:
        print(f"[OK] {len(symbols)} symbols available")
        
        # Check specific forex pairs
        forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        print("\nChecking major forex pairs:")
        for pair in forex_pairs:
            symbol_info = mt5.symbol_info(pair)
            if symbol_info:
                if symbol_info.visible:
                    tick = mt5.symbol_info_tick(pair)
                    if tick:
                        print(f"  {pair}: Bid={tick.bid:.5f} Ask={tick.ask:.5f} Spread={(tick.ask-tick.bid)*10000:.1f} pips")
                    else:
                        print(f"  {pair}: Available but no price data")
                else:
                    print(f"  {pair}: Not visible (need to add in Market Watch)")
            else:
                print(f"  {pair}: Not found")
    
    print("\n[SUCCESS] Everything is working! You can now:")
    print("1. Run MT5_REAL_TRADER.py for real price trading")
    print("2. Real prices will be used instead of simulated")
    print("3. Your trades will be paper trades but with real market data")
    
else:
    print(f"[ERROR] Failed to login")
    error = mt5.last_error()
    print(f"Error: {error}")
    print("\nCheck your credentials:")
    print(f"  Login: {account_number}")
    print(f"  Password: {password}")
    print(f"  Server: {server}")

# Shutdown connection
mt5.shutdown()
print("\n[OK] Connection closed")
print("="*70)