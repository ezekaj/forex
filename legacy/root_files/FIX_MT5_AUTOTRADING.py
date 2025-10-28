#!/usr/bin/env python
"""
FIX FOR MT5 BUILD 5233 - AUTOTRADING ISSUE
This script helps diagnose and fix the AutoTrading permission issue
"""

import MetaTrader5 as mt5
import time

print("\n" + "="*70)
print("MT5 AUTOTRADING DIAGNOSTIC & FIX")
print("Build 5233 / Version 5.0")
print("="*70)

# Initialize MT5
if not mt5.initialize():
    print("[ERROR] Cannot initialize MT5")
    print("\nMake sure:")
    print("1. MT5 is installed")
    print("2. MT5 is running")
    exit(1)

print("[OK] MT5 initialized")

# Get terminal info
terminal_info = mt5.terminal_info()
print(f"\n[TERMINAL INFO]")
print(f"Build: {terminal_info.build}")
print(f"Connected: {terminal_info.connected}")
print(f"Trade Allowed: {terminal_info.trade_allowed}")
print(f"Path: {terminal_info.path}")

# Login to account
print(f"\n[ACCOUNT LOGIN]")
authorized = mt5.login(
    login=95948709,
    password="To-4KyLg",
    server="MetaQuotes-Demo"
)

if not authorized:
    print("[ERROR] Login failed")
    print(f"Error: {mt5.last_error()}")
    exit(1)

print("[OK] Logged in successfully")

# Get account info
account_info = mt5.account_info()
print(f"\n[ACCOUNT INFO]")
print(f"Login: {account_info.login}")
print(f"Balance: ${account_info.balance:.2f}")
print(f"Trade Allowed: {account_info.trade_allowed}")
print(f"Trade Expert: {account_info.trade_expert}")
print(f"Trade Mode: {account_info.trade_mode}")

# Check symbol
print(f"\n[SYMBOL CHECK]")
symbol = "EURUSD"
selected = mt5.symbol_select(symbol, True)
if not selected:
    print(f"[ERROR] Cannot select {symbol}")
else:
    print(f"[OK] {symbol} selected")
    
symbol_info = mt5.symbol_info(symbol)
if symbol_info:
    print(f"Symbol: {symbol_info.name}")
    print(f"Spread: {symbol_info.spread}")
    print(f"Trade Mode: {symbol_info.trade_mode}")

# Try different order methods
print(f"\n[TESTING TRADE METHODS]")

# Method 1: Standard order
print("\n1. Testing standard order...")
tick = mt5.symbol_info_tick(symbol)
if tick:
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.01,  # Minimum lot
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick.ask,
        "deviation": 20,
        "magic": 234000,
        "comment": "Test order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # Check the request
    result = mt5.order_check(request)
    if result:
        print(f"Order check result: {result.comment}")
        print(f"Retcode: {result.retcode}")
        
        if result.retcode == mt5.TRADE_RETCODE_DONE or result.retcode == 0:
            print("[OK] Order check passed - trading should work")
        else:
            print(f"[ERROR] {result.comment}")
            
            # Specific error handling
            if result.retcode == 10027:
                print("\n[FIX REQUIRED] AutoTrading is disabled")
                print("\nTo fix this in MT5 Build 5233:")
                print("1. In MT5, press Ctrl+O for Options")
                print("2. Go to 'Expert Advisors' tab")
                print("3. Check these EXACT options:")
                print("   ☑ Allow automated trading")
                print("   ☑ Allow DLL imports")
                print("   ☑ Disable 'Confirm DLL function calls'")
                print("   ☑ Allow WebRequest for listed URL")
                print("4. Click OK")
                print("5. Press Ctrl+E to enable AutoTrading")
                print("6. RESTART MT5 completely")
                print("7. Run this script again")
            elif result.retcode == 10013:
                print("[ERROR] Invalid request")
            elif result.retcode == 10014:
                print("[ERROR] Invalid volume")
            elif result.retcode == 10015:
                print("[ERROR] Invalid price")
            elif result.retcode == 10016:
                print("[ERROR] Invalid stops")
            elif result.retcode == 10030:
                print("[ERROR] Invalid fill type")
                print("[FIX] Trying different fill type...")
                
                # Try different fill types
                fill_types = [
                    mt5.ORDER_FILLING_FOK,
                    mt5.ORDER_FILLING_IOC,
                    mt5.ORDER_FILLING_RETURN
                ]
                
                for fill in fill_types:
                    request["type_filling"] = fill
                    result2 = mt5.order_check(request)
                    if result2 and result2.retcode == 0:
                        print(f"[OK] Fill type {fill} works!")
                        break

# Method 2: Market order
print("\n2. Testing market order...")
request2 = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": 0.01,
    "type": mt5.ORDER_TYPE_BUY,
    "deviation": 20,
    "magic": 234000,
    "comment": "Market test",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_FOK,  # Different fill type
}

result2 = mt5.order_check(request2)
if result2:
    print(f"Market order check: {result2.comment}")
    if result2.retcode == 0:
        print("[OK] Market orders work")

# Final diagnosis
print(f"\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)

if account_info.trade_allowed and account_info.trade_expert:
    print("[✓] Account allows trading")
else:
    print("[✗] Account doesn't allow trading")

if terminal_info.trade_allowed:
    print("[✓] Terminal allows trading")
else:
    print("[✗] Terminal doesn't allow trading")

# Provide solution
print(f"\n[SOLUTION FOR BUILD 5233]")
print("1. CLOSE this Python script")
print("2. In MT5, go to Tools → Options → Expert Advisors")
print("3. Enable ALL checkboxes")
print("4. Click OK")
print("5. Click the AutoTrading button until it's GREEN")
print("6. RESTART MT5 (important!)")
print("7. Run the trading bot again")

print("\nAlternative method:")
print("1. In MT5, click View → Navigator (or Ctrl+N)")
print("2. Expand 'Expert Advisors'")
print("3. Drag any EA to a chart")
print("4. In the popup, check 'Allow live trading'")
print("5. This often fixes the permission issue")

mt5.shutdown()
print("\n[OK] Diagnostic complete")