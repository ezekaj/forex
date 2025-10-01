"""
Test your Alpha Vantage connection and system setup
"""

import os
import sys
from dotenv import load_dotenv

# No unicode for Windows console

# Add BayloZzi to path
sys.path.append('BayloZzi')

# Load environment variables
load_dotenv()

print("="*60)
print("FOREX TRADING SYSTEM - SETUP TEST")
print("="*60)

# Test 1: Check API Key
api_key = os.getenv("ALPHAVANTAGE_API_KEY")
if api_key and api_key != "your_alpha_vantage_key_here":
    print("[OK] Alpha Vantage API Key configured")
    print(f"   Key: {api_key[:8]}...")
else:
    print("[ERROR] Alpha Vantage API Key not set!")
    
# Test 2: Try to fetch data
print("\n[TEST] Testing market data download...")
try:
    from BayloZzi.core.data_loader import download_alpha_fx_daily
    
    # This will download EUR/USD daily data
    df = download_alpha_fx_daily()
    
    if not df.empty:
        print("[OK] Market data download successful!")
        print(f"   Latest data: {df.index[-1]}")
        print(f"   EUR/USD Price: {df['close'].iloc[-1]:.4f}")
        print(f"   Total days of data: {len(df)}")
    else:
        print("[ERROR] No data received")
        
except Exception as e:
    print(f"[ERROR] Data download failed: {e}")
    print("   Check your API key is correct")

# Test 3: Check broker configuration
print("\n[TEST] Checking broker configuration...")

# Check for OANDA
oanda_key = os.getenv("OANDA_API_KEY")
if oanda_key and oanda_key != "your_oanda_api_key":
    print("[OK] OANDA configured")
else:
    print("[WARNING] OANDA not configured (optional)")

# Check for MT5
mt5_login = os.getenv("MT5_LOGIN")
if mt5_login and mt5_login != "your_account_number":
    print("[OK] MT5 configured")
else:
    print("[WARNING] MT5 not configured (optional)")

# Test 4: Check trading settings
print("\n[SETTINGS] Trading Settings:")
print(f"   Trading Enabled: {os.getenv('TRADING_ENABLED', 'false')}")
print(f"   Max Risk: {os.getenv('MAX_RISK_PER_TRADE', '0.01')} (1% default)")
print(f"   Max Drawdown: {os.getenv('MAX_DRAWDOWN', '0.10')} (10% default)")

# Test 5: Run demo trade simulation
print("\n[DEMO] Want to run a demo trade? (no real money)")
response = input("Type 'yes' to run demo or press Enter to skip: ")

if response.lower() == 'yes':
    print("\nStarting demo trading...")
    os.system("cd BayloZzi && python run/live_trade_safe.py --broker demo")
else:
    print("\n[OK] Setup test complete!")
    print("\nNext steps:")
    print("1. Sign up with a broker (OANDA, XM, or Exness)")
    print("2. Add broker credentials to .env file")
    print("3. Run: cd BayloZzi && python run/live_trade_safe.py --broker demo")
    print("4. When ready, set TRADING_ENABLED=true for real trading")