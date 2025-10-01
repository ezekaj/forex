#!/usr/bin/env python
"""
QUICK SETUP - Configure OANDA API
"""

import os

print("\n" + "="*70)
print("OANDA API QUICK SETUP")
print("="*70)
print("\nYou need 2 things from OANDA website:")
print("1. API Token (65 characters long)")
print("2. Account ID (like 101-001-1234567-001)")
print("\nGet them from: https://www.oanda.com/account/login")
print("Then go to: Manage API Access")
print("="*70)

# Get credentials
print("\nPaste your credentials below:\n")
api_token = input("API Token: ").strip()
account_id = input("Account ID: ").strip()

# Validate
if len(api_token) < 50:
    print("\n[ERROR] API token seems too short. Should be ~65 characters")
    print("Make sure you copied the ENTIRE token")
    input("Press Enter to exit...")
    exit()

if '-' not in account_id:
    print("\n[ERROR] Account ID format seems wrong")
    print("Should look like: 101-001-1234567-001")
    input("Press Enter to exit...")
    exit()

# Ask account type
print("\nWhat type of account is this?")
print("1. Practice (demo money)")
print("2. Live (REAL money)")
choice = input("Enter 1 or 2: ")

is_live = "true" if choice == "2" else "false"

if choice == "2":
    print("\n" + "!"*70)
    print("WARNING: This will trade with REAL MONEY!")
    print("!"*70)
    confirm = input("Type 'YES' to confirm: ")
    if confirm != "YES":
        is_live = "false"
        print("Switched to practice mode for safety")

# Write config
config = f"""# OANDA Configuration
OANDA_API_KEY={api_token}
OANDA_ACCOUNT_ID={account_id}
TRADING_ENABLED={is_live}

# Alpha Vantage API for market data
ALPHAVANTAGE_API_KEY=KNF41ZTAUM44W2LN

# Risk Settings
MAX_RISK_PER_TRADE=0.02
MAX_DAILY_LOSS=0.10
TARGET_DAILY_PROFIT=0.05
"""

# Create directory if needed
os.makedirs("BayloZzi", exist_ok=True)

# Write file
with open("BayloZzi/.env", "w") as f:
    f.write(config)

print("\n" + "="*70)
print("SUCCESS! Configuration saved!")
print("="*70)

# Test connection
print("\nTesting connection...")
try:
    from dotenv import load_dotenv
    import requests
    
    load_dotenv('BayloZzi/.env')
    
    # Test API
    base_url = "https://api-fxpractice.oanda.com" if is_live == "false" else "https://api-fxtrade.oanda.com"
    headers = {
        'Authorization': f'Bearer {api_token}',
        'Content-Type': 'application/json'
    }
    
    url = f"{base_url}/v3/accounts/{account_id}"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        account = response.json()['account']
        balance = float(account['balance'])
        currency = account['currency']
        
        print(f"\n[SUCCESS] Connected to OANDA!")
        print(f"Account Type: {'PRACTICE' if is_live == 'false' else 'LIVE'}")
        print(f"Balance: {currency} {balance:,.2f}")
        print(f"Account ID: {account_id}")
        
        print("\n" + "="*70)
        print("READY TO TRADE!")
        print("="*70)
        print("\nTo start trading, run:")
        print("  python start_practice.py    (for practice)")
        print("  python start_live_24_7.py   (for 24/7 trading)")
        print("  python main.py              (for menu)")
        
    else:
        print(f"\n[ERROR] Connection failed: {response.status_code}")
        print("Check your API token and account ID")
        
except Exception as e:
    print(f"\n[ERROR] {e}")
    print("\nTry running: pip install requests python-dotenv")

input("\nPress Enter to exit...")