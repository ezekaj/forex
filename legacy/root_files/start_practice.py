#!/usr/bin/env python
"""
PRACTICE TRADING - Test with $100,000 fake money
Safe way to test the system before using real money
"""

import os
import sys
import time
from datetime import datetime

# Add paths
sys.path.append('01_LIVE_TRADING')
sys.path.append('03_CORE_ENGINE')

print("\n" + "="*70)
print("OANDA PRACTICE TRADING - $100,000 DEMO MONEY")
print("="*70)

def run_practice_trading():
    """Run practice trading session"""
    from dotenv import load_dotenv
    load_dotenv('BayloZzi/.env')
    
    # Check if configured
    api_key = os.getenv('OANDA_API_KEY', '')
    account_id = os.getenv('OANDA_ACCOUNT_ID', '')
    
    if not api_key or not account_id:
        print("\n[ERROR] OANDA credentials not found!")
        print("\nTo set up:")
        print("1. Create practice account at: https://www.oanda.com")
        print("2. Get your API token and account ID")
        print("3. Run: setup_practice.bat")
        return
    
    print(f"\n[OK] Account configured: {account_id[:10]}...")
    print("[OK] Using PRACTICE mode (fake money)")
    print("\nStarting capital: $100,000 (practice money)")
    
    # Import the enhanced trader
    from REAL_TRADER import AutomaticForexTrader
    
    # Create trader with practice capital
    trader = AutomaticForexTrader(initial_capital=100000)
    
    if trader.broker.connected:
        print("\n[SUCCESS] Connected to OANDA Practice Account!")
        
        # Get account info
        print("\n[ACCOUNT INFO]")
        print(f"  Mode: PRACTICE (Demo)")
        print(f"  Capital: $100,000 practice money")
        print(f"  Leverage: 50:1")
        print(f"  Pairs: EUR/USD, GBP/USD, USD/JPY, AUD/USD")
    else:
        print("\n[WARNING] Not connected - will run in simulation")
    
    print("\n" + "-"*70)
    print("TRADING OPTIONS:")
    print("-"*70)
    print("1. Quick Test (1 analysis)")
    print("2. 1 Hour Session")
    print("3. 24 Hour Session")
    print("4. Continuous (24/7)")
    
    choice = input("\nSelect option (1-4): ")
    
    if choice == "1":
        # Quick test
        print("\n[QUICK TEST] Analyzing markets...")
        for pair in ['EUR_USD', 'GBP_USD']:
            print(f"\nAnalyzing {pair}...")
            trader.analyze_and_trade(pair)
            time.sleep(2)
    
    elif choice == "2":
        # 1 hour session
        print("\n[1 HOUR SESSION] Starting...")
        trader.run_automatic_trading(hours=1, interval_minutes=5)
    
    elif choice == "3":
        # 24 hour session
        print("\n[24 HOUR SESSION] Starting...")
        trader.run_automatic_trading(hours=24, interval_minutes=5)
    
    elif choice == "4":
        # Continuous
        print("\n[24/7 MODE] Starting continuous trading...")
        print("[INFO] Press Ctrl+C to stop")
        trader.run_automatic_trading(hours=24*365, interval_minutes=5)
    
    else:
        print("Invalid choice")
        return
    
    # Show final results
    print("\n" + "="*70)
    print("PRACTICE SESSION COMPLETE")
    print("="*70)
    
    if trader.trades_executed:
        print(f"Total Trades: {len(trader.trades_executed)}")
        # Calculate profit/loss
        starting = 100000
        ending = trader.capital
        profit = ending - starting
        profit_pct = (profit / starting) * 100
        
        print(f"Starting Capital: ${starting:,.2f}")
        print(f"Ending Capital: ${ending:,.2f}")
        print(f"Profit/Loss: ${profit:+,.2f} ({profit_pct:+.2f}%)")
        
        if profit > 0:
            print("\n[GOOD] Profitable in practice! Consider:")
            print("  1. Run longer tests (24+ hours)")
            print("  2. Test for at least 1 week")
            print("  3. If consistently profitable, try small real money")
        else:
            print("\n[LEARNING] Lost money in practice - that's OK!")
            print("  1. Review the trades in detail")
            print("  2. Adjust risk settings")
            print("  3. Keep practicing until profitable")

def main():
    try:
        run_practice_trading()
    except KeyboardInterrupt:
        print("\n\n[STOPPED] Practice session ended")
    except Exception as e:
        print(f"\n[ERROR] {e}")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()