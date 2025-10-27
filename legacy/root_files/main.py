#!/usr/bin/env python
"""
MAIN LAUNCHER - Forex Trading System Control Center
"""

import os
import sys
import subprocess
from datetime import datetime

def clear_screen():
    """Clear console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print system header"""
    print("="*60)
    print("         FOREX TRADING SYSTEM - CONTROL CENTER")
    print("="*60)
    print(f"         Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)

def print_menu():
    """Print main menu"""
    print("\nMAIN MENU:")
    print("-"*40)
    print("1. TEST SYSTEM     - Check real market data")
    print("2. REAL TRADER     - Start automatic trading")
    print("3. ELITE AI        - Run kimi_k2 engine")
    print("4. TURBO MODE      - Aggressive trading (50% target)")
    print("5. SAFE MODE       - Conservative (10% target)")
    print("6. MULTI-STRATEGY  - Renaissance engine")
    print("7. SYSTEM STATUS   - Check configuration")
    print("8. SETUP HELP      - Installation guide")
    print("9. EXIT            - Close system")
    print("-"*40)

def test_system():
    """Run system test"""
    print("\n[TEST] Running system test with real data...")
    subprocess.run([sys.executable, "07_TESTING/QUICK_TEST.py"])

def start_real_trader():
    """Start real trader"""
    print("\n[TRADER] Starting automatic trading...")
    hours = input("How many hours to trade? (default=1): ").strip() or "1"
    subprocess.run([sys.executable, "01_LIVE_TRADING/REAL_TRADER.py", "--hours", hours])

def start_elite_ai():
    """Start elite AI system"""
    print("\n[AI] Starting Elite AI trading engine...")
    subprocess.run([sys.executable, "02_ELITE_SYSTEMS/kimi_k2.py", "--mode", "trade"])

def start_turbo():
    """Start turbo mode"""
    print("\n[TURBO] Starting aggressive trading mode...")
    subprocess.run([sys.executable, "01_LIVE_TRADING/forex.py", "turbo"])

def start_safe():
    """Start safe mode"""
    print("\n[SAFE] Starting conservative trading mode...")
    subprocess.run([sys.executable, "01_LIVE_TRADING/forex.py", "safe"])

def start_renaissance():
    """Start Renaissance multi-strategy engine"""
    print("\n[RENAISSANCE] Starting multi-strategy portfolio...")
    subprocess.run([sys.executable, "02_ELITE_SYSTEMS/renaissance_engine.py"])

def check_status():
    """Check system status"""
    print("\n[STATUS] Checking system configuration...")
    print("-"*40)
    
    # Check Alpha Vantage
    env_path = "BayloZzi/.env"
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            content = f.read()
            
        if "KNF41ZTAUM44W2LN" in content:
            print("[OK] Alpha Vantage API: Configured")
        else:
            print("[X] Alpha Vantage API: Not configured")
            
        if "OANDA_API_KEY=" in content and len(content.split("OANDA_API_KEY=")[1].split("\n")[0].strip()) > 0:
            print("[OK] OANDA Broker: Configured")
        else:
            print("[X] OANDA Broker: Not configured")
            
        if "TRADING_ENABLED=true" in content.lower():
            print("[OK] Live Trading: ENABLED")
        else:
            print("[!] Live Trading: DISABLED (demo mode)")
    else:
        print("[X] Configuration file not found")
    
    # Check for required files
    print("\nCore Files:")
    files_to_check = [
        ("01_LIVE_TRADING/REAL_TRADER.py", "Real Trader"),
        ("02_ELITE_SYSTEMS/kimi_k2.py", "Elite AI Engine"),
        ("03_CORE_ENGINE/turbo_engine.py", "Turbo Engine"),
        ("04_DATA/eurusd_daily_alpha.csv", "Market Data")
    ]
    
    for file_path, name in files_to_check:
        if os.path.exists(file_path):
            print(f"[OK] {name}: Found")
        else:
            print(f"[X] {name}: Missing")

def show_setup():
    """Show setup instructions"""
    print("\n" + "="*60)
    print("SETUP INSTRUCTIONS")
    print("="*60)
    print("\n1. INSTALL REQUIREMENTS:")
    print("   pip install numpy pandas torch")
    print("   pip install TA-Lib textblob")
    print("\n2. GET OANDA ACCOUNT:")
    print("   - Go to: https://www.oanda.com")
    print("   - Create Practice Account (free)")
    print("   - Get API key from account settings")
    print("\n3. CONFIGURE SYSTEM:")
    print("   - Edit: BayloZzi/.env")
    print("   - Add: OANDA_API_KEY=your_key")
    print("   - Add: OANDA_ACCOUNT_ID=your_id")
    print("   - Set: TRADING_ENABLED=true (for live)")
    print("\n4. START TRADING:")
    print("   - Run: python main.py")
    print("   - Choose option 2 for real trading")
    print("="*60)

def main():
    """Main launcher"""
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == "1":
            test_system()
        elif choice == "2":
            start_real_trader()
        elif choice == "3":
            start_elite_ai()
        elif choice == "4":
            start_turbo()
        elif choice == "5":
            start_safe()
        elif choice == "6":
            start_renaissance()
        elif choice == "7":
            check_status()
        elif choice == "8":
            show_setup()
        elif choice == "9":
            print("\n[EXIT] Shutting down system...")
            break
        else:
            print("\n[ERROR] Invalid choice!")
        
        if choice != "9":
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[STOPPED] System terminated by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        input("Press Enter to exit...")