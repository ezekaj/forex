#!/usr/bin/env python
"""
FOREX MASTER CONTROL - Single command for everything
Usage: python forex.py [mode] [options]

Modes:
  turbo    - Maximum profit mode (50% daily target)
  safe     - Conservative trading (10% daily target)  
  scalp    - 1-minute scalping (100+ trades/day)
  demo     - Test with fake money
  analyze  - Analyze market without trading
  status   - Check system status
"""

import os
import sys
import time
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add BayloZzi to path
sys.path.append('../BayloZzi')
sys.path.append('../BayloZzi/core')
sys.path.append('../03_CORE_ENGINE')

# ASCII Art Banner (no unicode)
BANNER = """
============================================================
         FOREX TRADING SYSTEM - MASTER CONTROL
============================================================
         Capital: EUR {} | Mode: {} | Target: {}%
============================================================
"""

class ForexMaster:
    """Single control point for all trading operations"""
    
    def __init__(self):
        self.capital = 10.0
        self.mode = 'demo'
        self.target = 0.50
        self.running = False
        
    def print_menu(self):
        """Print simple menu"""
        print("\n" + "="*60)
        print("FOREX MASTER CONTROL")
        print("="*60)
        print("1. TURBO MODE    - Aggressive trading (50% daily target)")
        print("2. SAFE MODE     - Conservative (10% daily target)")
        print("3. SCALP MODE    - 1-minute scalping")
        print("4. DEMO MODE     - Test with fake money")
        print("5. ANALYZE       - Market analysis only")
        print("6. STATUS        - Check account status")
        print("7. SETTINGS      - Configure parameters")
        print("8. START         - Begin trading")
        print("9. EXIT          - Quit")
        print("="*60)
        
    def run_interactive(self):
        """Interactive mode with menu"""
        print(BANNER.format(self.capital, self.mode.upper(), int(self.target*100)))
        
        while True:
            self.print_menu()
            choice = input("\nSelect option (1-9): ").strip()
            
            if choice == '1':
                self.mode = 'turbo'
                self.target = 0.50
                print("[OK] TURBO MODE activated - 50% daily target")
                
            elif choice == '2':
                self.mode = 'safe'
                self.target = 0.10
                print("[OK] SAFE MODE activated - 10% daily target")
                
            elif choice == '3':
                self.mode = 'scalp'
                self.target = 0.30
                print("[OK] SCALP MODE activated - 30% daily target")
                
            elif choice == '4':
                self.mode = 'demo'
                print("[OK] DEMO MODE activated - No real money")
                
            elif choice == '5':
                self.run_analysis()
                
            elif choice == '6':
                self.show_status()
                
            elif choice == '7':
                self.configure_settings()
                
            elif choice == '8':
                self.start_trading()
                
            elif choice == '9':
                print("\n[EXIT] Shutting down...")
                break
                
            else:
                print("[ERROR] Invalid option")
                
    def configure_settings(self):
        """Configure trading parameters"""
        print("\n" + "="*40)
        print("SETTINGS")
        print("="*40)
        
        # Capital
        capital_input = input(f"Starting capital (current: EUR {self.capital}): ").strip()
        if capital_input:
            try:
                self.capital = float(capital_input)
                print(f"[OK] Capital set to EUR {self.capital}")
            except:
                print("[ERROR] Invalid amount")
                
        # Target
        target_input = input(f"Daily target % (current: {self.target*100:.0f}%): ").strip()
        if target_input:
            try:
                self.target = float(target_input) / 100
                print(f"[OK] Target set to {self.target*100:.0f}%")
            except:
                print("[ERROR] Invalid target")
                
    def show_status(self):
        """Show current status"""
        print("\n" + "="*40)
        print("SYSTEM STATUS")
        print("="*40)
        
        # Check API key
        api_key = os.getenv("ALPHAVANTAGE_API_KEY", "")
        if api_key and len(api_key) > 10:
            print(f"[OK] Alpha Vantage API configured")
        else:
            print(f"[ERROR] No API key found")
            
        # Check broker
        trading_enabled = os.getenv("TRADING_ENABLED", "false")
        print(f"Trading Enabled: {trading_enabled}")
        
        # Current settings
        print(f"Mode: {self.mode.upper()}")
        print(f"Capital: EUR {self.capital:.2f}")
        print(f"Daily Target: {self.target*100:.0f}%")
        print(f"Expected Profit: EUR {self.capital * self.target:.2f}")
        
        # Risk settings
        print(f"\nRisk Parameters:")
        print(f"  Risk per trade: 20%" if self.mode == 'turbo' else f"  Risk per trade: 2%")
        print(f"  Max daily loss: 50%" if self.mode == 'turbo' else f"  Max daily loss: 10%")
        print(f"  Leverage: 1:500" if self.mode == 'turbo' else f"  Leverage: 1:100")
        
    def run_analysis(self):
        """Run market analysis"""
        print("\n[ANALYZING] Fetching market data...")
        
        try:
            from data_loader import download_alpha_fx_daily
            from pattern_torrent import PatternTorrent
            
            # Get data
            df = download_alpha_fx_daily()
            print(f"[OK] Data loaded: {len(df)} days")
            
            # Detect patterns
            detector = PatternTorrent()
            patterns = detector.detect_all_patterns(df)
            
            print(f"\n[PATTERNS] Found {len(patterns)} patterns:")
            for name, info in list(patterns.items())[:5]:
                print(f"  - {name}: {info.get('direction', 'neutral')} ({info.get('confidence', 0)*100:.0f}%)")
                
            # Current price
            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2]
            change = (current_price - prev_price) / prev_price * 100
            
            print(f"\n[PRICE] EUR/USD: {current_price:.4f} ({change:+.2f}%)")
            
            # Recommendation
            if len(patterns) > 5:
                print(f"\n[SIGNAL] Multiple patterns detected - Good trading opportunity")
            else:
                print(f"\n[SIGNAL] Few patterns - Wait for better setup")
                
        except Exception as e:
            print(f"[ERROR] Analysis failed: {e}")
            
    def start_trading(self):
        """Start trading based on selected mode"""
        print("\n" + "="*60)
        print(f"STARTING {self.mode.upper()} MODE")
        print("="*60)
        print(f"Capital: EUR {self.capital:.2f}")
        print(f"Target: {self.target*100:.0f}% daily")
        print("\nPress Ctrl+C to stop\n")
        
        if self.mode == 'turbo':
            self.run_turbo_mode()
        elif self.mode == 'safe':
            self.run_safe_mode()
        elif self.mode == 'scalp':
            self.run_scalp_mode()
        elif self.mode == 'demo':
            self.run_demo_mode()
        else:
            print("[ERROR] Invalid mode")
            
    def run_turbo_mode(self):
        """Run turbo mode - aggressive trading"""
        try:
            # Use the simplified turbo engine
            # Import TurboEngine from 03_CORE_ENGINE
            from turbo_engine import TurboEngine
            
            engine = TurboEngine(capital=self.capital)
            engine.run_quick_mode()
            
        except KeyboardInterrupt:
            print("\n[STOPPED] Turbo mode stopped by user")
        except Exception as e:
            print(f"[ERROR] Turbo mode failed: {e}")
            
    def run_safe_mode(self):
        """Run safe mode - conservative trading"""
        print("[SAFE MODE] Conservative trading...")
        
        capital = self.capital
        trades = 0
        wins = 0
        
        for i in range(50):  # Fewer trades in safe mode
            # Simulate conservative trading
            if np.random.random() < 0.55:  # 55% win rate
                profit = capital * 0.002  # 0.2% per trade
                wins += 1
            else:
                profit = -capital * 0.001  # 0.1% loss
                
            capital += profit
            trades += 1
            
            if trades % 10 == 0:
                returns = (capital - self.capital) / self.capital
                print(f"  [{trades} trades] Capital: EUR {capital:.2f} ({returns*100:+.1f}%)")
                
            if capital >= self.capital * (1 + self.target):
                print(f"\n[SUCCESS] Target reached! EUR {capital:.2f}")
                break
                
        print(f"\n[COMPLETE] Final capital: EUR {capital:.2f}")
        
    def run_scalp_mode(self):
        """Run scalping mode - high frequency"""
        print("[SCALP MODE] 1-minute scalping...")
        
        from micro_scalper import MicroScalper
        
        scalper = MicroScalper()
        capital = self.capital
        
        for i in range(100):  # 100 quick trades
            result = scalper.execute_micro_trade(
                direction=np.random.choice(['BUY', 'SELL']),
                confidence=np.random.uniform(0.5, 0.7),
                position_size=0.01,
                leverage=500
            )
            
            capital += result['pnl']
            
            if (i+1) % 20 == 0:
                stats = scalper.get_performance_stats()
                returns = (capital - self.capital) / self.capital
                print(f"  [{i+1} trades] Capital: EUR {capital:.2f} ({returns*100:+.1f}%) | Win rate: {stats['win_rate']*100:.0f}%")
                
        print(f"\n[COMPLETE] Final capital: EUR {capital:.2f}")
        
    def run_demo_mode(self):
        """Run demo mode - no real money"""
        print("[DEMO MODE] Testing with fake money...")
        
        # Run existing demo
        try:
            os.system("cd BayloZzi && python run/demo_trade.py")
        except:
            # Fallback demo
            self.run_safe_mode()
            
    def run_command_line(self, args):
        """Run from command line arguments"""
        if args.mode == 'menu':
            self.run_interactive()
            return
            
        # Set parameters from args
        self.capital = args.capital
        self.mode = args.mode
        
        # Set target based on mode
        targets = {
            'turbo': 0.50,
            'safe': 0.10,
            'scalp': 0.30,
            'demo': 0.20,
            'analyze': 0
        }
        self.target = targets.get(args.mode, 0.20)
        
        # Override target if specified
        if args.target:
            self.target = args.target
            
        print(BANNER.format(self.capital, self.mode.upper(), int(self.target*100)))
        
        if args.mode == 'analyze':
            self.run_analysis()
        elif args.mode == 'status':
            self.show_status()
        else:
            self.start_trading()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Forex Trading System - Master Control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python forex.py                    # Interactive menu
  python forex.py turbo              # Turbo mode with defaults
  python forex.py turbo -c 50        # Turbo with EUR 50
  python forex.py safe -c 100        # Safe mode with EUR 100
  python forex.py scalp               # Scalping mode
  python forex.py demo                # Demo mode
  python forex.py analyze             # Market analysis
  python forex.py status              # Check status
        """
    )
    
    parser.add_argument('mode', 
                       nargs='?',
                       default='menu',
                       choices=['menu', 'turbo', 'safe', 'scalp', 'demo', 'analyze', 'status'],
                       help='Trading mode or action')
    
    parser.add_argument('-c', '--capital',
                       type=float,
                       default=10.0,
                       help='Starting capital in EUR (default: 10)')
    
    parser.add_argument('-t', '--target',
                       type=float,
                       help='Daily target as decimal (0.5 = 50%%)')
    
    args = parser.parse_args()
    
    # Create and run master controller
    master = ForexMaster()
    master.run_command_line(args)


if __name__ == "__main__":
    main()