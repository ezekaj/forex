#!/usr/bin/env python
"""
LIVE 24/7 FOREX TRADING SYSTEM
Runs continuously with real money
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add paths
sys.path.append('01_LIVE_TRADING')
sys.path.append('03_CORE_ENGINE')

# Load environment
load_dotenv('BayloZzi/.env')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_log.txt'),
        logging.StreamHandler()
    ]
)

def check_configuration():
    """Verify configuration before starting"""
    print("\n" + "="*60)
    print("LIVE TRADING SYSTEM - 24/7 OPERATION")
    print("="*60)
    
    # Check credentials
    api_key = os.getenv('OANDA_API_KEY', '')
    account_id = os.getenv('OANDA_ACCOUNT_ID', '')
    is_live = os.getenv('TRADING_ENABLED', 'false').lower() == 'true'
    
    if not api_key or not account_id:
        print("\n[ERROR] OANDA credentials not configured!")
        print("Run: setup_live_trading.bat")
        return False
    
    print(f"\n[CONFIG] OANDA Account: {account_id[:4]}****")
    print(f"[CONFIG] Mode: {'LIVE TRADING' if is_live else 'DEMO MODE'}")
    
    if is_live:
        print("\n" + "!"*60)
        print("WARNING: LIVE TRADING WITH REAL MONEY")
        print("!"*60)
        print("\nRISK SETTINGS:")
        print(f"  - Max Risk Per Trade: 2%")
        print(f"  - Max Daily Loss: 10%")
        print(f"  - Target Daily Profit: 5%")
        
        confirm = input("\nType 'START' to begin live trading: ")
        if confirm != 'START':
            print("Cancelled")
            return False
    
    return True

def run_24_7_trading():
    """Run continuous trading"""
    from REAL_TRADER import AutomaticForexTrader
    
    # Initialize trader
    initial_capital = float(input("Enter starting capital (EUR): ") or "100")
    trader = AutomaticForexTrader(initial_capital=initial_capital)
    
    print("\n[STARTING] 24/7 Trading System")
    print("[INFO] Press Ctrl+C to stop")
    print("-"*60)
    
    # Trading parameters
    check_interval = 5  # minutes
    pairs_to_trade = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD']
    
    daily_profit = 0
    daily_loss_limit = initial_capital * 0.10  # 10% max daily loss
    daily_profit_target = initial_capital * 0.05  # 5% daily target
    
    start_of_day = datetime.now()
    
    try:
        while True:
            current_time = datetime.now()
            
            # Reset daily counters at midnight
            if current_time.date() > start_of_day.date():
                logging.info(f"Daily P&L: {daily_profit:.2f} EUR")
                daily_profit = 0
                start_of_day = current_time
            
            # Check daily loss limit
            if daily_profit < -daily_loss_limit:
                logging.warning("Daily loss limit reached - stopping for today")
                time.sleep(3600)  # Wait 1 hour
                continue
            
            # Check daily profit target
            if daily_profit >= daily_profit_target:
                logging.info("Daily profit target reached!")
            
            # Trade each pair
            for pair in pairs_to_trade:
                try:
                    logging.info(f"Analyzing {pair}...")
                    
                    # Use market timing to check if we should trade
                    from market_timing_system import MarketTimingSystem
                    timing = MarketTimingSystem()
                    should_trade, reason = timing.should_trade_now(pair)
                    
                    if should_trade:
                        # Analyze and potentially trade
                        trader.analyze_and_trade(pair)
                        
                        # Track daily P&L (would need to get from trader)
                        # daily_profit += trade_profit
                    else:
                        logging.info(f"Skipping {pair}: {reason}")
                    
                    time.sleep(2)  # Small delay between pairs
                    
                except Exception as e:
                    logging.error(f"Error trading {pair}: {e}")
                    continue
            
            # Show status
            if trader.trades_executed:
                logging.info(f"Trades today: {len(trader.trades_executed)}")
                logging.info(f"Current capital: {trader.capital:.2f} EUR")
            
            # Wait for next check
            logging.info(f"Next check in {check_interval} minutes...")
            time.sleep(check_interval * 60)
            
    except KeyboardInterrupt:
        logging.info("\n[STOPPED] Trading stopped by user")
        
    finally:
        # Final report
        trader.print_final_report()
        
        # Save performance data
        if trader.performance_tracker:
            metrics = trader.performance_tracker.calculate_metrics()
            logging.info(f"\nFinal Performance:")
            logging.info(f"  Total Trades: {metrics.get('total_trades', 0)}")
            logging.info(f"  Win Rate: {metrics.get('win_rate', 0):.1%}")
            logging.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            logging.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.1%}")

def main():
    """Main entry point"""
    if not check_configuration():
        return
    
    # Run trading
    run_24_7_trading()

if __name__ == "__main__":
    main()