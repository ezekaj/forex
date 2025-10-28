#!/usr/bin/env python
"""
PRODUCTION FOREX TRADING SYSTEM
Clean, organized, ready for real money
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from core.strategy import RealForexStrategy
from core.backtester import ForexBacktester
from core.live_trader import LiveTrader
from core.data_manager import DataManager
from core.risk_manager import RiskManager
from utils.logger import setup_logger

# Setup logging
logger = setup_logger('main')

def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("PRODUCTION FOREX TRADING SYSTEM")
    print("="*60)
    
    config = load_config()
    
    print("\n1. Backtest Strategy")
    print("2. Paper Trade")
    print("3. Live Trade (REAL MONEY)")
    print("4. System Analysis")
    print("5. Configuration")
    
    choice = input("\nSelect option (1-5): ")
    
    if choice == "1":
        run_backtest(config)
    elif choice == "2":
        run_paper_trading(config)
    elif choice == "3":
        run_live_trading(config)
    elif choice == "4":
        run_analysis(config)
    elif choice == "5":
        configure_system(config)
    else:
        print("Invalid choice")

def load_config():
    """Load system configuration"""
    config_path = Path(__file__).parent / 'config.json'
    
    if not config_path.exists():
        # Create default config
        config = {
            "initial_capital": 1000,
            "risk_per_trade": 0.01,
            "max_daily_loss": 0.05,
            "min_confidence": 0.65,
            "pairs": ["EURUSD", "GBPUSD", "USDJPY"],
            "broker": {
                "type": "demo",
                "api_key": "",
                "account_id": ""
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info("Created default configuration")
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    return config

def run_backtest(config):
    """Run strategy backtest"""
    logger.info("Starting backtest")
    
    # Get data
    data_manager = DataManager()
    df = data_manager.get_historical_data('EURUSD', '4H', 500)
    
    if df is None or df.empty:
        logger.error("No data available for backtest")
        return
    
    # Run backtest
    strategy = RealForexStrategy(min_confidence=config['min_confidence'])
    backtester = ForexBacktester(
        strategy, 
        initial_balance=config['initial_capital']
    )
    
    results = backtester.backtest(df, 'EURUSD')
    
    # Save results
    results_path = Path(__file__).parent / 'results' / f'backtest_{datetime.now():%Y%m%d_%H%M%S}.json'
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    logger.info(f"Backtest complete. Results saved to {results_path}")

def run_paper_trading(config):
    """Run paper trading"""
    logger.info("Starting paper trading")
    
    trader = LiveTrader(
        config=config,
        paper_trade=True
    )
    
    try:
        trader.run()
    except KeyboardInterrupt:
        logger.info("Paper trading stopped by user")
        trader.print_summary()

def run_live_trading(config):
    """Run live trading with real money"""
    logger.warning("LIVE TRADING MODE - REAL MONEY AT RISK")
    
    if not config['broker']['api_key']:
        logger.error("No broker API key configured")
        print("\nTo trade live, you need to:")
        print("1. Get a broker account (OANDA, etc.)")
        print("2. Add API credentials to config.json")
        return
    
    confirm = input("\n⚠️  This will trade with REAL MONEY. Type 'YES' to confirm: ")
    if confirm != 'YES':
        print("Cancelled")
        return
    
    trader = LiveTrader(
        config=config,
        paper_trade=False
    )
    
    try:
        trader.run()
    except KeyboardInterrupt:
        logger.info("Live trading stopped by user")
        trader.print_summary()

def run_analysis(config):
    """Analyze system performance"""
    logger.info("Running system analysis")
    
    # Load recent results
    results_dir = Path(__file__).parent / 'results'
    if not results_dir.exists():
        print("No results to analyze")
        return
    
    # Get all backtest results
    backtest_files = list(results_dir.glob('backtest_*.json'))
    
    if not backtest_files:
        print("No backtest results found")
        return
    
    print(f"\nFound {len(backtest_files)} backtest results")
    
    total_trades = 0
    total_profit = 0
    total_wins = 0
    
    for file in backtest_files:
        with open(file, 'r') as f:
            result = json.load(f)
            total_trades += result.get('total_trades', 0)
            total_profit += result.get('total_profit', 0)
            total_wins += result.get('wins', 0)
    
    if total_trades > 0:
        win_rate = (total_wins / total_trades) * 100
        avg_profit = total_profit / len(backtest_files)
        
        print(f"\nAGGREGATE RESULTS:")
        print(f"Total Backtests: {len(backtest_files)}")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Average Profit: ${avg_profit:.2f}")

def configure_system(config):
    """Configure system settings"""
    print("\nCurrent Configuration:")
    print(json.dumps(config, indent=2))
    
    print("\nWhat would you like to change?")
    print("1. Risk Settings")
    print("2. Trading Pairs")
    print("3. Broker Configuration")
    print("4. Back to Main Menu")
    
    choice = input("\nChoice (1-4): ")
    
    if choice == "1":
        config['risk_per_trade'] = float(input(f"Risk per trade (current: {config['risk_per_trade']}): ") or config['risk_per_trade'])
        config['max_daily_loss'] = float(input(f"Max daily loss (current: {config['max_daily_loss']}): ") or config['max_daily_loss'])
        config['min_confidence'] = float(input(f"Min confidence (current: {config['min_confidence']}): ") or config['min_confidence'])
    
    elif choice == "2":
        pairs = input(f"Trading pairs comma-separated (current: {','.join(config['pairs'])}): ")
        if pairs:
            config['pairs'] = [p.strip() for p in pairs.split(',')]
    
    elif choice == "3":
        config['broker']['api_key'] = input(f"API Key (current: {'SET' if config['broker']['api_key'] else 'NOT SET'}): ") or config['broker']['api_key']
        config['broker']['account_id'] = input(f"Account ID: ") or config['broker']['account_id']
    
    # Save config
    config_path = Path(__file__).parent / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print("\nConfiguration saved!")

if __name__ == "__main__":
    main()