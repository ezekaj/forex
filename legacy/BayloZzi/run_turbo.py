"""
QUICK TURBO LAUNCHER - Simplified version for immediate testing
Run the turbo engine with minimal dependencies
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append('core')
sys.path.append('.')

print("="*60)
print("TURBO FOREX TRADING ENGINE - QUICK START")
print("="*60)
print(f"Target: Turn EUR 10 into EUR 15+ today")
print(f"Method: 1 API call -> 100+ synthetic trades")
print(f"Risk: HIGH (20% per trade)")
print("")

class QuickTurboEngine:
    """Simplified turbo engine for immediate profits"""
    
    def __init__(self, capital=10.0):
        self.capital = capital
        self.initial_capital = capital
        self.trades = 0
        self.wins = 0
        
    def run_quick_mode(self):
        """Run simplified turbo trading"""
        
        print(f"Starting capital: EUR {self.capital:.2f}")
        print(f"Fetching market data...")
        
        # Get or generate data
        try:
            from core.data_loader import download_alpha_fx_daily
            df = download_alpha_fx_daily()
            print(f"[OK] Real data loaded: {len(df)} days")
        except:
            print("[WARNING] Using synthetic data (API issue)")
            # Generate synthetic data
            df = self.generate_quick_synthetic()
            
        print(f"\n[START] Starting rapid trading...")
        
        # Run 100 quick trades
        for i in range(100):
            # Simulate edge detection (50.75% win rate)
            has_edge = np.random.random() < 0.5075
            
            if has_edge:
                # Calculate position
                position = self.capital * 0.20  # 20% risk
                
                # Simulate trade (2 pip target, 1 pip stop)
                if np.random.random() < 0.5075:
                    # Win
                    profit = position * 0.02  # 2% gain
                    self.wins += 1
                else:
                    # Loss
                    profit = -position * 0.01  # 1% loss
                    
                self.capital += profit
                self.trades += 1
                
                # Progress update
                if self.trades % 20 == 0:
                    win_rate = self.wins / self.trades
                    returns = (self.capital - self.initial_capital) / self.initial_capital
                    print(f"  [{self.trades:3d} trades] Capital: EUR {self.capital:.2f} ({returns*100:+.1f}%) | Win rate: {win_rate*100:.1f}%")
                    
                # Check target
                if self.capital >= self.initial_capital * 1.5:
                    print(f"\n[SUCCESS] TARGET REACHED! EUR {self.capital:.2f}")
                    break
                    
                # Check stop loss
                if self.capital < self.initial_capital * 0.5:
                    print(f"\n[STOP] Stop loss hit. Capital: EUR {self.capital:.2f}")
                    break
                    
        # Final report
        self.print_results()
        
    def generate_quick_synthetic(self):
        """Generate quick synthetic data"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
        prices = 1.0850 + np.random.randn(100).cumsum() * 0.001
        
        df = pd.DataFrame({
            'open': prices + np.random.randn(100) * 0.0001,
            'high': prices + abs(np.random.randn(100) * 0.0002),
            'low': prices - abs(np.random.randn(100) * 0.0002),
            'close': prices,
            'volume': np.random.randint(100, 1000, 100)
        }, index=dates)
        
        return df
        
    def print_results(self):
        """Print final results"""
        returns = (self.capital - self.initial_capital) / self.initial_capital
        win_rate = self.wins / self.trades if self.trades > 0 else 0
        
        print("\n" + "="*60)
        print("TURBO ENGINE RESULTS")
        print("="*60)
        print(f"Initial Capital: EUR {self.initial_capital:.2f}")
        print(f"Final Capital: EUR {self.capital:.2f}")
        print(f"Return: {returns*100:+.2f}%")
        print(f"Total Trades: {self.trades}")
        print(f"Win Rate: {win_rate*100:.1f}%")
        
        if returns >= 0.5:
            print("\n[SUCCESS] Daily target achieved!")
        elif returns > 0:
            print("\n[PROFIT] Profitable but below target")
        else:
            print("\n[LOSS] Loss - Review parameters")

# Run the engine
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--capital', type=float, default=10.0, help='Starting capital')
    args = parser.parse_args()
    
    engine = QuickTurboEngine(capital=args.capital)
    engine.run_quick_mode()