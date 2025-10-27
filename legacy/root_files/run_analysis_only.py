#!/usr/bin/env python
"""
ANALYSIS-ONLY MODE - No Broker Needed!
Uses real market data to generate signals without executing trades
Perfect for testing and paper trading
"""

import sys
import os
import time
import pandas as pd
from datetime import datetime

sys.path.append('03_CORE_ENGINE')
sys.path.append('01_LIVE_TRADING')

print("\n" + "="*70)
print("FOREX ANALYSIS SYSTEM - NO BROKER REQUIRED")
print("="*70)
print("Using Alpha Vantage for real market data")
print("Generating trading signals without execution")
print("="*70)

class PaperTrader:
    """Paper trading with real signals"""
    
    def __init__(self, initial_capital=1000):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.trades = []
        self.open_positions = {}
        
        # Import components
        from smart_position_sizer import SmartPositionSizer
        from win_rate_optimizer import WinRateOptimizer
        from market_timing_system import MarketTimingSystem
        from advanced_features import AdvancedFeatureEngineering
        from performance_analytics import PerformanceAnalytics
        
        self.position_sizer = SmartPositionSizer()
        self.win_optimizer = WinRateOptimizer()
        self.timing_system = MarketTimingSystem()
        self.feature_engineer = AdvancedFeatureEngineering()
        self.performance_tracker = PerformanceAnalytics(initial_capital)
        
    def get_market_data(self, pair='EURUSD'):
        """Get real market data from Alpha Vantage"""
        print(f"\n[FETCHING] Real {pair} data...")
        
        import requests
        
        # Use Alpha Vantage API (already configured)
        from_currency = pair[:3]
        to_currency = pair[3:]
        
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'FX_INTRADAY',
            'from_symbol': from_currency,
            'to_symbol': to_currency,
            'interval': '5min',
            'apikey': 'KNF41ZTAUM44W2LN'
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'Time Series FX (5min)' in data:
                time_series = data['Time Series FX (5min)']
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)
                df.columns = ['open', 'high', 'low', 'close']
                df = df.sort_index()
                
                print(f"[OK] Got {len(df)} data points")
                print(f"[PRICE] Current: {df['close'].iloc[-1]:.5f}")
                return df
            else:
                print("[WARNING] Using simulated data (API limit reached)")
                return self.generate_simulated_data()
                
        except Exception as e:
            print(f"[ERROR] {e}")
            return self.generate_simulated_data()
    
    def generate_simulated_data(self):
        """Generate realistic simulated data"""
        import numpy as np
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
        
        # Realistic forex prices
        base_price = 1.1850  # EURUSD typical price
        returns = np.random.normal(0, 0.0005, 100)  # Small forex movements
        prices = base_price * (1 + returns).cumprod()
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.0001, 100)),
            'high': prices * (1 + abs(np.random.normal(0.0002, 0.0001, 100))),
            'low': prices * (1 - abs(np.random.normal(0.0002, 0.0001, 100))),
            'close': prices,
            'volume': np.random.randint(10000, 100000, 100)  # Add volume data
        }, index=dates)
        
        return df
    
    def analyze_and_trade(self, pair='EURUSD'):
        """Analyze market and paper trade"""
        
        # Check market timing
        should_trade, timing_reason = self.timing_system.should_trade_now(pair)
        status = self.timing_system.get_current_market_status()
        
        print(f"\n[TIMING] {timing_reason}")
        print(f"[LIQUIDITY] Score: {status['liquidity_score']:.2f}")
        
        if not should_trade:
            print(f"[SKIP] Poor market conditions")
            return None
        
        # Get market data
        df = self.get_market_data(pair)
        if df is None or df.empty:
            return None
        
        # Apply advanced features
        df_enhanced = self.feature_engineer.engineer_all_features(df)
        
        # Generate signal
        current_price = df['close'].iloc[-1]
        rsi = df_enhanced['rsi'].iloc[-1] if 'rsi' in df_enhanced else 50
        
        # Simple signal generation
        if rsi > 70:
            signal = 'SELL'
            confidence = min(0.8, (rsi - 70) / 30)
        elif rsi < 30:
            signal = 'BUY'
            confidence = min(0.8, (30 - rsi) / 30)
        else:
            signal = 'HOLD'
            confidence = 0.0
        
        print(f"\n[SIGNAL] {signal} (RSI: {rsi:.1f}, Confidence: {confidence:.1%})")
        
        if signal == 'HOLD':
            return None
        
        # Check with win rate optimizer
        decision = self.win_optimizer.get_trade_decision(
            signal_confidence=confidence,
            market_volatility=0.01
        )
        
        if not decision['trade']:
            print(f"[OPTIMIZER] Trade rejected: {decision['reason']}")
            return None
        
        # Calculate position size
        stop_loss = current_price * (0.98 if signal == 'BUY' else 1.02)
        take_profit = current_price * (1.02 if signal == 'BUY' else 0.98)
        
        position = self.position_sizer.calculate_position_size(
            account_equity=self.capital,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence
        )
        
        # Execute paper trade
        trade = {
            'pair': pair,
            'signal': signal,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position['position_size'],
            'confidence': confidence,
            'timestamp': datetime.now()
        }
        
        self.trades.append(trade)
        self.open_positions[pair] = trade
        
        print(f"\n[PAPER TRADE EXECUTED]")
        print(f"  Direction: {signal}")
        print(f"  Entry: {current_price:.5f}")
        print(f"  Stop Loss: {stop_loss:.5f}")
        print(f"  Take Profit: {take_profit:.5f}")
        print(f"  Position Size: ${position['position_size']:.2f}")
        print(f"  Risk: {position['risk_percentage']:.2f}% of capital")
        
        return trade
    
    def check_open_positions(self):
        """Check if positions hit TP or SL"""
        for pair, position in list(self.open_positions.items()):
            # Simulate price movement
            import numpy as np
            random_move = np.random.normal(0, 0.002)  # 0.2% std dev
            new_price = position['entry_price'] * (1 + random_move)
            
            # Check stop loss / take profit
            if position['signal'] == 'BUY':
                if new_price <= position['stop_loss']:
                    profit = -position['position_size'] * 0.02
                    print(f"[STOP LOSS HIT] {pair}: Lost ${abs(profit):.2f}")
                    self.capital += profit
                    del self.open_positions[pair]
                elif new_price >= position['take_profit']:
                    profit = position['position_size'] * 0.02
                    print(f"[TAKE PROFIT HIT] {pair}: Won ${profit:.2f}")
                    self.capital += profit
                    del self.open_positions[pair]
            else:  # SELL
                if new_price >= position['stop_loss']:
                    profit = -position['position_size'] * 0.02
                    print(f"[STOP LOSS HIT] {pair}: Lost ${abs(profit):.2f}")
                    self.capital += profit
                    del self.open_positions[pair]
                elif new_price <= position['take_profit']:
                    profit = position['position_size'] * 0.02
                    print(f"[TAKE PROFIT HIT] {pair}: Won ${profit:.2f}")
                    self.capital += profit
                    del self.open_positions[pair]
    
    def run_session(self, hours=1):
        """Run paper trading session"""
        print(f"\n[STARTING] {hours}-hour paper trading session")
        print(f"Initial Capital: ${self.capital:.2f}")
        
        end_time = datetime.now() + pd.Timedelta(hours=hours)
        check_interval = 5  # minutes
        pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        
        while datetime.now() < end_time:
            for pair in pairs:
                self.analyze_and_trade(pair)
                self.check_open_positions()
                time.sleep(2)
            
            # Show status
            profit = self.capital - self.initial_capital
            profit_pct = (profit / self.initial_capital) * 100
            
            print(f"\n[STATUS] Capital: ${self.capital:.2f} ({profit_pct:+.2f}%)")
            print(f"[POSITIONS] Open: {len(self.open_positions)}, Total: {len(self.trades)}")
            
            print(f"\nNext check in {check_interval} minutes...")
            time.sleep(check_interval * 60)
        
        # Final report
        self.print_report()
    
    def print_report(self):
        """Print trading report"""
        print("\n" + "="*70)
        print("PAPER TRADING REPORT")
        print("="*70)
        
        profit = self.capital - self.initial_capital
        profit_pct = (profit / self.initial_capital) * 100
        
        print(f"Starting Capital: ${self.initial_capital:.2f}")
        print(f"Ending Capital: ${self.capital:.2f}")
        print(f"Profit/Loss: ${profit:+.2f} ({profit_pct:+.2f}%)")
        print(f"Total Trades: {len(self.trades)}")
        
        if len(self.trades) > 0:
            wins = sum(1 for t in self.trades if self.capital > self.initial_capital)
            win_rate = wins / len(self.trades) * 100
            print(f"Win Rate: {win_rate:.1f}%")

def main():
    print("\n[NO BROKER REQUIRED!]")
    print("This mode uses real market data but doesn't execute real trades")
    print("Perfect for testing without any broker account\n")
    
    print("Select mode:")
    print("1. Quick Test (5 minutes)")
    print("2. 1 Hour Session")
    print("3. 24 Hour Simulation")
    
    choice = input("\nChoice (1-3): ")
    
    trader = PaperTrader(initial_capital=1000)
    
    if choice == "1":
        # Just analyze once
        for pair in ['EURUSD', 'GBPUSD']:
            trader.analyze_and_trade(pair)
    elif choice == "2":
        trader.run_session(hours=1)
    elif choice == "3":
        trader.run_session(hours=24)
    else:
        print("Invalid choice")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()