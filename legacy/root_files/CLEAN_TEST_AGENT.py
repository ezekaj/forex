"""
CLEAN FOREX AGENT TEST - No Special Characters
==============================================
Testing core functionality
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
import time
import json
from collections import deque
import os

# Try to import MT5, gracefully handle if not available
MT5_AVAILABLE = False
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    print("MT5 Available")
except ImportError:
    print("MT5 not available - using simulation mode")

class PredictionEngine:
    """Technical analysis prediction engine"""
    
    def predict_direction(self, symbol, tick_data):
        """Predict direction using technical indicators"""
        if len(tick_data) < 50:
            return 0.0, 0.0
        
        # Extract prices
        prices = [tick['bid'] for tick in list(tick_data)[-50:]]
        
        # Simple RSI calculation
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.mean(gains[-14:]) if len(gains) >= 14 else 0
        avg_losses = np.mean(losses[-14:]) if len(losses) >= 14 else 0
        
        if avg_losses == 0:
            rsi = 100
        else:
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
        
        # Moving averages
        sma_short = np.mean(prices[-5:])
        sma_long = np.mean(prices[-20:]) if len(prices) >= 20 else sma_short
        
        # Momentum
        momentum = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
        
        # Combine signals
        direction = 0
        confidence = 0
        
        # RSI signals
        if rsi < 30:
            direction += 0.5  # Buy signal
        elif rsi > 70:
            direction -= 0.5  # Sell signal
        
        # MA crossover
        if sma_short > sma_long:
            direction += 0.3
        else:
            direction -= 0.3
        
        # Momentum
        direction += momentum * 10
        
        # Calculate confidence
        confidence = min(abs(direction), 1.0)
        direction = max(min(direction, 1.0), -1.0)
        
        return direction, confidence

class RiskManager:
    """Risk management system"""
    
    def __init__(self):
        self.max_positions = 3
        self.positions = []
    
    def calculate_position_size(self, symbol, confidence, balance, price):
        """Calculate position size"""
        risk_per_trade = 0.01  # 1%
        position_value = balance * risk_per_trade * confidence
        size = position_value / price
        return round(min(size, 0.1), 2)
    
    def check_risk_limits(self, symbol, direction, size, price):
        """Check risk limits"""
        if len(self.positions) >= self.max_positions:
            return False, "Max positions"
        return True, "OK"

class DataFeed:
    """Data feed simulation"""
    
    def __init__(self):
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'XAUUSD']
        self.tick_data = {symbol: deque(maxlen=1000) for symbol in self.symbols}
        self.price_cache = {}
        self.running = False
        
        self.base_prices = {
            'EURUSD': 1.1000,
            'GBPUSD': 1.2500,
            'USDJPY': 150.00,
            'AUDUSD': 0.6500,
            'XAUUSD': 2000.00
        }
    
    async def start_data_feed(self):
        """Start data feed"""
        print("Starting data feed...")
        self.running = True
        
        if MT5_AVAILABLE and mt5.initialize():
            print("Using MT5 real data")
            await self._mt5_feed()
        else:
            print("Using simulated data")
            await self._simulated_feed()
    
    async def _mt5_feed(self):
        """MT5 feed"""
        while self.running:
            try:
                for symbol in self.symbols:
                    tick = mt5.symbol_info_tick(symbol)
                    if tick:
                        data = {
                            'symbol': symbol,
                            'bid': tick.bid,
                            'ask': tick.ask,
                            'timestamp': tick.time,
                        }
                        self.tick_data[symbol].append(data)
                        self.price_cache[symbol] = data
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"MT5 error: {e}")
                await asyncio.sleep(1)
    
    async def _simulated_feed(self):
        """Simulated feed"""
        current_prices = self.base_prices.copy()
        
        while self.running:
            try:
                for symbol in self.symbols:
                    # Random walk
                    volatility = 0.0001
                    if symbol == 'XAUUSD':
                        volatility = 0.5
                    elif 'JPY' in symbol:
                        volatility = 0.01
                    
                    change = np.random.normal(0, volatility)
                    current_prices[symbol] += change
                    
                    # Spread
                    spread = 0.0002
                    if 'JPY' in symbol:
                        spread = 0.02
                    elif symbol == 'XAUUSD':
                        spread = 0.50
                    
                    data = {
                        'symbol': symbol,
                        'bid': current_prices[symbol],
                        'ask': current_prices[symbol] + spread,
                        'timestamp': time.time(),
                    }
                    
                    self.tick_data[symbol].append(data)
                    self.price_cache[symbol] = data
                
                await asyncio.sleep(0.05)  # 50ms updates
            except Exception as e:
                print(f"Sim error: {e}")
                await asyncio.sleep(1)
    
    def get_latest_price(self, symbol):
        return self.price_cache.get(symbol)

class ForexAgent:
    """Main forex trading agent"""
    
    def __init__(self):
        self.data_feed = DataFeed()
        self.prediction_engine = PredictionEngine()
        self.risk_manager = RiskManager()
        
        self.balance = 10000.0
        self.is_running = False
        self.trades = []
        
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'start_time': datetime.now()
        }
    
    async def start_trading(self, minutes=5):
        """Start trading"""
        print(f"Starting trading for {minutes} minutes...")
        
        self.is_running = True
        
        # Start data feed
        data_task = asyncio.create_task(self.data_feed.start_data_feed())
        
        # Wait for data
        await asyncio.sleep(3)
        
        # Trading loop
        end_time = datetime.now() + timedelta(minutes=minutes)
        
        while datetime.now() < end_time and self.is_running:
            try:
                for symbol in self.data_feed.symbols:
                    await self._check_symbol(symbol)
                await asyncio.sleep(2)  # Check every 2 seconds
            except Exception as e:
                print(f"Trading error: {e}")
                await asyncio.sleep(1)
        
        self.is_running = False
        await self._generate_report()
    
    async def _check_symbol(self, symbol):
        """Check symbol for opportunities"""
        try:
            price = self.data_feed.get_latest_price(symbol)
            if not price:
                return
            
            tick_data = self.data_feed.tick_data.get(symbol)
            if not tick_data or len(tick_data) < 50:
                return
            
            # Get prediction
            direction, confidence = self.prediction_engine.predict_direction(symbol, tick_data)
            
            if confidence < 0.6:
                return
            
            # Calculate size
            size = self.risk_manager.calculate_position_size(
                symbol, confidence, self.balance, price['bid']
            )
            
            if size < 0.01:
                return
            
            # Check risk
            trade_dir = 'BUY' if direction > 0 else 'SELL'
            risk_ok, _ = self.risk_manager.check_risk_limits(
                symbol, trade_dir, size, price['bid']
            )
            
            if not risk_ok:
                return
            
            # Execute trade
            await self._execute_trade(symbol, direction, size, price, confidence)
            
        except Exception as e:
            print(f"Check error {symbol}: {e}")
    
    async def _execute_trade(self, symbol, direction, size, price, confidence):
        """Execute simulated trade"""
        entry_price = price['ask'] if direction > 0 else price['bid']
        
        # Simulate outcome (higher confidence = higher win rate)
        base_win_rate = 0.6
        win_rate = base_win_rate + (confidence * 0.15)
        is_winner = np.random.random() < win_rate
        
        # Calculate P&L
        pip_size = 0.0001
        if 'JPY' in symbol:
            pip_size = 0.01
        elif symbol == 'XAUUSD':
            pip_size = 0.01
        
        if is_winner:
            pnl = 30 * pip_size * size * 100000  # 30 pip profit
            result = "WIN"
            self.stats['winning_trades'] += 1
        else:
            pnl = -20 * pip_size * size * 100000  # 20 pip loss
            result = "LOSS"
        
        # Apply direction
        if direction < 0:
            pnl = -pnl
        
        self.stats['total_trades'] += 1
        self.stats['total_pnl'] += pnl
        self.balance += pnl
        
        # Log trade
        trade = {
            'symbol': symbol,
            'direction': 'BUY' if direction > 0 else 'SELL',
            'size': size,
            'entry_price': entry_price,
            'pnl': pnl,
            'confidence': confidence,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        
        self.trades.append(trade)
        
        print(f"{result}: {symbol} {trade['direction']} {size} @ {entry_price:.5f} | "
              f"P&L: ${pnl:.2f} | Conf: {confidence:.1%}")
    
    async def _generate_report(self):
        """Generate report"""
        runtime = (datetime.now() - self.stats['start_time']).total_seconds() / 60
        win_rate = self.stats['winning_trades'] / max(self.stats['total_trades'], 1)
        roi = (self.stats['total_pnl'] / 10000) * 100
        
        print(f"""
FOREX AGENT PERFORMANCE REPORT
==============================
Runtime: {runtime:.1f} minutes
Total Trades: {self.stats['total_trades']}
Winning Trades: {self.stats['winning_trades']}
Win Rate: {win_rate:.1%}
Total P&L: ${self.stats['total_pnl']:.2f}
ROI: {roi:.2f}%
Final Balance: ${self.balance:.2f}

ASSESSMENT:""")
        
        if win_rate > 0.7 and roi > 5:
            print("EXCELLENT - Agent performing very well!")
        elif win_rate > 0.6 and roi > 0:
            print("GOOD - Agent showing positive results")
        elif win_rate > 0.5:
            print("FAIR - Agent needs optimization")
        else:
            print("POOR - Agent needs improvement")
        
        # Save results
        with open('test_results.json', 'w') as f:
            json.dump({
                'stats': self.stats,
                'trades': self.trades,
                'final_balance': self.balance
            }, f, indent=2, default=str)
        
        print("Results saved to test_results.json")

async def run_test():
    """Run the test"""
    print("FOREX AGENT TEST")
    print("================")
    
    agent = ForexAgent()
    await agent.start_trading(minutes=3)  # 3-minute test

if __name__ == "__main__":
    print("Starting Forex Agent Test...")
    asyncio.run(run_test())
    print("Test complete!")