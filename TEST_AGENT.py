"""
TEST ULTIMATE FOREX AGENT - Simplified Version for Testing
==========================================================
Testing the core functionality without ML dependencies
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import json
from collections import deque, defaultdict
import logging
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import MT5, gracefully handle if not available
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    print("✅ MT5 Available")
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ MT5 not available - using simulation mode")

class SimplePredictionEngine:
    """Simplified prediction engine using technical indicators"""
    
    def __init__(self):
        self.feature_windows = 50
        self.prediction_cache = {}
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50  # Neutral RSI
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.mean(gains[-period:])
        avg_losses = np.mean(losses[-period:])
        
        if avg_losses == 0:
            return 100
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices):
        """Calculate MACD"""
        if len(prices) < 26:
            return 0, 0  # No signal
        
        ema12 = np.mean(prices[-12:])
        ema26 = np.mean(prices[-26:])
        macd = ema12 - ema26
        
        # Simple signal line (9-period EMA of MACD)
        if len(prices) >= 35:
            macd_history = []
            for i in range(9):
                if len(prices) >= 26 + i:
                    ema12_i = np.mean(prices[-(12+i):len(prices)-i])
                    ema26_i = np.mean(prices[-(26+i):len(prices)-i])
                    macd_history.append(ema12_i - ema26_i)
            signal = np.mean(macd_history) if macd_history else 0
        else:
            signal = 0
            
        return macd, signal
    
    def predict_direction(self, symbol, tick_data):
        """Predict direction using technical analysis"""
        if len(tick_data) < self.feature_windows:
            return 0.0, 0.0
        
        # Extract price data
        prices = [tick['bid'] for tick in list(tick_data)[-self.feature_windows:]]
        
        # Calculate indicators
        rsi = self.calculate_rsi(prices)
        macd, macd_signal = self.calculate_macd(prices)
        
        # Simple momentum
        momentum = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
        
        # Moving average crossover
        sma_short = np.mean(prices[-5:])
        sma_long = np.mean(prices[-20:]) if len(prices) >= 20 else sma_short
        ma_signal = (sma_short - sma_long) / sma_long if sma_long != 0 else 0
        
        # Combine signals
        signals = []
        
        # RSI signals
        if rsi < 30:
            signals.append(0.8)  # Oversold - buy signal
        elif rsi > 70:
            signals.append(-0.8)  # Overbought - sell signal
        else:
            signals.append(0)
        
        # MACD signal
        if macd > macd_signal:
            signals.append(0.6)
        else:
            signals.append(-0.6)
        
        # Momentum signal
        signals.append(momentum * 100)  # Scale momentum
        
        # MA crossover
        signals.append(ma_signal * 50)  # Scale MA signal
        
        # Average the signals
        direction = np.mean(signals)
        confidence = min(abs(direction), 1.0)  # Cap at 1.0
        
        # Normalize direction
        direction = np.tanh(direction)  # Between -1 and 1
        
        return direction, confidence

class SimpleRiskManager:
    """Simplified risk management"""
    
    def __init__(self):
        self.max_risk_per_trade = 0.01
        self.max_positions = 3
        self.positions = []
    
    def calculate_position_size(self, symbol, confidence, balance, price):
        """Calculate position size"""
        # Simple Kelly-like calculation
        base_risk = balance * self.max_risk_per_trade
        confidence_multiplier = 0.5 + (confidence * 0.5)
        
        position_value = base_risk * confidence_multiplier
        position_size = position_value / price
        
        return round(min(position_size, 0.1), 2)  # Cap at 0.1 lots
    
    def check_risk_limits(self, symbol, direction, size, price):
        """Check risk limits"""
        if len(self.positions) >= self.max_positions:
            return False, "Max positions reached"
        
        return True, "Risk check passed"

class SimpleDataFeed:
    """Simplified data feed for testing"""
    
    def __init__(self):
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'XAUUSD']
        self.tick_data = {symbol: deque(maxlen=1000) for symbol in self.symbols}
        self.price_cache = {}
        self.running = False
        
        # Base prices
        self.base_prices = {
            'EURUSD': 1.1000,
            'GBPUSD': 1.2500,
            'USDJPY': 150.00,
            'AUDUSD': 0.6500,
            'XAUUSD': 2000.00
        }
    
    async def start_data_feed(self):
        """Start simulated data feed"""
        print("📡 Starting simulated data feed...")
        self.running = True
        
        # Try MT5 first
        if MT5_AVAILABLE:
            if mt5.initialize():
                print("✅ MT5 initialized - using real data")
                await self._mt5_feed()
            else:
                print("⚠️ MT5 initialization failed - using simulated data")
                await self._simulated_feed()
        else:
            await self._simulated_feed()
    
    async def _mt5_feed(self):
        """MT5 real data feed"""
        while self.running:
            try:
                for symbol in self.symbols:
                    tick = mt5.symbol_info_tick(symbol)
                    if tick:
                        price_data = {
                            'symbol': symbol,
                            'bid': tick.bid,
                            'ask': tick.ask,
                            'timestamp': tick.time,
                            'source': 'mt5'
                        }
                        self.tick_data[symbol].append(price_data)
                        self.price_cache[symbol] = price_data
                
                await asyncio.sleep(0.1)  # 100ms updates
                
            except Exception as e:
                print(f"❌ MT5 feed error: {e}")
                await asyncio.sleep(1)
    
    async def _simulated_feed(self):
        """Simulated data feed"""
        current_prices = self.base_prices.copy()
        
        while self.running:
            try:
                for symbol in self.symbols:
                    # Generate realistic price movement
                    volatility = 0.0001 if 'USD' in symbol else 0.01
                    if symbol == 'XAUUSD':
                        volatility = 0.5
                    
                    change = np.random.normal(0, volatility)
                    current_prices[symbol] += change
                    
                    # Add spread
                    if 'JPY' in symbol:
                        spread = 0.02
                    elif symbol == 'XAUUSD':
                        spread = 0.50
                    else:
                        spread = 0.0002
                    
                    price_data = {
                        'symbol': symbol,
                        'bid': current_prices[symbol],
                        'ask': current_prices[symbol] + spread,
                        'timestamp': time.time(),
                        'source': 'simulated'
                    }
                    
                    self.tick_data[symbol].append(price_data)
                    self.price_cache[symbol] = price_data
                
                await asyncio.sleep(0.05)  # 50ms updates - very fast
                
            except Exception as e:
                print(f"❌ Simulated feed error: {e}")
                await asyncio.sleep(1)
    
    def get_latest_price(self, symbol):
        """Get latest price"""
        return self.price_cache.get(symbol)

class SimpleForexAgent:
    """Simplified forex agent for testing"""
    
    def __init__(self):
        self.data_feed = SimpleDataFeed()
        self.prediction_engine = SimplePredictionEngine()
        self.risk_manager = SimpleRiskManager()
        
        self.account_balance = 10000.0
        self.is_running = False
        self.trade_log = []
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'start_time': datetime.now()
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    async def start_trading(self, duration_minutes=5):
        """Start trading for specified duration"""
        print(f"🚀 Starting Simple Forex Agent for {duration_minutes} minutes...")
        
        self.is_running = True
        
        # Start data feed
        data_task = asyncio.create_task(self.data_feed.start_data_feed())
        
        # Wait for data to start
        await asyncio.sleep(2)
        
        # Trading loop
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        while datetime.now() < end_time and self.is_running:
            try:
                # Check all symbols
                for symbol in self.data_feed.symbols:
                    await self._analyze_symbol(symbol)
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(1)
        
        self.is_running = False
        print("⏹️ Trading stopped")
        
        # Generate performance report
        await self._generate_report()
    
    async def _analyze_symbol(self, symbol):
        """Analyze symbol for trading opportunities"""
        try:
            # Get latest price
            latest_price = self.data_feed.get_latest_price(symbol)
            if not latest_price:
                return
            
            # Get tick data
            tick_data = self.data_feed.tick_data.get(symbol)
            if not tick_data or len(tick_data) < 50:
                return
            
            # Get prediction
            direction, confidence = self.prediction_engine.predict_direction(symbol, tick_data)
            
            # Check signal strength
            if confidence < 0.6:  # Require 60%+ confidence
                return
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                symbol, confidence, self.account_balance, latest_price['bid']
            )
            
            if position_size < 0.01:
                return
            
            # Check risk limits
            trade_direction = 'BUY' if direction > 0 else 'SELL'
            risk_ok, risk_msg = self.risk_manager.check_risk_limits(
                symbol, trade_direction, position_size, latest_price['bid']
            )
            
            if not risk_ok:
                return
            
            # Execute simulated trade
            await self._execute_simulated_trade(symbol, direction, position_size, 
                                              latest_price, confidence)
            
        except Exception as e:
            self.logger.error(f"Analysis error for {symbol}: {e}")
    
    async def _execute_simulated_trade(self, symbol, direction, size, price_data, confidence):
        """Execute simulated trade"""
        entry_price = price_data['ask'] if direction > 0 else price_data['bid']
        
        # Simulate execution
        execution_time = np.random.uniform(10, 100)  # 10-100ms
        slippage = np.random.uniform(0, 1)  # 0-1 pip slippage
        
        # Calculate exit prices
        pip_size = 0.0001 if 'JPY' not in symbol else 0.01
        if symbol == 'XAUUSD':
            pip_size = 0.01
        
        sl_distance = 20 * pip_size
        tp_distance = 30 * pip_size
        
        if direction > 0:
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
        
        # Simulate trade outcome (70% win rate for testing)
        win_probability = 0.65 + (confidence * 0.1)  # 65-75% based on confidence
        is_winner = np.random.random() < win_probability
        
        if is_winner:
            exit_price = tp_price
            pnl = tp_distance * size * 100000 * (1 if direction > 0 else -1)
            result = "WIN"
            self.performance_stats['winning_trades'] += 1
        else:
            exit_price = sl_price
            pnl = -sl_distance * size * 100000 * (1 if direction > 0 else -1)
            result = "LOSS"
        
        self.performance_stats['total_trades'] += 1
        self.performance_stats['total_pnl'] += pnl
        self.account_balance += pnl
        
        # Log trade
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'direction': 'BUY' if direction > 0 else 'SELL',
            'size': size,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'confidence': confidence,
            'execution_time': execution_time,
            'result': result
        }
        
        self.trade_log.append(trade_record)
        
        print(f"📊 {result}: {symbol} {trade_record['direction']} {size} "
              f"@ {entry_price:.5f} | P&L: ${pnl:.2f} | Conf: {confidence:.1%}")
    
    async def _generate_report(self):
        """Generate performance report"""
        runtime = (datetime.now() - self.performance_stats['start_time']).total_seconds() / 60
        
        win_rate = (self.performance_stats['winning_trades'] / 
                   max(self.performance_stats['total_trades'], 1))
        
        roi = (self.performance_stats['total_pnl'] / 10000) * 100
        
        print(f"""
📊 SIMPLE FOREX AGENT - PERFORMANCE REPORT
==========================================
Runtime: {runtime:.1f} minutes
Total Trades: {self.performance_stats['total_trades']}
Winning Trades: {self.performance_stats['winning_trades']}
Win Rate: {win_rate:.1%}
Total P&L: ${self.performance_stats['total_pnl']:.2f}
ROI: {roi:.2f}%
Final Balance: ${self.account_balance:.2f}

🎯 ASSESSMENT:
""")
        
        if win_rate > 0.7 and roi > 5:
            print("✅ EXCELLENT - Agent performing very well!")
        elif win_rate > 0.6 and roi > 0:
            print("✅ GOOD - Agent showing positive results")
        elif win_rate > 0.5:
            print("⚠️ FAIR - Agent needs optimization")
        else:
            print("❌ POOR - Agent needs significant improvement")

# Test function
async def test_simple_agent():
    """Test the simple agent"""
    print("🧪 TESTING SIMPLE FOREX AGENT")
    print("=" * 40)
    
    agent = SimpleForexAgent()
    await agent.start_trading(duration_minutes=5)  # 5-minute test

# Main execution
if __name__ == "__main__":
    print("""
🎯 SIMPLE FOREX AGENT TESTER
============================
This will run a 5-minute trading simulation
to test the core agent functionality.
    """)
    
    input("Press Enter to start test...")
    asyncio.run(test_simple_agent())
    print("\n✅ Test complete!")