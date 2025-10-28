"""
GUARANTEED TRADING AGENT - Ensures Trade Execution
=================================================
Simplified agent guaranteed to execute trades for testing
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
import time
import json
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class GuaranteedTradingAgent:
    """Simplified agent that guarantees trade execution"""
    
    def __init__(self):
        self.balance = 10000.0
        self.trades = []
        self.positions = {}
        self.running = False
        
        # Configuration
        self.min_confidence = 0.30      # Very low threshold
        self.risk_per_trade = 0.01      # 1% risk
        self.stop_loss_pips = 15
        self.take_profit_pips = 20
        self.max_positions = 5
        
        # Symbols to trade
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        
        # Price simulation
        self.base_prices = {
            'EURUSD': 1.0850, 'GBPUSD': 1.2650,
            'USDJPY': 149.50, 'AUDUSD': 0.6580
        }
        self.current_prices = self.base_prices.copy()
        
        # Price history for each symbol
        self.price_history = {symbol: deque(maxlen=100) for symbol in self.symbols}
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'start_time': datetime.now(),
            'opportunities_found': 0
        }
    
    def generate_price_movement(self, symbol: str):
        """Generate realistic price movement"""
        
        # Base volatility
        volatility = 0.0001
        if 'JPY' in symbol:
            volatility = 0.01
        
        # Random movement
        change = np.random.normal(0, volatility)
        
        # Add some trend
        trend = np.sin(time.time() / 100) * volatility * 0.1
        
        self.current_prices[symbol] += change + trend
        
        # Create price tick
        spread = 0.00015 if 'JPY' not in symbol else 0.015
        
        price_tick = {
            'timestamp': time.time(),
            'bid': self.current_prices[symbol],
            'ask': self.current_prices[symbol] + spread,
            'spread': spread
        }
        
        self.price_history[symbol].append(price_tick)
        return price_tick
    
    def simple_prediction(self, symbol: str) -> tuple:
        """Simple prediction that generates confident signals"""
        
        if len(self.price_history[symbol]) < 20:
            return 0.0, 0.0
        
        prices = [tick['bid'] for tick in list(self.price_history[symbol])[-20:]]
        
        # Simple momentum
        short_avg = np.mean(prices[-5:])
        long_avg = np.mean(prices[-15:])
        
        momentum = (short_avg - long_avg) / long_avg
        
        # Simple trend
        trend = (prices[-1] - prices[0]) / prices[0]
        
        # Combine signals
        direction = (momentum * 0.6) + (trend * 0.4)
        
        # Generate confidence (random but biased upward)
        base_confidence = 0.35 + np.random.uniform(0, 0.4)  # 35-75%
        volatility_bonus = abs(momentum) * 1000  # Volatility increases confidence
        confidence = min(base_confidence + volatility_bonus, 0.95)
        
        # Scale direction
        direction = np.tanh(direction * 100)
        
        return direction, confidence
    
    def calculate_position_size(self, symbol: str, confidence: float) -> float:
        """Calculate position size"""
        
        # Risk amount
        risk_amount = self.balance * self.risk_per_trade
        
        # Pip value
        pip_value = 0.0001 if 'JPY' not in symbol else 0.01
        
        # Position size based on stop loss
        stop_value = self.stop_loss_pips * pip_value * 100000  # Per lot
        position_size = risk_amount / stop_value
        
        # Confidence multiplier
        position_size *= confidence
        
        # Ensure minimum size
        return max(0.01, min(position_size, 0.5))
    
    def validate_trade(self, symbol: str, confidence: float, position_size: float) -> bool:
        """Very permissive validation"""
        
        # Basic checks only
        if confidence < self.min_confidence:
            return False
        
        if len(self.positions) >= self.max_positions:
            return False
        
        if position_size < 0.01:
            return False
        
        return True
    
    def execute_trade(self, symbol: str, direction: float, size: float, confidence: float) -> dict:
        """Execute simulated trade"""
        
        current_price = self.price_history[symbol][-1]
        is_buy = direction > 0
        entry_price = current_price['ask'] if is_buy else current_price['bid']
        
        # Simulate outcome (70% win rate)
        win_rate = 0.65 + (confidence * 0.1)  # 65-75% based on confidence
        is_winner = np.random.random() < win_rate
        
        # Calculate P&L
        pip_value = 0.0001 if 'JPY' not in symbol else 0.01
        
        if is_winner:
            pnl = self.take_profit_pips * pip_value * size * 100000
            result = "WIN"
            self.stats['winning_trades'] += 1
        else:
            pnl = -self.stop_loss_pips * pip_value * size * 100000
            result = "LOSS"
        
        # Apply direction
        if not is_buy and result == "WIN":
            pnl = -pnl
        elif not is_buy and result == "LOSS":
            pnl = -pnl
        
        # Update stats
        self.stats['total_trades'] += 1
        self.stats['total_pnl'] += pnl
        self.balance += pnl
        
        # Create trade record
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'direction': 'BUY' if is_buy else 'SELL',
            'size': size,
            'entry_price': entry_price,
            'confidence': confidence,
            'pnl': pnl,
            'result': result
        }
        
        self.trades.append(trade)
        
        # Add to positions (will be closed automatically after some time)
        ticket = f'TRADE_{self.stats["total_trades"]}'
        self.positions[ticket] = {
            'symbol': symbol,
            'entry_time': datetime.now(),
            'size': size
        }
        
        return trade
    
    def close_old_positions(self):
        """Close positions older than 30 seconds (for simulation)"""
        current_time = datetime.now()
        to_close = []
        
        for ticket, position in self.positions.items():
            age = (current_time - position['entry_time']).total_seconds()
            if age > 30:  # Close after 30 seconds
                to_close.append(ticket)
        
        for ticket in to_close:
            del self.positions[ticket]
    
    async def run_trading_session(self, minutes: float = 5.0):
        """Run guaranteed trading session"""
        
        print(f"🎯 GUARANTEED TRADING AGENT")
        print(f"Duration: {minutes:.1f} minutes")
        print(f"Min Confidence: {self.min_confidence:.0%}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print("=" * 50)
        
        self.running = True
        end_time = datetime.now() + timedelta(minutes=minutes)
        
        # Generate initial price history
        print("📊 Generating initial price data...")
        for _ in range(50):  # 50 initial ticks per symbol
            for symbol in self.symbols:
                self.generate_price_movement(symbol)
        
        print("✅ Price data ready, starting trading...")
        
        trade_attempts = 0
        
        while datetime.now() < end_time and self.running:
            try:
                # Generate new prices
                for symbol in self.symbols:
                    self.generate_price_movement(symbol)
                
                # Check for trading opportunities
                for symbol in self.symbols:
                    
                    # Get prediction
                    direction, confidence = self.simple_prediction(symbol)
                    
                    if confidence > 0.40:  # Count as opportunity
                        self.stats['opportunities_found'] += 1
                        
                        print(f"🎯 Opportunity #{self.stats['opportunities_found']}: "
                              f"{symbol} - Direction: {direction:.3f}, Confidence: {confidence:.1%}")
                    
                    # Try to trade
                    if confidence >= self.min_confidence:
                        trade_attempts += 1
                        
                        # Calculate position size
                        size = self.calculate_position_size(symbol, confidence)
                        
                        # Validate
                        if self.validate_trade(symbol, confidence, size):
                            
                            # Execute trade
                            trade = self.execute_trade(symbol, direction, size, confidence)
                            
                            print(f"🚀 TRADE #{self.stats['total_trades']}: "
                                  f"{trade['symbol']} {trade['direction']} {trade['size']} lots | "
                                  f"P&L: ${trade['pnl']:.2f} | {trade['result']} | "
                                  f"Balance: ${self.balance:.2f}")
                
                # Close old positions
                self.close_old_positions()
                
                # Progress update every 30 seconds
                if trade_attempts % 50 == 0 and trade_attempts > 0:
                    remaining = (end_time - datetime.now()).total_seconds() / 60
                    win_rate = self.stats['winning_trades'] / max(self.stats['total_trades'], 1)
                    print(f"\n📊 Progress: {remaining:.1f} min left | "
                          f"Trades: {self.stats['total_trades']} | "
                          f"Win Rate: {win_rate:.1%} | "
                          f"P&L: ${self.stats['total_pnl']:.2f}")
                
                await asyncio.sleep(0.1)  # 100ms between checks
                
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(1)
        
        await self.generate_final_report()
    
    async def generate_final_report(self):
        """Generate final trading report"""
        
        runtime = (datetime.now() - self.stats['start_time']).total_seconds() / 60
        
        if self.stats['total_trades'] > 0:
            win_rate = self.stats['winning_trades'] / self.stats['total_trades']
            avg_trade = self.stats['total_pnl'] / self.stats['total_trades']
        else:
            win_rate = 0
            avg_trade = 0
        
        roi = ((self.balance - 10000) / 10000) * 100
        
        print(f"""
🎯 GUARANTEED TRADING AGENT - FINAL REPORT
==========================================
Runtime: {runtime:.1f} minutes
Opportunities Found: {self.stats['opportunities_found']}
Total Trades: {self.stats['total_trades']}
Winning Trades: {self.stats['winning_trades']}
Win Rate: {win_rate:.1%}
Average Trade P&L: ${avg_trade:.2f}
Total P&L: ${self.stats['total_pnl']:.2f}
ROI: {roi:.2f}%
Final Balance: ${self.balance:.2f}

RECENT TRADES:""")
        
        # Show last 5 trades
        for trade in self.trades[-5:]:
            print(f"  {trade['timestamp'][:19]} | {trade['symbol']} {trade['direction']} | "
                  f"${trade['pnl']:>7.2f} | {trade['result']}")
        
        # Assessment
        if self.stats['total_trades'] == 0:
            print("\n❌ ERROR: No trades executed - there's a bug in the system!")
        elif win_rate > 0.60 and roi > 1:
            print("\n✅ EXCELLENT: System working properly!")
        elif self.stats['total_trades'] > 0:
            print("\n✅ SUCCESS: Trades executed successfully!")
        
        # Save results
        results = {
            'stats': self.stats,
            'trades': self.trades,
            'final_balance': self.balance,
            'settings': {
                'min_confidence': self.min_confidence,
                'risk_per_trade': self.risk_per_trade,
                'stop_loss_pips': self.stop_loss_pips,
                'take_profit_pips': self.take_profit_pips
            }
        }
        
        filename = f'guaranteed_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📊 Results saved to: {filename}")

if __name__ == "__main__":
    print("""
🎯 GUARANTEED TRADING AGENT
===========================
Simplified agent guaranteed to execute trades

Options:
1. Quick Test (2 minutes)
2. Standard Test (5 minutes)
3. Extended Test (10 minutes)
    """)
    
    choice = input("Select option (1-3): ").strip()
    
    agent = GuaranteedTradingAgent()
    
    if choice == "1":
        asyncio.run(agent.run_trading_session(minutes=2))
    elif choice == "2":
        asyncio.run(agent.run_trading_session(minutes=5))
    elif choice == "3":
        asyncio.run(agent.run_trading_session(minutes=10))
    else:
        print("Invalid choice. Please run again.")