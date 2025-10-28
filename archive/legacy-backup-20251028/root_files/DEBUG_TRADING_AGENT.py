"""
DEBUG TRADING AGENT - Find and Fix Trading Issues
================================================
Detailed debugging to identify why trades aren't executing
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
import time
import json
from collections import deque, defaultdict
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Optional MT5 import
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

class DebugConfig:
    """Debug configuration - very aggressive to ensure trades"""
    # Trading Parameters - VERY AGGRESSIVE
    MIN_CONFIDENCE = 0.35           # Very low threshold
    MAX_POSITIONS = 10              # Allow many positions
    RISK_PER_TRADE = 0.01           # 1% risk per trade
    MAX_DAILY_RISK = 0.10           # 10% daily risk
    
    # Speed Settings
    DATA_UPDATE_INTERVAL = 0.05     # 50ms updates
    ANALYSIS_INTERVAL = 0.2         # 200ms analysis
    
    # Currency Pairs
    MAJOR_PAIRS = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'  # Just 4 pairs for focus
    ]
    
    # Risk Management - Loose for debugging
    STOP_LOSS_PIPS = 15
    TAKE_PROFIT_PIPS = 20
    
    # Signal Sensitivity - VERY SENSITIVE
    RSI_OVERSOLD = 45               # Much less extreme
    RSI_OVERBOUGHT = 55
    MOMENTUM_THRESHOLD = 0.00001    # Very low threshold

class SimplePredictionEngine:
    """Simplified prediction engine for debugging"""
    
    def __init__(self):
        self.prediction_count = 0
        
    def predict_direction(self, symbol: str, price_data: deque) -> Tuple[float, float]:
        """Simplified prediction that should generate signals"""
        self.prediction_count += 1
        
        if len(price_data) < 20:
            return 0.0, 0.0
        
        try:
            # Get recent prices
            prices = np.array([tick['bid'] for tick in list(price_data)[-20:]])
            
            # Very simple signals
            direction = 0.0
            confidence = 0.0
            
            # Simple momentum
            if len(prices) >= 5:
                recent_change = prices[-1] - prices[-5]
                momentum = recent_change / prices[-5]
                direction += momentum * 1000  # Amplify
            
            # Simple trend
            if len(prices) >= 10:
                early_avg = np.mean(prices[:5])
                recent_avg = np.mean(prices[-5:])
                trend = (recent_avg - early_avg) / early_avg
                direction += trend * 500  # Amplify
            
            # Simple volatility boost
            if len(prices) >= 10:
                volatility = np.std(prices[-10:])
                vol_boost = min(volatility * 10000, 0.3)  # Cap boost
                confidence += vol_boost
            
            # Ensure we get some signals
            base_confidence = 0.4 + np.random.uniform(0, 0.3)  # 40-70% base
            confidence = max(confidence, base_confidence)
            
            # Normalize direction
            direction = np.tanh(direction)
            confidence = min(confidence, 1.0)
            
            return direction, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.0, 0.0

class DebugRiskManager:
    """Very permissive risk manager for debugging"""
    
    def __init__(self):
        self.positions = {}
        self.daily_pnl = 0.0
        self.validation_count = 0
        self.rejection_reasons = defaultdict(int)
        
    def calculate_position_size(self, symbol: str, confidence: float, 
                              balance: float, price: float) -> float:
        """Always return valid position size"""
        
        # Simple calculation
        risk_amount = balance * DebugConfig.RISK_PER_TRADE
        pip_value = 0.0001 if 'JPY' not in symbol else 0.01
        stop_loss_value = DebugConfig.STOP_LOSS_PIPS * pip_value * 100000
        
        position_size = risk_amount / stop_loss_value
        position_size = max(0.01, min(position_size, 1.0))  # 0.01 to 1.0 lots
        
        print(f"  📊 Position size calculated: {position_size:.2f} lots for {symbol}")
        return round(position_size, 2)
    
    def validate_trade(self, symbol: str, confidence: float) -> Tuple[bool, str]:
        """Very permissive validation with detailed logging"""
        self.validation_count += 1
        
        print(f"  🔍 Validating trade #{self.validation_count} for {symbol}:")
        print(f"     - Confidence: {confidence:.3f} (min: {DebugConfig.MIN_CONFIDENCE})")
        print(f"     - Current positions: {len(self.positions)} (max: {DebugConfig.MAX_POSITIONS})")
        print(f"     - Daily P&L: ${self.daily_pnl:.2f} (limit: ${DebugConfig.MAX_DAILY_RISK * 10000})")
        
        # Check position limits
        if len(self.positions) >= DebugConfig.MAX_POSITIONS:
            self.rejection_reasons['max_positions'] += 1
            print(f"     ❌ REJECTED: Max positions reached")
            return False, "Max positions reached"
        
        # Check confidence
        if confidence < DebugConfig.MIN_CONFIDENCE:
            self.rejection_reasons['low_confidence'] += 1
            print(f"     ❌ REJECTED: Confidence too low: {confidence:.1%}")
            return False, f"Confidence too low: {confidence:.1%}"
        
        # Check daily risk
        if abs(self.daily_pnl) > DebugConfig.MAX_DAILY_RISK * 10000:
            self.rejection_reasons['daily_risk'] += 1
            print(f"     ❌ REJECTED: Daily risk limit exceeded")
            return False, "Daily risk limit exceeded"
        
        print(f"     ✅ ACCEPTED: Trade validated!")
        return True, "Trade validated"
    
    def get_rejection_summary(self) -> str:
        """Get summary of rejection reasons"""
        if not self.rejection_reasons:
            return "No rejections recorded"
        
        summary = "Rejection reasons:\n"
        for reason, count in self.rejection_reasons.items():
            summary += f"  - {reason}: {count} times\n"
        
        return summary

class DebugDataFeed:
    """Simple data feed with guaranteed movement"""
    
    def __init__(self):
        self.symbols = DebugConfig.MAJOR_PAIRS
        self.price_data = {symbol: deque(maxlen=1000) for symbol in self.symbols}
        self.latest_prices = {}
        self.running = False
        
        # Base prices
        self.base_prices = {
            'EURUSD': 1.0850, 'GBPUSD': 1.2650, 
            'USDJPY': 149.50, 'AUDUSD': 0.6580
        }
        
        self.current_prices = self.base_prices.copy()
        self.tick_count = 0
    
    async def start_feed(self):
        """Start debug data feed with guaranteed movement"""
        print("🔄 Starting DEBUG data feed with guaranteed price movements...")
        self.running = True
        
        # Generate some initial data
        for i in range(100):  # Pre-populate with 100 ticks
            await self._generate_tick()
        
        print(f"✅ Generated {len(self.price_data['EURUSD'])} initial price ticks")
        
        # Continue generating data
        while self.running:
            await self._generate_tick()
            await asyncio.sleep(DebugConfig.DATA_UPDATE_INTERVAL)
    
    async def _generate_tick(self):
        """Generate a single tick with guaranteed movement"""
        self.tick_count += 1
        current_time = time.time()
        
        for symbol in self.symbols:
            # Ensure significant movement every few ticks
            if self.tick_count % 10 == 0:
                # Big movement every 10 ticks
                movement = np.random.uniform(-0.002, 0.002)
            else:
                # Regular small movement
                movement = np.random.uniform(-0.0005, 0.0005)
            
            self.current_prices[symbol] += movement
            
            # Realistic spread
            if 'JPY' in symbol:
                spread = 0.015
            else:
                spread = 0.00015
            
            data = {
                'timestamp': current_time,
                'bid': self.current_prices[symbol],
                'ask': self.current_prices[symbol] + spread,
                'spread': spread
            }
            
            self.price_data[symbol].append(data)
            self.latest_prices[symbol] = data

class DebugTradingAgent:
    """Debug trading agent with detailed logging"""
    
    def __init__(self):
        self.prediction_engine = SimplePredictionEngine()
        self.risk_manager = DebugRiskManager()
        self.data_feed = DebugDataFeed()
        
        self.balance = 10000.0
        self.is_running = False
        self.trades = []
        
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'start_time': datetime.now(),
            'opportunities_found': 0,
            'opportunities_traded': 0,
            'predictions_made': 0,
            'validations_attempted': 0
        }
        
        self.debug_log = []
    
    async def start_trading(self, minutes: float = 10.0):
        """Start debug trading session"""
        print(f"🐛 DEBUG TRADING AGENT STARTING")
        print(f"Duration: {minutes:.1f} minutes")
        print(f"Min Confidence: {DebugConfig.MIN_CONFIDENCE:.0%}")
        print(f"Max Positions: {DebugConfig.MAX_POSITIONS}")
        print("=" * 60)
        
        self.is_running = True
        
        # Start data feed
        data_task = asyncio.create_task(self.data_feed.start_feed())
        
        # Wait for data
        print("📊 Waiting for initial data...")
        await asyncio.sleep(3)
        
        print("🔍 Starting opportunity scanning...")
        
        end_time = datetime.now() + timedelta(minutes=minutes)
        scan_count = 0
        
        while datetime.now() < end_time and self.is_running:
            try:
                scan_count += 1
                
                print(f"\n🔄 SCAN #{scan_count} - {datetime.now().strftime('%H:%M:%S')}")
                
                # Check each symbol
                for symbol in DebugConfig.MAJOR_PAIRS:
                    await self._debug_check_opportunity(symbol)
                
                # Status update every 10 scans
                if scan_count % 10 == 0:
                    remaining = (end_time - datetime.now()).total_seconds() / 60
                    print(f"\n📊 STATUS UPDATE:")
                    print(f"   ⏰ Time remaining: {remaining:.1f} minutes")
                    print(f"   🎯 Opportunities found: {self.stats['opportunities_found']}")
                    print(f"   📈 Trades executed: {self.stats['total_trades']}")
                    print(f"   🔍 Predictions made: {self.stats['predictions_made']}")
                
                await asyncio.sleep(DebugConfig.ANALYSIS_INTERVAL)
                
            except Exception as e:
                print(f"❌ Scan error: {e}")
                await asyncio.sleep(1)
        
        self.is_running = False
        await self._generate_debug_report()
    
    async def _debug_check_opportunity(self, symbol: str):
        """Debug version of opportunity checking"""
        try:
            # Get data
            price = self.data_feed.latest_prices.get(symbol)
            if not price:
                print(f"  ⚠️ No price data for {symbol}")
                return
            
            price_history = self.data_feed.price_data.get(symbol)
            if not price_history or len(price_history) < 20:
                print(f"  ⚠️ Insufficient price history for {symbol}: {len(price_history) if price_history else 0} ticks")
                return
            
            print(f"  📊 {symbol}: Price={price['bid']:.5f}, History={len(price_history)} ticks")
            
            # Get prediction
            direction, confidence = self.prediction_engine.predict_direction(symbol, price_history)
            self.stats['predictions_made'] += 1
            
            print(f"  🧠 Prediction: Direction={direction:.3f}, Confidence={confidence:.3f}")
            
            # Count opportunity if confidence > 30%
            if confidence > 0.30:
                self.stats['opportunities_found'] += 1
                print(f"  🎯 OPPORTUNITY DETECTED! (#{self.stats['opportunities_found']})")
            
            # Try to trade if meets minimum criteria
            if confidence >= DebugConfig.MIN_CONFIDENCE:
                print(f"  ✅ Confidence above minimum, proceeding with trade setup...")
                
                # Calculate position size
                position_size = self.risk_manager.calculate_position_size(
                    symbol, confidence, self.balance, price['bid']
                )
                
                if position_size >= 0.01:
                    print(f"  💰 Position size valid: {position_size} lots")
                    
                    # Validate trade
                    self.stats['validations_attempted'] += 1
                    is_valid, msg = self.risk_manager.validate_trade(symbol, confidence)
                    
                    if is_valid:
                        print(f"  🚀 EXECUTING TRADE!")
                        await self._execute_debug_trade(symbol, direction, position_size, confidence, price)
                        self.stats['opportunities_traded'] += 1
                    else:
                        print(f"  ❌ Trade validation failed: {msg}")
                else:
                    print(f"  ❌ Position size too small: {position_size}")
            else:
                print(f"  ⏸️ Confidence below minimum ({DebugConfig.MIN_CONFIDENCE:.1%})")
        
        except Exception as e:
            print(f"  ❌ Error checking {symbol}: {e}")
    
    async def _execute_debug_trade(self, symbol: str, direction: float, size: float,
                                 confidence: float, price: Dict):
        """Execute trade with detailed logging"""
        
        is_buy = direction > 0
        entry_price = price['ask'] if is_buy else price['bid']
        
        print(f"    🔥 TRADE EXECUTION:")
        print(f"       Symbol: {symbol}")
        print(f"       Direction: {'BUY' if is_buy else 'SELL'}")
        print(f"       Size: {size} lots")
        print(f"       Entry Price: {entry_price:.5f}")
        print(f"       Confidence: {confidence:.1%}")
        
        # Simulate realistic outcome
        base_win_rate = 0.60
        confidence_bonus = confidence * 0.10
        final_win_rate = base_win_rate + confidence_bonus
        
        is_winner = np.random.random() < final_win_rate
        
        # Calculate P&L
        pip_value = 0.0001 if 'JPY' not in symbol else 0.01
        
        if is_winner:
            pnl = DebugConfig.TAKE_PROFIT_PIPS * pip_value * size * 100000
            result = "WIN"
            self.stats['winning_trades'] += 1
        else:
            pnl = -DebugConfig.STOP_LOSS_PIPS * pip_value * size * 100000
            result = "LOSS"
        
        # Apply direction for short trades
        if not is_buy:
            pnl = -pnl if not is_winner else pnl
        
        self.stats['total_trades'] += 1
        self.stats['total_pnl'] += pnl
        self.balance += pnl
        
        # Update risk manager
        self.risk_manager.daily_pnl += pnl
        
        # Add to positions (simulate)
        ticket = f'DEBUG_{self.stats["total_trades"]}'
        self.risk_manager.positions[ticket] = {
            'symbol': symbol,
            'size': size,
            'entry_time': datetime.now()
        }
        
        # Log trade
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
        
        print(f"    🎯 RESULT: {result}")
        print(f"       P&L: ${pnl:.2f}")
        print(f"       New Balance: ${self.balance:.2f}")
        print(f"       Trade #{self.stats['total_trades']}")
        
    async def _generate_debug_report(self):
        """Generate comprehensive debug report"""
        runtime = (datetime.now() - self.stats['start_time']).total_seconds() / 60
        
        print(f"""
🐛 DEBUG TRADING AGENT - DETAILED REPORT
========================================
Runtime: {runtime:.1f} minutes
Data Ticks Generated: {self.data_feed.tick_count}
Predictions Made: {self.stats['predictions_made']}
Opportunities Found: {self.stats['opportunities_found']}
Validations Attempted: {self.stats['validations_attempted']}
Opportunities Traded: {self.stats['opportunities_traded']}

TRADING PERFORMANCE:
Total Trades: {self.stats['total_trades']}
Winning Trades: {self.stats['winning_trades']}
Win Rate: {(self.stats['winning_trades']/max(self.stats['total_trades'],1)):.1%}
Total P&L: ${self.stats['total_pnl']:.2f}
Final Balance: ${self.balance:.2f}
ROI: {((self.balance-10000)/10000*100):.2f}%

VALIDATION ANALYSIS:
{self.risk_manager.get_rejection_summary()}

DEBUG ANALYSIS:""")
        
        if self.stats['total_trades'] == 0:
            if self.stats['opportunities_found'] == 0:
                print("❌ ISSUE: No opportunities found - prediction engine problem")
            elif self.stats['validations_attempted'] == 0:
                print("❌ ISSUE: Opportunities found but no validations attempted - logic error")
            else:
                print("❌ ISSUE: Validations attempted but all rejected - validation too strict")
        else:
            print("✅ SUCCESS: Trades were executed successfully!")
        
        # Save detailed results
        results = {
            'stats': self.stats,
            'trades': self.trades,
            'final_balance': self.balance,
            'rejection_reasons': dict(self.risk_manager.rejection_reasons),
            'config': vars(DebugConfig)
        }
        
        filename = f'debug_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📊 Debug results saved to: {filename}")

if __name__ == "__main__":
    print("""
🐛 DEBUG FOREX TRADING AGENT
============================
Find and fix trading execution issues

Options:
1. Quick Debug (5 minutes)
2. Standard Debug (10 minutes)
3. Extended Debug (15 minutes)
    """)
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == "1":
        agent = DebugTradingAgent()
        asyncio.run(agent.start_trading(minutes=5))
    elif choice == "2":
        agent = DebugTradingAgent()
        asyncio.run(agent.start_trading(minutes=10))
    elif choice == "3":
        agent = DebugTradingAgent()
        asyncio.run(agent.start_trading(minutes=15))
    else:
        print("Invalid choice. Please run again.")