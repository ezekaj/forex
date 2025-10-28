"""
ULTRA TRADING AGENT - Bulletproof Position Management
====================================================
Crystal clear entry/exit logic with guaranteed position management
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
import time
import json
from collections import deque
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class Position:
    """Individual position with clear lifecycle management"""
    
    def __init__(self, symbol: str, direction: str, size: float, entry_price: float, 
                 confidence: float, stop_loss: float, take_profit: float):
        self.symbol = symbol
        self.direction = direction  # 'BUY' or 'SELL'
        self.size = size
        self.entry_price = entry_price
        self.confidence = confidence
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # Position tracking
        self.id = f"{symbol}_{direction}_{int(time.time() * 1000)}"
        self.entry_time = datetime.now()
        self.status = "OPEN"  # OPEN, CLOSED_TP, CLOSED_SL, CLOSED_TIME
        self.exit_time = None
        self.exit_price = None
        self.pnl = 0.0
        
        # Position limits
        self.max_age_minutes = 30  # Close after 30 minutes max
        
        print(f"🚀 POSITION OPENED: {self.id}")
        print(f"   Symbol: {symbol} | Direction: {direction} | Size: {size}")
        print(f"   Entry: {entry_price:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
    
    def update(self, current_bid: float, current_ask: float) -> Optional[Dict]:
        """Update position and check for exit conditions"""
        
        if self.status != "OPEN":
            return None
        
        # Get current price based on position direction
        if self.direction == "BUY":
            current_price = current_bid  # Exit at bid for buy positions
        else:
            current_price = current_ask   # Exit at ask for sell positions
        
        # Calculate unrealized P&L
        if self.direction == "BUY":
            price_diff = current_price - self.entry_price
        else:
            price_diff = self.entry_price - current_price
        
        pip_value = 0.0001 if 'JPY' not in self.symbol else 0.01
        pips_profit = price_diff / pip_value
        self.pnl = pips_profit * pip_value * self.size * 100000
        
        # Check exit conditions
        exit_reason = None
        
        # 1. Take Profit Check
        if self.direction == "BUY" and current_price >= self.take_profit:
            exit_reason = "TAKE_PROFIT"
            self.status = "CLOSED_TP"
        elif self.direction == "SELL" and current_price <= self.take_profit:
            exit_reason = "TAKE_PROFIT"
            self.status = "CLOSED_TP"
        
        # 2. Stop Loss Check
        elif self.direction == "BUY" and current_price <= self.stop_loss:
            exit_reason = "STOP_LOSS"
            self.status = "CLOSED_SL"
        elif self.direction == "SELL" and current_price >= self.stop_loss:
            exit_reason = "STOP_LOSS"
            self.status = "CLOSED_SL"
        
        # 3. Time-based exit (prevent overnight positions)
        elif (datetime.now() - self.entry_time).total_seconds() > self.max_age_minutes * 60:
            exit_reason = "TIME_EXIT"
            self.status = "CLOSED_TIME"
        
        # If exit triggered, finalize the position
        if exit_reason:
            self.exit_time = datetime.now()
            self.exit_price = current_price
            
            # Recalculate final P&L
            if self.direction == "BUY":
                final_price_diff = self.exit_price - self.entry_price
            else:
                final_price_diff = self.entry_price - self.exit_price
            
            final_pips = final_price_diff / pip_value
            self.pnl = final_pips * pip_value * self.size * 100000
            
            duration = (self.exit_time - self.entry_time).total_seconds()
            
            print(f"🔄 POSITION CLOSED: {self.id}")
            print(f"   Exit Reason: {exit_reason}")
            print(f"   Exit Price: {self.exit_price:.5f}")
            print(f"   Duration: {duration/60:.1f} minutes")
            print(f"   P&L: ${self.pnl:.2f}")
            
            return {
                'position_id': self.id,
                'symbol': self.symbol,
                'direction': self.direction,
                'size': self.size,
                'entry_price': self.entry_price,
                'exit_price': self.exit_price,
                'entry_time': self.entry_time,
                'exit_time': self.exit_time,
                'duration_minutes': duration / 60,
                'pnl': self.pnl,
                'exit_reason': exit_reason,
                'confidence': self.confidence
            }
        
        return None
    
    def is_closed(self) -> bool:
        return self.status != "OPEN"

class UltraTradingEngine:
    """Ultra-sophisticated trading engine with bulletproof logic"""
    
    def __init__(self):
        # Trading configuration
        self.config = {
            'min_confidence': 0.40,      # 40% minimum confidence
            'max_positions': 5,          # Maximum 5 open positions
            'risk_per_trade': 0.015,     # 1.5% risk per trade
            'stop_loss_pips': 15,        # 15 pip stop loss
            'take_profit_pips': 20,      # 20 pip take profit
            'min_position_size': 0.01,   # Minimum 0.01 lots
            'max_position_size': 0.50,   # Maximum 0.50 lots
        }
        
        # Symbols to trade
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        
        # Position management
        self.open_positions: List[Position] = []
        self.closed_positions: List[Dict] = []
        self.position_history: List[Dict] = []
        
        # Account management
        self.starting_balance = 10000.0
        self.current_balance = 10000.0
        self.total_pnl = 0.0
        self.daily_pnl = 0.0
        
        # Market data simulation
        self.base_prices = {
            'EURUSD': 1.0850, 'GBPUSD': 1.2650,
            'USDJPY': 149.50, 'AUDUSD': 0.6580
        }
        self.current_prices = self.base_prices.copy()
        self.price_history = {symbol: deque(maxlen=200) for symbol in self.symbols}
        
        # Trading statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'opportunities_found': 0,
            'positions_opened': 0,
            'positions_closed': 0,
            'start_time': datetime.now()
        }
        
        # Control flags
        self.running = False
        self.trading_enabled = True
    
    def generate_market_data(self):
        """Generate realistic market movements"""
        
        for symbol in self.symbols:
            # Calculate volatility
            base_vol = 0.00008  # Base volatility
            if 'JPY' in symbol:
                base_vol = 0.008
            
            # Time-based volatility (higher during trading hours)
            hour = datetime.now().hour
            if 8 <= hour <= 16:  # London/European session
                base_vol *= 1.8
            elif 13 <= hour <= 21:  # New York session
                base_vol *= 1.5
            
            # Generate price movement with trend
            random_move = np.random.normal(0, base_vol)
            trend_component = np.sin(time.time() / 1000) * base_vol * 0.2
            mean_reversion = -(self.current_prices[symbol] - self.base_prices[symbol]) / self.base_prices[symbol] * 0.001
            
            total_move = random_move + trend_component + mean_reversion
            self.current_prices[symbol] += total_move
            
            # Create realistic spread
            if 'JPY' in symbol:
                spread = 0.012 + np.random.uniform(-0.003, 0.005)  # 1.2 ± 0.3 pips
            else:
                spread = 0.00012 + np.random.uniform(-0.00003, 0.00005)  # 1.2 ± 0.3 pips
            
            # Create tick data
            tick = {
                'timestamp': time.time(),
                'symbol': symbol,
                'bid': self.current_prices[symbol],
                'ask': self.current_prices[symbol] + spread,
                'spread': spread
            }
            
            self.price_history[symbol].append(tick)
    
    def analyze_market_signal(self, symbol: str) -> tuple:
        """Generate trading signal with clear entry logic"""
        
        price_data = self.price_history[symbol]
        if len(price_data) < 50:
            return 0.0, 0.0
        
        # Extract prices
        prices = [tick['bid'] for tick in list(price_data)[-50:]]
        
        # Multiple signal analysis
        signals = []
        
        # 1. Momentum Signal (short vs medium term)
        if len(prices) >= 10:
            short_avg = np.mean(prices[-3:])   # Last 3 ticks
            medium_avg = np.mean(prices[-10:]) # Last 10 ticks
            momentum = (short_avg - medium_avg) / medium_avg
            signals.append(momentum * 500)  # Scale for sensitivity
        
        # 2. Trend Signal (medium vs long term)
        if len(prices) >= 20:
            medium_avg = np.mean(prices[-10:])
            long_avg = np.mean(prices[-20:])
            trend = (medium_avg - long_avg) / long_avg
            signals.append(trend * 300)
        
        # 3. Volatility Breakout Signal
        if len(prices) >= 15:
            recent_vol = np.std(prices[-5:])
            historical_vol = np.std(prices[-15:])
            
            if historical_vol > 0:
                vol_ratio = recent_vol / historical_vol
                if vol_ratio > 1.5:  # High volatility breakout
                    recent_direction = prices[-1] - prices[-5]
                    breakout_signal = (recent_direction / prices[-5]) * 200
                    signals.append(breakout_signal)
        
        # 4. Mean Reversion Signal
        if len(prices) >= 20:
            current_price = prices[-1]
            sma_20 = np.mean(prices[-20:])
            std_20 = np.std(prices[-20:])
            
            if std_20 > 0:
                z_score = (current_price - sma_20) / std_20
                # Mean reversion: extreme values tend to revert
                if abs(z_score) > 1.5:
                    reversion_signal = -z_score * 0.3  # Opposite direction
                    signals.append(reversion_signal)
        
        # Combine signals
        if not signals:
            return 0.0, 0.0
        
        # Weight recent signals more heavily
        if len(signals) > 1:
            weights = [0.4, 0.3, 0.2, 0.1][:len(signals)]  # Favor recent signals
            direction = sum(s * w for s, w in zip(signals, weights))
        else:
            direction = signals[0]
        
        # Calculate confidence based on signal strength and agreement
        signal_strength = abs(direction)
        
        # Agreement factor (how much signals agree)
        if len(signals) > 1:
            positive_signals = sum(1 for s in signals if s > 0)
            negative_signals = sum(1 for s in signals if s < 0)
            total_signals = len(signals)
            agreement = max(positive_signals, negative_signals) / total_signals
        else:
            agreement = 1.0
        
        # Base confidence with bonuses
        base_confidence = 0.30  # 30% base
        strength_bonus = min(signal_strength * 0.5, 0.35)  # Up to 35% for strength
        agreement_bonus = agreement * 0.25  # Up to 25% for agreement
        
        confidence = base_confidence + strength_bonus + agreement_bonus
        confidence = min(confidence, 0.95)  # Cap at 95%
        
        # Normalize direction
        direction = np.tanh(direction)
        
        return direction, confidence
    
    def calculate_position_size(self, symbol: str, confidence: float) -> float:
        """Calculate optimal position size"""
        
        # Base risk calculation
        risk_amount = self.current_balance * self.config['risk_per_trade']
        
        # Adjust for confidence
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5x to 1.0x
        adjusted_risk = risk_amount * confidence_multiplier
        
        # Calculate size based on stop loss
        pip_value = 0.0001 if 'JPY' not in symbol else 0.01
        stop_loss_value = self.config['stop_loss_pips'] * pip_value * 100000  # Per lot
        
        position_size = adjusted_risk / stop_loss_value
        
        # Apply limits
        position_size = max(self.config['min_position_size'], 
                           min(position_size, self.config['max_position_size']))
        
        return round(position_size, 2)
    
    def can_open_position(self, symbol: str, confidence: float) -> tuple:
        """Check if we can open a new position"""
        
        # Check confidence threshold
        if confidence < self.config['min_confidence']:
            return False, f"Confidence {confidence:.1%} below minimum {self.config['min_confidence']:.1%}"
        
        # Check maximum positions
        if len(self.open_positions) >= self.config['max_positions']:
            return False, f"Maximum positions reached ({self.config['max_positions']})"
        
        # Check if we already have position in this symbol
        existing_positions = [p for p in self.open_positions if p.symbol == symbol]
        if len(existing_positions) >= 2:  # Max 2 positions per symbol
            return False, f"Maximum positions for {symbol} reached (2)"
        
        # Check daily loss limit
        if self.daily_pnl < -500:  # $500 daily loss limit
            return False, "Daily loss limit exceeded"
        
        # Check if trading is enabled
        if not self.trading_enabled:
            return False, "Trading disabled"
        
        return True, "Position can be opened"
    
    def open_position(self, symbol: str, direction: float, confidence: float) -> bool:
        """Open a new trading position"""
        
        # Get current market data
        latest_tick = self.price_history[symbol][-1]
        
        # Determine direction
        is_buy = direction > 0
        direction_str = "BUY" if is_buy else "SELL"
        
        # Calculate position size
        size = self.calculate_position_size(symbol, confidence)
        
        # Determine entry price
        entry_price = latest_tick['ask'] if is_buy else latest_tick['bid']
        
        # Calculate stop loss and take profit
        pip_value = 0.0001 if 'JPY' not in symbol else 0.01
        
        if is_buy:
            stop_loss = entry_price - (self.config['stop_loss_pips'] * pip_value)
            take_profit = entry_price + (self.config['take_profit_pips'] * pip_value)
        else:
            stop_loss = entry_price + (self.config['stop_loss_pips'] * pip_value)
            take_profit = entry_price - (self.config['take_profit_pips'] * pip_value)
        
        # Create position object
        position = Position(
            symbol=symbol,
            direction=direction_str,
            size=size,
            entry_price=entry_price,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Add to active positions
        self.open_positions.append(position)
        
        # Update statistics
        self.stats['positions_opened'] += 1
        self.stats['total_trades'] += 1
        
        print(f"✅ Position opened successfully! Total open: {len(self.open_positions)}")
        
        return True
    
    def update_all_positions(self):
        """Update all open positions and handle exits"""
        
        if not self.open_positions:
            return
        
        positions_to_remove = []
        
        for position in self.open_positions:
            # Get current market data for this symbol
            latest_tick = self.price_history[position.symbol][-1]
            
            # Update position
            closed_position = position.update(latest_tick['bid'], latest_tick['ask'])
            
            if closed_position:
                # Position closed, add to closed positions
                self.closed_positions.append(closed_position)
                self.position_history.append(closed_position)
                positions_to_remove.append(position)
                
                # Update account balance
                self.current_balance += closed_position['pnl']
                self.total_pnl += closed_position['pnl']
                self.daily_pnl += closed_position['pnl']
                
                # Update statistics
                self.stats['positions_closed'] += 1
                if closed_position['pnl'] > 0:
                    self.stats['winning_trades'] += 1
                else:
                    self.stats['losing_trades'] += 1
        
        # Remove closed positions
        for position in positions_to_remove:
            self.open_positions.remove(position)
    
    def print_status(self):
        """Print current trading status"""
        
        runtime = (datetime.now() - self.stats['start_time']).total_seconds() / 60
        win_rate = self.stats['winning_trades'] / max(self.stats['total_trades'], 1)
        
        print(f"\n📊 TRADING STATUS - {datetime.now().strftime('%H:%M:%S')}")
        print(f"Runtime: {runtime:.1f}m | Open Positions: {len(self.open_positions)}")
        print(f"Total Trades: {self.stats['total_trades']} | Win Rate: {win_rate:.1%}")
        print(f"Balance: ${self.current_balance:.2f} | P&L: ${self.total_pnl:.2f}")
        
        if self.open_positions:
            print("Open Positions:")
            for pos in self.open_positions:
                age = (datetime.now() - pos.entry_time).total_seconds() / 60
                print(f"  {pos.symbol} {pos.direction} {pos.size} | Age: {age:.1f}m | Unrealized P&L: ${pos.pnl:.2f}")
    
    async def run_trading_session(self, minutes: float = 10.0):
        """Run complete trading session"""
        
        print(f"🚀 ULTRA TRADING AGENT - STARTING SESSION")
        print(f"Duration: {minutes:.1f} minutes")
        print(f"Configuration: {self.config}")
        print("=" * 60)
        
        self.running = True
        end_time = datetime.now() + timedelta(minutes=minutes)
        
        # Generate initial market data
        print("📊 Generating initial market data...")
        for _ in range(100):
            self.generate_market_data()
            await asyncio.sleep(0.001)  # 1ms between ticks
        
        print("✅ Market data ready, starting trading...")
        
        cycle_count = 0
        last_status_time = datetime.now()
        
        while datetime.now() < end_time and self.running:
            try:
                cycle_count += 1
                
                # 1. Generate new market data
                self.generate_market_data()
                
                # 2. Update existing positions first
                self.update_all_positions()
                
                # 3. Look for new trading opportunities
                for symbol in self.symbols:
                    
                    # Analyze market
                    direction, confidence = self.analyze_market_signal(symbol)
                    
                    # Count opportunities
                    if confidence > 0.35:
                        self.stats['opportunities_found'] += 1
                    
                    # Check if we can and should open position
                    if confidence >= self.config['min_confidence']:
                        
                        can_open, reason = self.can_open_position(symbol, confidence)
                        
                        if can_open:
                            print(f"\n🎯 TRADING OPPORTUNITY: {symbol}")
                            print(f"   Direction: {direction:.3f} ({'BUY' if direction > 0 else 'SELL'})")
                            print(f"   Confidence: {confidence:.1%}")
                            
                            # Open position
                            if self.open_position(symbol, direction, confidence):
                                print(f"   🚀 Position opened successfully!")
                        # else:
                        #     print(f"   ⏸️ Cannot open position: {reason}")
                
                # 4. Print status every 30 seconds
                if (datetime.now() - last_status_time).total_seconds() > 30:
                    self.print_status()
                    last_status_time = datetime.now()
                
                # 5. Sleep between cycles
                await asyncio.sleep(0.05)  # 50ms cycle time
                
            except Exception as e:
                print(f"❌ Trading cycle error: {e}")
                await asyncio.sleep(1)
        
        # Close any remaining positions
        print("\n⏹️ Session ending, closing all positions...")
        for position in self.open_positions[:]:  # Copy list
            # Force close at current market price
            latest_tick = self.price_history[position.symbol][-1]
            closed_pos = position.update(latest_tick['bid'], latest_tick['ask'])
            if not closed_pos:  # Force close if not naturally closed
                position.status = "CLOSED_TIME"
                closed_pos = position.update(latest_tick['bid'], latest_tick['ask'])
            
            if closed_pos:
                self.closed_positions.append(closed_pos)
                self.current_balance += closed_pos['pnl']
                self.total_pnl += closed_pos['pnl']
        
        self.open_positions.clear()
        
        await self.generate_final_report()
    
    async def generate_final_report(self):
        """Generate comprehensive final report"""
        
        runtime = (datetime.now() - self.stats['start_time']).total_seconds() / 60
        
        print(f"""
🎯 ULTRA TRADING AGENT - FINAL REPORT
=====================================
Session Duration: {runtime:.1f} minutes
Starting Balance: ${self.starting_balance:.2f}
Final Balance: ${self.current_balance:.2f}
Total P&L: ${self.total_pnl:.2f}
ROI: {(self.total_pnl/self.starting_balance)*100:.2f}%

TRADING STATISTICS:
Opportunities Found: {self.stats['opportunities_found']}
Positions Opened: {self.stats['positions_opened']}
Positions Closed: {self.stats['positions_closed']}
Winning Trades: {self.stats['winning_trades']}
Losing Trades: {self.stats['losing_trades']}
Win Rate: {(self.stats['winning_trades']/max(self.stats['total_trades'],1)):.1%}

POSITION ANALYSIS:""")
        
        if self.closed_positions:
            winning_trades = [p for p in self.closed_positions if p['pnl'] > 0]
            losing_trades = [p for p in self.closed_positions if p['pnl'] <= 0]
            
            if winning_trades:
                avg_win = np.mean([p['pnl'] for p in winning_trades])
                print(f"Average Winning Trade: ${avg_win:.2f}")
            
            if losing_trades:
                avg_loss = np.mean([p['pnl'] for p in losing_trades])
                print(f"Average Losing Trade: ${avg_loss:.2f}")
            
            avg_duration = np.mean([p['duration_minutes'] for p in self.closed_positions])
            print(f"Average Trade Duration: {avg_duration:.1f} minutes")
            
            print("\nRECENT TRADES:")
            for trade in self.closed_positions[-5:]:
                print(f"  {trade['entry_time'].strftime('%H:%M:%S')} | "
                      f"{trade['symbol']} {trade['direction']} | "
                      f"${trade['pnl']:>7.2f} | {trade['exit_reason']}")
        
        # Assessment
        print("\nPERFORMACE ASSESSMENT:")
        if self.stats['total_trades'] == 0:
            print("❌ NO TRADES EXECUTED - Check trading logic")
        elif self.stats['winning_trades'] / max(self.stats['total_trades'], 1) > 0.60 and self.total_pnl > 0:
            print("✅ EXCELLENT - Strong performance!")
        elif self.stats['total_trades'] > 0 and self.total_pnl >= 0:
            print("✅ GOOD - Positive results")
        elif self.stats['total_trades'] > 0:
            print("⚠️ MIXED - Some issues to address")
        else:
            print("❌ POOR - Major issues detected")
        
        # Save results
        results = {
            'session_info': {
                'start_time': self.stats['start_time'].isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_minutes': runtime,
                'starting_balance': self.starting_balance,
                'final_balance': self.current_balance,
                'total_pnl': self.total_pnl,
                'roi_percent': (self.total_pnl/self.starting_balance)*100
            },
            'statistics': self.stats,
            'configuration': self.config,
            'closed_positions': self.closed_positions,
            'symbols_traded': self.symbols
        }
        
        filename = f'ultra_trading_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📊 Detailed results saved to: {filename}")

# Quick test function
async def quick_ultra_test():
    """Quick test of ultra trading agent"""
    engine = UltraTradingEngine()
    await engine.run_trading_session(minutes=5)

if __name__ == "__main__":
    print("""
🚀 ULTRA TRADING AGENT
======================
Bulletproof position management with guaranteed execution

Options:
1. Quick Test (5 minutes) - Recommended
2. Standard Session (10 minutes)
3. Extended Session (20 minutes)
4. Marathon Session (60 minutes)
    """)
    
    choice = input("Select option (1-4): ").strip()
    
    engine = UltraTradingEngine()
    
    if choice == "1":
        print("🚀 Starting 5-minute quick test...")
        asyncio.run(engine.run_trading_session(minutes=5))
    elif choice == "2":
        print("🚀 Starting 10-minute standard session...")
        asyncio.run(engine.run_trading_session(minutes=10))
    elif choice == "3":
        print("🚀 Starting 20-minute extended session...")
        asyncio.run(engine.run_trading_session(minutes=20))
    elif choice == "4":
        print("🚀 Starting 60-minute marathon session...")
        asyncio.run(engine.run_trading_session(minutes=60))
    else:
        print("Invalid choice. Please run again.")