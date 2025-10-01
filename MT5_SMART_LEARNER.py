"""
MT5 SMART LEARNER - Advanced Self-Learning Trading Bot
=======================================================
Realistic learning system that adapts and improves continuously
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ADAPTIVE CONFIGURATION
# ============================================================================

class AdaptiveConfig:
    """Configuration that adapts based on performance"""
    
    # Account
    LOGIN = 95948709
    PASSWORD = 'To-4KyLg'
    SERVER = 'MetaQuotes-Demo'
    MAGIC = 999999
    
    # Adaptive Parameters (will change based on learning)
    def __init__(self):
        self.load_config()
    
    def load_config(self):
        """Load saved configuration or use defaults"""
        if os.path.exists('adaptive_config.json'):
            with open('adaptive_config.json', 'r') as f:
                saved = json.load(f)
                self.__dict__.update(saved)
        else:
            # Default starting values
            self.SCAN_INTERVAL = 2  # seconds
            self.MAX_POSITIONS = 3  # Start conservative
            self.BASE_LOT = 0.01  # Start small
            self.MIN_CONFIDENCE = 0.65  # Will adapt
            self.MIN_SIGNALS = 2  # Will adapt
            self.STOP_LOSS_PIPS = 15  # Will adapt
            self.TAKE_PROFIT_PIPS = 10  # Will adapt
            self.MAX_SPREAD_PIPS = 3  # Realistic for most pairs
            self.TRAILING_START_PIPS = 5
            self.TRAILING_DISTANCE_PIPS = 3
    
    def save_config(self):
        """Save current configuration"""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open('adaptive_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def adapt(self, performance):
        """Adapt configuration based on performance"""
        if performance['total_trades'] < 10:
            return  # Need more data
        
        win_rate = performance['wins'] / max(1, performance['total_trades'])
        
        # Adapt based on win rate
        if win_rate < 0.3:
            # Losing too much - be more conservative
            self.MIN_CONFIDENCE = min(0.85, self.MIN_CONFIDENCE + 0.05)
            self.MIN_SIGNALS = min(4, self.MIN_SIGNALS + 1)
            self.STOP_LOSS_PIPS = max(10, self.STOP_LOSS_PIPS - 2)
            self.BASE_LOT = max(0.01, self.BASE_LOT - 0.01)
            print(f"[ADAPT] Tightening requirements - Win rate too low: {win_rate:.1%}")
        
        elif win_rate > 0.6:
            # Winning well - can be slightly more aggressive
            self.MIN_CONFIDENCE = max(0.60, self.MIN_CONFIDENCE - 0.02)
            self.MAX_POSITIONS = min(5, self.MAX_POSITIONS + 1)
            if performance['total_profit'] > 50:
                self.BASE_LOT = min(0.05, self.BASE_LOT + 0.01)
            print(f"[ADAPT] Loosening requirements - Win rate good: {win_rate:.1%}")
        
        # Adapt risk/reward based on average win/loss
        if performance['avg_win'] > 0 and performance['avg_loss'] < 0:
            current_rr = abs(performance['avg_win'] / performance['avg_loss'])
            if current_rr < 1.2:
                # Need better risk/reward
                self.TAKE_PROFIT_PIPS = min(25, self.TAKE_PROFIT_PIPS + 2)
            elif current_rr > 2:
                # Risk/reward is good, can take smaller wins
                self.TAKE_PROFIT_PIPS = max(8, self.TAKE_PROFIT_PIPS - 1)
        
        self.save_config()

# ============================================================================
# LEARNING MEMORY SYSTEM
# ============================================================================

class LearningMemory:
    """Advanced memory system that learns from all trades"""
    
    def __init__(self):
        self.trade_history = deque(maxlen=10000)
        self.pattern_success = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pips': 0})
        self.timeframe_performance = defaultdict(lambda: {'trades': 0, 'profit': 0})
        self.symbol_performance = defaultdict(lambda: {'trades': 0, 'profit': 0, 'avg_spread': 0})
        self.indicator_weights = self.load_weights()
        self.market_conditions = {}
        
    def load_weights(self):
        """Load indicator weights from previous learning"""
        if os.path.exists('indicator_weights.json'):
            with open('indicator_weights.json', 'r') as f:
                return json.load(f)
        
        # Default weights
        return {
            'rsi': 1.0,
            'macd': 1.0,
            'ema_cross': 1.0,
            'support_resistance': 1.0,
            'momentum': 1.0,
            'volume': 1.0,
            'pattern': 1.0,
            'structure': 1.0,
            'bollinger': 1.0,
            'stochastic': 1.0
        }
    
    def save_weights(self):
        """Save learned weights"""
        with open('indicator_weights.json', 'w') as f:
            json.dump(self.indicator_weights, f, indent=2)
    
    def record_trade(self, trade_data):
        """Record a trade and learn from it"""
        self.trade_history.append(trade_data)
        
        # Update pattern success
        pattern_key = f"{trade_data['setup']}_{trade_data['timeframe']}"
        if trade_data['profit'] > 0:
            self.pattern_success[pattern_key]['wins'] += 1
        else:
            self.pattern_success[pattern_key]['losses'] += 1
        self.pattern_success[pattern_key]['total_pips'] += trade_data['pips']
        
        # Update timeframe performance
        hour = trade_data['entry_time'].hour
        self.timeframe_performance[hour]['trades'] += 1
        self.timeframe_performance[hour]['profit'] += trade_data['profit']
        
        # Update symbol performance
        self.symbol_performance[trade_data['symbol']]['trades'] += 1
        self.symbol_performance[trade_data['symbol']]['profit'] += trade_data['profit']
        
        # Adjust indicator weights based on success
        if 'indicators_used' in trade_data:
            for indicator in trade_data['indicators_used']:
                if trade_data['profit'] > 0:
                    self.indicator_weights[indicator] = min(2.0, self.indicator_weights[indicator] * 1.02)
                else:
                    self.indicator_weights[indicator] = max(0.5, self.indicator_weights[indicator] * 0.98)
        
        self.save_weights()
        self.save_memory()
    
    def get_pattern_confidence(self, pattern_key):
        """Get confidence for a pattern based on history"""
        if pattern_key not in self.pattern_success:
            return 0.5  # Neutral for unknown patterns
        
        stats = self.pattern_success[pattern_key]
        total = stats['wins'] + stats['losses']
        
        if total < 3:
            return 0.5  # Not enough data
        
        win_rate = stats['wins'] / total
        avg_pips = stats['total_pips'] / total
        
        # Combine win rate and profitability
        confidence = (win_rate * 0.7) + (min(avg_pips / 10, 0.3))
        
        return max(0.1, min(0.9, confidence))
    
    def get_best_hours(self):
        """Get the most profitable trading hours"""
        sorted_hours = sorted(
            self.timeframe_performance.items(),
            key=lambda x: x[1]['profit'] / max(1, x[1]['trades']),
            reverse=True
        )
        
        # Return top 8 hours
        return [hour for hour, _ in sorted_hours[:8]]
    
    def get_symbol_rating(self, symbol):
        """Rate a symbol based on past performance"""
        if symbol not in self.symbol_performance:
            return 1.0  # Neutral for new symbols
        
        stats = self.symbol_performance[symbol]
        if stats['trades'] < 5:
            return 1.0
        
        avg_profit = stats['profit'] / stats['trades']
        
        # Rating based on average profit
        if avg_profit > 5:
            return 1.5
        elif avg_profit > 0:
            return 1.2
        elif avg_profit > -5:
            return 0.8
        else:
            return 0.5
    
    def save_memory(self):
        """Save memory to disk"""
        memory_data = {
            'trade_history': list(self.trade_history)[-1000:],  # Last 1000 trades
            'pattern_success': dict(self.pattern_success),
            'timeframe_performance': dict(self.timeframe_performance),
            'symbol_performance': dict(self.symbol_performance),
            'market_conditions': self.market_conditions
        }
        
        with open('learning_memory.json', 'w') as f:
            json.dump(memory_data, f, default=str)
    
    def load_memory(self):
        """Load memory from disk"""
        if os.path.exists('learning_memory.json'):
            try:
                with open('learning_memory.json', 'r') as f:
                    data = json.load(f)
                    
                    self.pattern_success = defaultdict(
                        lambda: {'wins': 0, 'losses': 0, 'total_pips': 0},
                        data.get('pattern_success', {})
                    )
                    self.timeframe_performance = defaultdict(
                        lambda: {'trades': 0, 'profit': 0},
                        data.get('timeframe_performance', {})
                    )
                    self.symbol_performance = defaultdict(
                        lambda: {'trades': 0, 'profit': 0, 'avg_spread': 0},
                        data.get('symbol_performance', {})
                    )
                    self.market_conditions = data.get('market_conditions', {})
                    
                    print(f"[MEMORY] Loaded {len(self.pattern_success)} patterns, "
                          f"{len(self.symbol_performance)} symbols")
            except Exception as e:
                print(f"[MEMORY] Error loading: {e}")

# ============================================================================
# SMART ANALYSIS ENGINE
# ============================================================================

class SmartAnalyzer:
    """Intelligent market analysis with learning"""
    
    def __init__(self, memory):
        self.memory = memory
        self.config = AdaptiveConfig()
        
    def analyze_symbol(self, symbol):
        """Complete analysis with weighted indicators"""
        
        # Get data
        m1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 100)
        m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
        m15 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 50)
        
        if m5 is None or len(m5) < 100:
            return None
        
        df = pd.DataFrame(m5)
        
        # Calculate all indicators
        indicators = self.calculate_indicators(df)
        
        # Get weighted signals
        signals = []
        indicators_used = []
        
        # RSI Signal
        if indicators['rsi'] < 30:
            signals.append({
                'type': 'BUY',
                'strength': (30 - indicators['rsi']) / 30 * self.memory.indicator_weights['rsi'],
                'indicator': 'rsi'
            })
            indicators_used.append('rsi')
        elif indicators['rsi'] > 70:
            signals.append({
                'type': 'SELL',
                'strength': (indicators['rsi'] - 70) / 30 * self.memory.indicator_weights['rsi'],
                'indicator': 'rsi'
            })
            indicators_used.append('rsi')
        
        # MACD Signal
        if indicators['macd'] > indicators['macd_signal']:
            signals.append({
                'type': 'BUY',
                'strength': min(abs(indicators['macd'] - indicators['macd_signal']) * 100, 1.0) * 
                           self.memory.indicator_weights['macd'],
                'indicator': 'macd'
            })
            indicators_used.append('macd')
        elif indicators['macd'] < indicators['macd_signal']:
            signals.append({
                'type': 'SELL',
                'strength': min(abs(indicators['macd'] - indicators['macd_signal']) * 100, 1.0) * 
                           self.memory.indicator_weights['macd'],
                'indicator': 'macd'
            })
            indicators_used.append('macd')
        
        # EMA Cross
        if indicators['ema_fast'] > indicators['ema_slow']:
            signals.append({
                'type': 'BUY',
                'strength': min((indicators['ema_fast'] - indicators['ema_slow']) / indicators['ema_slow'] * 100, 1.0) * 
                           self.memory.indicator_weights['ema_cross'],
                'indicator': 'ema_cross'
            })
            indicators_used.append('ema_cross')
        else:
            signals.append({
                'type': 'SELL',
                'strength': min((indicators['ema_slow'] - indicators['ema_fast']) / indicators['ema_slow'] * 100, 1.0) * 
                           self.memory.indicator_weights['ema_cross'],
                'indicator': 'ema_cross'
            })
            indicators_used.append('ema_cross')
        
        # Bollinger Bands
        if indicators['bb_position'] < 0.2:
            signals.append({
                'type': 'BUY',
                'strength': (0.2 - indicators['bb_position']) * 5 * self.memory.indicator_weights['bollinger'],
                'indicator': 'bollinger'
            })
            indicators_used.append('bollinger')
        elif indicators['bb_position'] > 0.8:
            signals.append({
                'type': 'SELL',
                'strength': (indicators['bb_position'] - 0.8) * 5 * self.memory.indicator_weights['bollinger'],
                'indicator': 'bollinger'
            })
            indicators_used.append('bollinger')
        
        # Stochastic
        if indicators['stochastic'] < 20:
            signals.append({
                'type': 'BUY',
                'strength': (20 - indicators['stochastic']) / 20 * self.memory.indicator_weights['stochastic'],
                'indicator': 'stochastic'
            })
            indicators_used.append('stochastic')
        elif indicators['stochastic'] > 80:
            signals.append({
                'type': 'SELL',
                'strength': (indicators['stochastic'] - 80) / 20 * self.memory.indicator_weights['stochastic'],
                'indicator': 'stochastic'
            })
            indicators_used.append('stochastic')
        
        # Support/Resistance
        sr_signal = self.check_support_resistance(df)
        if sr_signal:
            signals.append({
                'type': sr_signal['type'],
                'strength': sr_signal['strength'] * self.memory.indicator_weights['support_resistance'],
                'indicator': 'support_resistance'
            })
            indicators_used.append('support_resistance')
        
        # Pattern Recognition
        pattern = self.find_pattern(df)
        if pattern:
            signals.append({
                'type': pattern['type'],
                'strength': pattern['strength'] * self.memory.indicator_weights['pattern'],
                'indicator': 'pattern'
            })
            indicators_used.append('pattern')
        
        # Aggregate signals
        if not signals:
            return None
        
        buy_strength = sum(s['strength'] for s in signals if s['type'] == 'BUY')
        sell_strength = sum(s['strength'] for s in signals if s['type'] == 'SELL')
        
        # Get symbol rating
        symbol_rating = self.memory.get_symbol_rating(symbol)
        
        # Determine final signal
        if buy_strength > sell_strength:
            confidence = min(0.95, (buy_strength / len(signals)) * symbol_rating)
            setup = self.determine_setup(indicators)
            
            return {
                'type': 'BUY',
                'confidence': confidence,
                'setup': setup,
                'indicators_used': indicators_used,
                'buy_strength': buy_strength,
                'sell_strength': sell_strength,
                'num_signals': len([s for s in signals if s['type'] == 'BUY'])
            }
        
        elif sell_strength > buy_strength:
            confidence = min(0.95, (sell_strength / len(signals)) * symbol_rating)
            setup = self.determine_setup(indicators)
            
            return {
                'type': 'SELL',
                'confidence': confidence,
                'setup': setup,
                'indicators_used': indicators_used,
                'buy_strength': buy_strength,
                'sell_strength': sell_strength,
                'num_signals': len([s for s in signals if s['type'] == 'SELL'])
            }
        
        return None
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators"""
        
        indicators = {}
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = float(100 - (100 / (1 + rs)).iloc[-1])
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        indicators['macd'] = float(macd.iloc[-1])
        indicators['macd_signal'] = float(signal.iloc[-1])
        
        # EMAs
        indicators['ema_fast'] = float(df['close'].ewm(span=12).mean().iloc[-1])
        indicators['ema_slow'] = float(df['close'].ewm(span=26).mean().iloc[-1])
        
        # Bollinger Bands
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        upper_band = sma20 + 2 * std20
        lower_band = sma20 - 2 * std20
        bb_position = (df['close'].iloc[-1] - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
        indicators['bb_position'] = float(bb_position) if not pd.isna(bb_position) else 0.5
        
        # Stochastic
        low_min = df['low'].rolling(14).min()
        high_max = df['high'].rolling(14).max()
        stoch = 100 * ((df['close'] - low_min) / (high_max - low_min))
        indicators['stochastic'] = float(stoch.iloc[-1]) if not pd.isna(stoch.iloc[-1]) else 50
        
        # ATR for volatility
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        indicators['atr'] = float(true_range.rolling(14).mean().iloc[-1])
        
        # Momentum
        indicators['momentum'] = float((df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] * 100)
        
        # Volume
        indicators['volume_ratio'] = float(df['tick_volume'].iloc[-1] / df['tick_volume'].rolling(20).mean().iloc[-1])
        
        return indicators
    
    def check_support_resistance(self, df):
        """Check for support and resistance levels"""
        
        # Find recent highs and lows
        window = 20
        resistance = df['high'].rolling(window).max().iloc[-1]
        support = df['low'].rolling(window).min().iloc[-1]
        current = df['close'].iloc[-1]
        
        # Check distance to levels
        dist_to_resistance = (resistance - current) / current
        dist_to_support = (current - support) / current
        
        # Near support and bouncing
        if dist_to_support < 0.002 and df['close'].iloc[-1] > df['close'].iloc[-2]:
            return {'type': 'BUY', 'strength': 0.8}
        
        # Near resistance and rejecting
        elif dist_to_resistance < 0.002 and df['close'].iloc[-1] < df['close'].iloc[-2]:
            return {'type': 'SELL', 'strength': 0.8}
        
        return None
    
    def find_pattern(self, df):
        """Find candlestick patterns"""
        
        # Hammer/Pin Bar
        body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
        upper_wick = df['high'].iloc[-1] - max(df['close'].iloc[-1], df['open'].iloc[-1])
        lower_wick = min(df['close'].iloc[-1], df['open'].iloc[-1]) - df['low'].iloc[-1]
        
        # Bullish hammer
        if lower_wick > body * 2 and upper_wick < body * 0.5:
            return {'type': 'BUY', 'strength': 0.7}
        
        # Bearish shooting star
        elif upper_wick > body * 2 and lower_wick < body * 0.5:
            return {'type': 'SELL', 'strength': 0.7}
        
        # Engulfing patterns
        if len(df) >= 2:
            prev_body = abs(df['close'].iloc[-2] - df['open'].iloc[-2])
            curr_body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
            
            # Bullish engulfing
            if (df['close'].iloc[-2] < df['open'].iloc[-2] and  # Previous bearish
                df['close'].iloc[-1] > df['open'].iloc[-1] and  # Current bullish
                curr_body > prev_body * 1.5):  # Current engulfs previous
                return {'type': 'BUY', 'strength': 0.75}
            
            # Bearish engulfing
            elif (df['close'].iloc[-2] > df['open'].iloc[-2] and  # Previous bullish
                  df['close'].iloc[-1] < df['open'].iloc[-1] and  # Current bearish
                  curr_body > prev_body * 1.5):  # Current engulfs previous
                return {'type': 'SELL', 'strength': 0.75}
        
        return None
    
    def determine_setup(self, indicators):
        """Determine the trading setup type"""
        
        if abs(indicators['momentum']) > 0.5:
            return 'momentum'
        elif indicators['rsi'] < 35 or indicators['rsi'] > 65:
            return 'reversal'
        elif abs(indicators['ema_fast'] - indicators['ema_slow']) / indicators['ema_slow'] > 0.002:
            return 'trend'
        else:
            return 'range'

# ============================================================================
# SMART LEARNER TRADER
# ============================================================================

class SmartLearnerTrader:
    """Main trading system with advanced learning"""
    
    def __init__(self):
        self.config = AdaptiveConfig()
        self.memory = LearningMemory()
        self.memory.load_memory()
        self.analyzer = SmartAnalyzer(self.memory)
        
        self.performance = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_profit': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'consecutive_losses': 0,
            'consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'max_consecutive_wins': 0
        }
        
        self.active_trades = {}
        self.start_balance = 0
        self.session_high = 0
        self.session_low = 0
        
        # All forex pairs
        self.symbols = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
            'NZDUSD', 'USDCHF', 'EURJPY', 'GBPJPY', 'EURGBP',
            'EURAUD', 'EURCAD', 'GBPAUD', 'GBPCAD', 'GBPNZD'
        ]
    
    def initialize(self):
        """Initialize MT5 connection"""
        
        if not mt5.initialize():
            print("[ERROR] Failed to initialize MT5")
            return False
        
        if not mt5.login(self.config.LOGIN, self.config.PASSWORD, self.config.SERVER):
            print("[ERROR] Failed to login")
            mt5.shutdown()
            return False
        
        account = mt5.account_info()
        if account:
            self.start_balance = account.balance
            self.session_high = account.balance
            self.session_low = account.balance
            
            print(f"\n{'='*70}")
            print(f"MT5 SMART LEARNER - ADAPTIVE TRADING SYSTEM")
            print(f"{'='*70}")
            print(f"Account Balance: ${account.balance:.2f}")
            print(f"Learned Patterns: {len(self.memory.pattern_success)}")
            print(f"Symbol History: {len(self.memory.symbol_performance)} symbols")
            print(f"\nAdaptive Settings:")
            print(f"  Min Confidence: {self.config.MIN_CONFIDENCE:.0%}")
            print(f"  Min Signals: {self.config.MIN_SIGNALS}")
            print(f"  Risk/Reward: {self.config.STOP_LOSS_PIPS}:{self.config.TAKE_PROFIT_PIPS}")
            print(f"  Max Spread: {self.config.MAX_SPREAD_PIPS} pips")
            print(f"{'='*70}\n")
            
            return True
        
        return False
    
    def should_trade_now(self):
        """Determine if current conditions are good for trading"""
        
        current_hour = datetime.now().hour
        
        # Use learned best hours if available
        best_hours = self.memory.get_best_hours()
        if best_hours and len(best_hours) >= 4:
            if current_hour not in best_hours:
                return False
        
        # Check recent performance
        if self.performance['consecutive_losses'] >= 3:
            # Take a break after 3 losses
            return False
        
        return True
    
    def check_spread(self, symbol):
        """Check if spread is acceptable"""
        
        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        
        if not tick or not info:
            return False
        
        spread_points = tick.ask - tick.bid
        spread_pips = spread_points / (info.point * 10)
        
        # Update symbol average spread
        if symbol in self.memory.symbol_performance:
            old_avg = self.memory.symbol_performance[symbol]['avg_spread']
            trades = self.memory.symbol_performance[symbol]['trades']
            new_avg = (old_avg * trades + spread_pips) / (trades + 1)
            self.memory.symbol_performance[symbol]['avg_spread'] = new_avg
        
        # Dynamic spread tolerance based on symbol
        max_spread = self.config.MAX_SPREAD_PIPS
        
        # Adjust for specific pairs
        if 'JPY' in symbol:
            max_spread *= 1.5  # JPY pairs often have higher spreads
        if symbol in ['GBPNZD', 'GBPAUD', 'EURAUD']:
            max_spread *= 2  # Exotic pairs have higher spreads
        
        return spread_pips <= max_spread
    
    def execute_trade(self, symbol, signal):
        """Execute trade with learning"""
        
        # Check spread
        if not self.check_spread(symbol):
            return False
        
        # Get market info
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        
        if not info or not tick:
            return False
        
        # Check pattern confidence
        pattern_key = f"{signal['setup']}_{datetime.now().hour}"
        pattern_confidence = self.memory.get_pattern_confidence(pattern_key)
        
        # Combine confidences
        final_confidence = signal['confidence'] * pattern_confidence
        
        if final_confidence < self.config.MIN_CONFIDENCE:
            return False
        
        # Dynamic lot size based on confidence and performance
        lot_size = self.config.BASE_LOT
        if final_confidence > 0.8 and self.performance['consecutive_wins'] > 0:
            lot_size *= 1.5
        elif self.performance['consecutive_losses'] > 0:
            lot_size *= 0.5
        
        lot_size = round(lot_size, 2)
        lot_size = max(info.volume_min, min(lot_size, info.volume_max))
        
        # Setup trade
        if signal['type'] == 'BUY':
            price = tick.ask
            sl = price - self.config.STOP_LOSS_PIPS * info.point * 10
            tp = price + self.config.TAKE_PROFIT_PIPS * info.point * 10
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl = price + self.config.STOP_LOSS_PIPS * info.point * 10
            tp = price - self.config.TAKE_PROFIT_PIPS * info.point * 10
            order_type = mt5.ORDER_TYPE_SELL
        
        # Send order
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': lot_size,
            'type': order_type,
            'price': price,
            'sl': round(sl, info.digits),
            'tp': round(tp, info.digits),
            'deviation': 10,
            'magic': self.config.MAGIC,
            'comment': f"{signal['setup']}_{final_confidence:.2f}",
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            # Record trade
            self.active_trades[result.order] = {
                'symbol': symbol,
                'type': signal['type'],
                'setup': signal['setup'],
                'entry_price': price,
                'entry_time': datetime.now(),
                'sl': sl,
                'tp': tp,
                'lot_size': lot_size,
                'confidence': final_confidence,
                'indicators_used': signal.get('indicators_used', [])
            }
            
            self.performance['total_trades'] += 1
            
            print(f"\n[TRADE #{self.performance['total_trades']}] {signal['type']} {symbol}")
            print(f"  Setup: {signal['setup']} | Confidence: {final_confidence:.1%}")
            print(f"  Signals: {signal['num_signals']} | Lot: {lot_size}")
            print(f"  Entry: {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f}")
            
            return True
        
        return False
    
    def manage_positions(self):
        """Manage open positions with trailing stop"""
        
        positions = mt5.positions_get(magic=self.config.MAGIC)
        if not positions:
            return
        
        for pos in positions:
            info = mt5.symbol_info(pos.symbol)
            tick = mt5.symbol_info_tick(pos.symbol)
            
            if not info or not tick:
                continue
            
            # Calculate pips
            if pos.type == 0:  # BUY
                pips = (tick.bid - pos.price_open) / (info.point * 10)
            else:  # SELL
                pips = (pos.price_open - tick.ask) / (info.point * 10)
            
            # Trailing stop
            if pips >= self.config.TRAILING_START_PIPS:
                if pos.type == 0:  # BUY
                    new_sl = tick.bid - self.config.TRAILING_DISTANCE_PIPS * info.point * 10
                    if new_sl > pos.sl:
                        request = {
                            'action': mt5.TRADE_ACTION_SLTP,
                            'symbol': pos.symbol,
                            'position': pos.ticket,
                            'sl': round(new_sl, info.digits),
                            'tp': pos.tp,
                            'magic': self.config.MAGIC
                        }
                        mt5.order_send(request)
                else:  # SELL
                    new_sl = tick.ask + self.config.TRAILING_DISTANCE_PIPS * info.point * 10
                    if new_sl < pos.sl:
                        request = {
                            'action': mt5.TRADE_ACTION_SLTP,
                            'symbol': pos.symbol,
                            'position': pos.ticket,
                            'sl': round(new_sl, info.digits),
                            'tp': pos.tp,
                            'magic': self.config.MAGIC
                        }
                        mt5.order_send(request)
    
    def check_closed_trades(self):
        """Check and learn from closed trades"""
        
        # Get recent deals
        from_date = datetime.now() - timedelta(hours=24)
        deals = mt5.history_deals_get(from_date, datetime.now())
        
        if not deals:
            return
        
        for deal in deals:
            if deal.magic != self.config.MAGIC or deal.entry != 1:
                continue
            
            # Find in active trades
            for trade_id, trade_info in list(self.active_trades.items()):
                if deal.position_id in self.active_trades:
                    continue
                
                # This trade closed
                profit = deal.profit
                symbol = deal.symbol
                
                # Calculate pips
                info = mt5.symbol_info(symbol)
                if info:
                    if trade_info['type'] == 'BUY':
                        pips = (deal.price - trade_info['entry_price']) / (info.point * 10)
                    else:
                        pips = (trade_info['entry_price'] - deal.price) / (info.point * 10)
                else:
                    pips = 0
                
                # Update performance
                if profit > 0:
                    self.performance['wins'] += 1
                    self.performance['consecutive_wins'] += 1
                    self.performance['consecutive_losses'] = 0
                    
                    # Update average win
                    self.performance['avg_win'] = (
                        (self.performance['avg_win'] * (self.performance['wins'] - 1) + profit) /
                        self.performance['wins']
                    )
                    
                    print(f"[WIN] {symbol} +${profit:.2f} (+{pips:.1f} pips)")
                else:
                    self.performance['losses'] += 1
                    self.performance['consecutive_losses'] += 1
                    self.performance['consecutive_wins'] = 0
                    
                    # Update average loss
                    self.performance['avg_loss'] = (
                        (self.performance['avg_loss'] * (self.performance['losses'] - 1) + profit) /
                        self.performance['losses']
                    )
                    
                    print(f"[LOSS] {symbol} ${profit:.2f} ({pips:.1f} pips)")
                
                self.performance['total_profit'] += profit
                
                # Update max consecutive
                self.performance['max_consecutive_wins'] = max(
                    self.performance['max_consecutive_wins'],
                    self.performance['consecutive_wins']
                )
                self.performance['max_consecutive_losses'] = max(
                    self.performance['max_consecutive_losses'],
                    self.performance['consecutive_losses']
                )
                
                # Record to memory for learning
                trade_data = {
                    'symbol': symbol,
                    'type': trade_info['type'],
                    'setup': trade_info['setup'],
                    'profit': profit,
                    'pips': pips,
                    'entry_time': trade_info['entry_time'],
                    'exit_time': datetime.now(),
                    'timeframe': datetime.now().hour,
                    'indicators_used': trade_info.get('indicators_used', [])
                }
                
                self.memory.record_trade(trade_data)
                
                # Remove from active trades
                del self.active_trades[trade_id]
                break
    
    def display_status(self):
        """Display current status and learning progress"""
        
        account = mt5.account_info()
        if not account:
            return
        
        positions = mt5.positions_get(magic=self.config.MAGIC)
        open_positions = len(positions) if positions else 0
        
        # Update session high/low
        if account.balance > self.session_high:
            self.session_high = account.balance
        if account.balance < self.session_low:
            self.session_low = account.balance
        
        # Calculate metrics
        total_profit = account.balance - self.start_balance
        roi = (total_profit / self.start_balance * 100) if self.start_balance > 0 else 0
        
        win_rate = 0
        if self.performance['total_trades'] > 0:
            win_rate = self.performance['wins'] / self.performance['total_trades'] * 100
        
        print(f"\n{'='*70}")
        print(f"SMART LEARNER STATUS")
        print(f"{'='*70}")
        print(f"Balance: ${account.balance:.2f} | Equity: ${account.equity:.2f}")
        print(f"Session P&L: ${total_profit:+.2f} ({roi:+.1f}%)")
        print(f"Session High: ${self.session_high:.2f} | Low: ${self.session_low:.2f}")
        
        print(f"\nPerformance:")
        print(f"  Trades: {self.performance['total_trades']} | "
              f"Win Rate: {win_rate:.1f}%")
        print(f"  Wins: {self.performance['wins']} | Losses: {self.performance['losses']}")
        
        if self.performance['wins'] > 0:
            print(f"  Avg Win: ${self.performance['avg_win']:.2f}")
        if self.performance['losses'] > 0:
            print(f"  Avg Loss: ${abs(self.performance['avg_loss']):.2f}")
        
        print(f"  Streak: {self.performance['consecutive_wins']}W / "
              f"{self.performance['consecutive_losses']}L")
        
        print(f"\nLearning Progress:")
        print(f"  Patterns Learned: {len(self.memory.pattern_success)}")
        print(f"  Best Hours: {self.memory.get_best_hours()[:5]}")
        
        # Show top performing setups
        top_patterns = sorted(
            self.memory.pattern_success.items(),
            key=lambda x: x[1]['wins'] / max(1, x[1]['wins'] + x[1]['losses']),
            reverse=True
        )[:3]
        
        if top_patterns:
            print(f"  Top Patterns:")
            for pattern, stats in top_patterns:
                total = stats['wins'] + stats['losses']
                if total > 0:
                    wr = stats['wins'] / total * 100
                    print(f"    {pattern}: {wr:.0f}% ({stats['wins']}W/{stats['losses']}L)")
        
        print(f"\nOpen Positions: {open_positions}/{self.config.MAX_POSITIONS}")
        print(f"{'='*70}")
    
    def run(self):
        """Main trading loop"""
        
        if not self.initialize():
            return
        
        print("Smart Learner starting... Learning from every trade!\n")
        
        last_display = time.time()
        last_adapt = time.time()
        
        try:
            while True:
                # Check if we should trade
                if not self.should_trade_now():
                    time.sleep(30)
                    continue
                
                # Manage positions
                self.manage_positions()
                self.check_closed_trades()
                
                # Check position limit
                positions = mt5.positions_get(magic=self.config.MAGIC)
                if positions and len(positions) >= self.config.MAX_POSITIONS:
                    time.sleep(self.config.SCAN_INTERVAL)
                    continue
                
                # Scan all symbols
                for symbol in self.symbols:
                    # Skip if position limit reached
                    positions = mt5.positions_get(magic=self.config.MAGIC)
                    if positions and len(positions) >= self.config.MAX_POSITIONS:
                        break
                    
                    # Analyze symbol
                    signal = self.analyzer.analyze_symbol(symbol)
                    
                    if signal and signal['num_signals'] >= self.config.MIN_SIGNALS:
                        # Try to execute trade
                        if self.execute_trade(symbol, signal):
                            # Wait before next trade
                            time.sleep(5)
                
                # Display status
                if time.time() - last_display > 30:
                    self.display_status()
                    last_display = time.time()
                
                # Adapt configuration
                if time.time() - last_adapt > 300:  # Every 5 minutes
                    self.config.adapt(self.performance)
                    last_adapt = time.time()
                
                # Sleep
                time.sleep(self.config.SCAN_INTERVAL)
                
        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Saving learning data...")
            self.memory.save_memory()
            self.config.save_config()
            self.final_report()
        finally:
            mt5.shutdown()
    
    def final_report(self):
        """Final report with learning summary"""
        
        account = mt5.account_info()
        if account:
            total_profit = account.balance - self.start_balance
            roi = (total_profit / self.start_balance * 100) if self.start_balance > 0 else 0
            
            print(f"\n{'='*70}")
            print(f"SMART LEARNER - FINAL REPORT")
            print(f"{'='*70}")
            print(f"Starting Balance: ${self.start_balance:.2f}")
            print(f"Final Balance: ${account.balance:.2f}")
            print(f"Total Profit: ${total_profit:.2f}")
            print(f"ROI: {roi:.1f}%")
            
            print(f"\nLearning Summary:")
            print(f"  Patterns Learned: {len(self.memory.pattern_success)}")
            print(f"  Symbols Traded: {len(self.memory.symbol_performance)}")
            print(f"  Total Trades: {self.performance['total_trades']}")
            
            if self.performance['total_trades'] > 0:
                win_rate = self.performance['wins'] / self.performance['total_trades'] * 100
                print(f"  Final Win Rate: {win_rate:.1f}%")
            
            print(f"\nAdapted Settings:")
            print(f"  Min Confidence: {self.config.MIN_CONFIDENCE:.0%}")
            print(f"  Min Signals: {self.config.MIN_SIGNALS}")
            print(f"  Base Lot: {self.config.BASE_LOT}")
            
            print(f"{'='*70}")
            print("\nAll learning has been saved for next session!")

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == '__main__':
    trader = SmartLearnerTrader()
    trader.run()