"""
MT5 PROFITABLE TRADER - Realistic & Profitable
===============================================
Focus on quality trades with high win rate
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PROFITABLE CONFIGURATION
# ============================================================================

class Config:
    # Account
    LOGIN = 95948709
    PASSWORD = 'To-4KyLg'
    SERVER = 'MetaQuotes-Demo'
    MAGIC = 888888  # Lucky 8s
    
    # Conservative Settings for Profitability
    SCAN_INTERVAL = 1  # 1 second - no rush
    MAX_POSITIONS = 5  # Few quality positions
    BASE_LOT = 0.02  # Your profitable size
    
    # Strict Entry Requirements
    MIN_CONFIDENCE = 0.75  # Only high confidence trades
    MIN_SIGNALS = 3  # Need 3+ confirming signals
    
    # Risk Management
    STOP_LOSS_PIPS = 10
    TAKE_PROFIT_PIPS = 15  # 1.5:1 reward/risk
    TRAILING_START_PIPS = 8
    TRAILING_DISTANCE_PIPS = 5
    
    # Only Best Pairs
    SYMBOLS = [
        'EURUSD',  # Most liquid
        'GBPUSD',  # Good volatility
        'USDJPY',  # Stable trends
        'AUDUSD',  # Commodity correlation
        'USDCHF'   # Safe haven
    ]
    
    # Session Times (Best trading hours)
    LONDON_OPEN = 8
    LONDON_CLOSE = 16
    NY_OPEN = 13
    NY_CLOSE = 22
    
    # Avoid trading during:
    AVOID_HOURS = [23, 0, 1, 2, 3, 4, 5]  # Low liquidity hours

# ============================================================================
# PROFITABLE STRATEGY ENGINE
# ============================================================================

class ProfitableStrategy:
    """Only take high-probability trades"""
    
    def __init__(self):
        self.performance = {}
        self.winning_setups = self.load_winning_setups()
        self.losing_setups = self.load_losing_setups()
        
    def load_winning_setups(self):
        """Load historically winning setups"""
        if os.path.exists('winning_setups.json'):
            with open('winning_setups.json', 'r') as f:
                return json.load(f)
        
        # Default winning setups
        return {
            'trend_continuation': {'wins': 0, 'losses': 0, 'avg_profit': 0},
            'support_bounce': {'wins': 0, 'losses': 0, 'avg_profit': 0},
            'resistance_rejection': {'wins': 0, 'losses': 0, 'avg_profit': 0},
            'momentum_breakout': {'wins': 0, 'losses': 0, 'avg_profit': 0},
            'mean_reversion': {'wins': 0, 'losses': 0, 'avg_profit': 0}
        }
    
    def load_losing_setups(self):
        """Load setups to avoid"""
        if os.path.exists('losing_setups.json'):
            with open('losing_setups.json', 'r') as f:
                return json.load(f)
        return {}
    
    def save_setups(self):
        """Save setup performance"""
        with open('winning_setups.json', 'w') as f:
            json.dump(self.winning_setups, f)
        with open('losing_setups.json', 'w') as f:
            json.dump(self.losing_setups, f)
    
    def analyze_market(self, symbol):
        """Comprehensive market analysis"""
        
        # Get multiple timeframes
        m1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 60)
        m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 60)
        m15 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 40)
        h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 24)
        
        if m5 is None or len(m5) < 60:
            return None
        
        signals = []
        
        # 1. Trend Analysis
        trend_signal = self.analyze_trend(m5, m15, h1)
        if trend_signal:
            signals.append(trend_signal)
        
        # 2. Support/Resistance
        sr_signal = self.analyze_support_resistance(m5)
        if sr_signal:
            signals.append(sr_signal)
        
        # 3. Momentum
        momentum_signal = self.analyze_momentum(m1, m5)
        if momentum_signal:
            signals.append(momentum_signal)
        
        # 4. Volume Analysis
        volume_signal = self.analyze_volume(m5)
        if volume_signal:
            signals.append(volume_signal)
        
        # 5. Pattern Recognition
        pattern_signal = self.find_patterns(m5)
        if pattern_signal:
            signals.append(pattern_signal)
        
        # 6. Market Structure
        structure_signal = self.analyze_structure(m15)
        if structure_signal:
            signals.append(structure_signal)
        
        return signals
    
    def analyze_trend(self, m5, m15, h1):
        """Multi-timeframe trend analysis"""
        
        df5 = pd.DataFrame(m5)
        df15 = pd.DataFrame(m15) if m15 is not None and len(m15) > 20 else None
        dfh1 = pd.DataFrame(h1) if h1 is not None and len(h1) > 20 else None
        
        # Calculate EMAs
        df5['ema20'] = df5['close'].ewm(span=20).mean()
        df5['ema50'] = df5['close'].ewm(span=50).mean()
        
        # Current trend
        current_price = df5['close'].iloc[-1]
        ema20 = df5['ema20'].iloc[-1]
        ema50 = df5['ema50'].iloc[-1]
        
        # Trend strength
        if current_price > ema20 > ema50:
            # Check higher timeframe confirmation
            if df15 is not None:
                df15['ema20'] = df15['close'].ewm(span=20).mean()
                if df15['close'].iloc[-1] > df15['ema20'].iloc[-1]:
                    return {
                        'type': 'BUY',
                        'setup': 'trend_continuation',
                        'confidence': 0.8,
                        'reason': 'Strong uptrend on multiple timeframes'
                    }
        
        elif current_price < ema20 < ema50:
            # Check higher timeframe confirmation
            if df15 is not None:
                df15['ema20'] = df15['close'].ewm(span=20).mean()
                if df15['close'].iloc[-1] < df15['ema20'].iloc[-1]:
                    return {
                        'type': 'SELL',
                        'setup': 'trend_continuation',
                        'confidence': 0.8,
                        'reason': 'Strong downtrend on multiple timeframes'
                    }
        
        return None
    
    def analyze_support_resistance(self, rates):
        """Find support and resistance levels"""
        
        df = pd.DataFrame(rates)
        
        # Find recent highs and lows
        window = 20
        df['resistance'] = df['high'].rolling(window).max()
        df['support'] = df['low'].rolling(window).min()
        
        current_price = df['close'].iloc[-1]
        resistance = df['resistance'].iloc[-1]
        support = df['support'].iloc[-1]
        
        # Calculate distances
        dist_to_resistance = (resistance - current_price) / current_price
        dist_to_support = (current_price - support) / current_price
        
        # Bounce from support
        if dist_to_support < 0.001 and df['close'].iloc[-1] > df['close'].iloc[-2]:
            # Check if support held multiple times
            support_tests = sum(1 for i in range(-20, 0) if abs(df['low'].iloc[i] - support) / support < 0.001)
            if support_tests >= 2:
                return {
                    'type': 'BUY',
                    'setup': 'support_bounce',
                    'confidence': 0.75 + (support_tests * 0.05),
                    'reason': f'Bounce from support tested {support_tests} times'
                }
        
        # Rejection from resistance
        elif dist_to_resistance < 0.001 and df['close'].iloc[-1] < df['close'].iloc[-2]:
            # Check if resistance held multiple times
            resistance_tests = sum(1 for i in range(-20, 0) if abs(df['high'].iloc[i] - resistance) / resistance < 0.001)
            if resistance_tests >= 2:
                return {
                    'type': 'SELL',
                    'setup': 'resistance_rejection',
                    'confidence': 0.75 + (resistance_tests * 0.05),
                    'reason': f'Rejection from resistance tested {resistance_tests} times'
                }
        
        return None
    
    def analyze_momentum(self, m1, m5):
        """Momentum analysis"""
        
        if m1 is None or len(m1) < 20:
            return None
        
        df1 = pd.DataFrame(m1)
        df5 = pd.DataFrame(m5)
        
        # RSI
        delta = df5['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        
        # MACD
        ema12 = df5['close'].ewm(span=12).mean()
        ema26 = df5['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        
        # Strong momentum signals
        if current_rsi < 30 and prev_rsi < current_rsi:
            # Oversold and turning up
            return {
                'type': 'BUY',
                'setup': 'momentum_breakout',
                'confidence': 0.75,
                'reason': f'RSI oversold reversal at {current_rsi:.0f}'
            }
        
        elif current_rsi > 70 and prev_rsi > current_rsi:
            # Overbought and turning down
            return {
                'type': 'SELL',
                'setup': 'momentum_breakout',
                'confidence': 0.75,
                'reason': f'RSI overbought reversal at {current_rsi:.0f}'
            }
        
        # MACD crossover
        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            return {
                'type': 'BUY',
                'setup': 'momentum_breakout',
                'confidence': 0.7,
                'reason': 'MACD bullish crossover'
            }
        elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            return {
                'type': 'SELL',
                'setup': 'momentum_breakout',
                'confidence': 0.7,
                'reason': 'MACD bearish crossover'
            }
        
        return None
    
    def analyze_volume(self, rates):
        """Volume analysis"""
        
        df = pd.DataFrame(rates)
        
        # Volume moving average
        df['vol_ma'] = df['tick_volume'].rolling(20).mean()
        df['vol_std'] = df['tick_volume'].rolling(20).std()
        
        current_volume = df['tick_volume'].iloc[-1]
        avg_volume = df['vol_ma'].iloc[-1]
        
        # Volume spike
        if current_volume > avg_volume * 2:
            # Check price action
            if df['close'].iloc[-1] > df['open'].iloc[-1]:
                # Bullish volume
                return {
                    'type': 'BUY',
                    'setup': 'momentum_breakout',
                    'confidence': 0.7,
                    'reason': f'Volume spike {current_volume/avg_volume:.1f}x average'
                }
            else:
                # Bearish volume
                return {
                    'type': 'SELL',
                    'setup': 'momentum_breakout',
                    'confidence': 0.7,
                    'reason': f'Volume spike {current_volume/avg_volume:.1f}x average'
                }
        
        return None
    
    def find_patterns(self, rates):
        """Pattern recognition"""
        
        df = pd.DataFrame(rates)
        
        # Pin bar / Hammer
        for i in range(-3, 0):
            body = abs(df['close'].iloc[i] - df['open'].iloc[i])
            upper_wick = df['high'].iloc[i] - max(df['close'].iloc[i], df['open'].iloc[i])
            lower_wick = min(df['close'].iloc[i], df['open'].iloc[i]) - df['low'].iloc[i]
            
            # Bullish pin bar
            if lower_wick > body * 2 and upper_wick < body:
                if df['close'].iloc[-1] > df['high'].iloc[i]:
                    return {
                        'type': 'BUY',
                        'setup': 'support_bounce',
                        'confidence': 0.75,
                        'reason': 'Bullish pin bar confirmation'
                    }
            
            # Bearish pin bar
            elif upper_wick > body * 2 and lower_wick < body:
                if df['close'].iloc[-1] < df['low'].iloc[i]:
                    return {
                        'type': 'SELL',
                        'setup': 'resistance_rejection',
                        'confidence': 0.75,
                        'reason': 'Bearish pin bar confirmation'
                    }
        
        # Engulfing pattern
        if len(df) >= 2:
            # Bullish engulfing
            if (df['close'].iloc[-1] > df['open'].iloc[-1] and
                df['close'].iloc[-2] < df['open'].iloc[-2] and
                df['close'].iloc[-1] > df['open'].iloc[-2] and
                df['open'].iloc[-1] < df['close'].iloc[-2]):
                return {
                    'type': 'BUY',
                    'setup': 'momentum_breakout',
                    'confidence': 0.75,
                    'reason': 'Bullish engulfing pattern'
                }
            
            # Bearish engulfing
            elif (df['close'].iloc[-1] < df['open'].iloc[-1] and
                  df['close'].iloc[-2] > df['open'].iloc[-2] and
                  df['close'].iloc[-1] < df['open'].iloc[-2] and
                  df['open'].iloc[-1] > df['close'].iloc[-2]):
                return {
                    'type': 'SELL',
                    'setup': 'momentum_breakout',
                    'confidence': 0.75,
                    'reason': 'Bearish engulfing pattern'
                }
        
        return None
    
    def analyze_structure(self, rates):
        """Market structure analysis"""
        
        if rates is None or len(rates) < 20:
            return None
        
        df = pd.DataFrame(rates)
        
        # Find swing highs and lows
        highs = []
        lows = []
        
        for i in range(2, len(df) - 2):
            # Swing high
            if df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i-2] and \
               df['high'].iloc[i] > df['high'].iloc[i+1] and df['high'].iloc[i] > df['high'].iloc[i+2]:
                highs.append((i, df['high'].iloc[i]))
            
            # Swing low
            if df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i-2] and \
               df['low'].iloc[i] < df['low'].iloc[i+1] and df['low'].iloc[i] < df['low'].iloc[i+2]:
                lows.append((i, df['low'].iloc[i]))
        
        # Check for higher highs and higher lows (uptrend)
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1][1] > highs[-2][1] and lows[-1][1] > lows[-2][1]:
                return {
                    'type': 'BUY',
                    'setup': 'trend_continuation',
                    'confidence': 0.8,
                    'reason': 'Higher highs and higher lows'
                }
            
            # Lower highs and lower lows (downtrend)
            elif highs[-1][1] < highs[-2][1] and lows[-1][1] < lows[-2][1]:
                return {
                    'type': 'SELL',
                    'setup': 'trend_continuation',
                    'confidence': 0.8,
                    'reason': 'Lower highs and lower lows'
                }
        
        return None
    
    def should_trade(self, signals):
        """Determine if we should take the trade"""
        
        if not signals or len(signals) < Config.MIN_SIGNALS:
            return None
        
        # Count signal types
        buy_signals = [s for s in signals if s['type'] == 'BUY']
        sell_signals = [s for s in signals if s['type'] == 'SELL']
        
        # Need clear direction
        if len(buy_signals) >= Config.MIN_SIGNALS:
            # Calculate average confidence
            avg_confidence = np.mean([s['confidence'] for s in buy_signals])
            
            if avg_confidence >= Config.MIN_CONFIDENCE:
                # Check if setup has good history
                setup = buy_signals[0]['setup']
                if setup in self.winning_setups:
                    win_rate = self.winning_setups[setup]['wins'] / max(1, self.winning_setups[setup]['wins'] + self.winning_setups[setup]['losses'])
                    if win_rate < 0.3 and self.winning_setups[setup]['wins'] + self.winning_setups[setup]['losses'] > 10:
                        return None  # Avoid losing setup
                
                return {
                    'type': 'BUY',
                    'confidence': avg_confidence,
                    'setup': setup,
                    'reasons': [s['reason'] for s in buy_signals]
                }
        
        elif len(sell_signals) >= Config.MIN_SIGNALS:
            # Calculate average confidence
            avg_confidence = np.mean([s['confidence'] for s in sell_signals])
            
            if avg_confidence >= Config.MIN_CONFIDENCE:
                # Check if setup has good history
                setup = sell_signals[0]['setup']
                if setup in self.winning_setups:
                    win_rate = self.winning_setups[setup]['wins'] / max(1, self.winning_setups[setup]['wins'] + self.winning_setups[setup]['losses'])
                    if win_rate < 0.3 and self.winning_setups[setup]['wins'] + self.winning_setups[setup]['losses'] > 10:
                        return None  # Avoid losing setup
                
                return {
                    'type': 'SELL',
                    'confidence': avg_confidence,
                    'setup': setup,
                    'reasons': [s['reason'] for s in sell_signals]
                }
        
        return None
    
    def update_performance(self, setup, profit):
        """Update setup performance"""
        
        if setup not in self.winning_setups:
            self.winning_setups[setup] = {'wins': 0, 'losses': 0, 'avg_profit': 0}
        
        if profit > 0:
            self.winning_setups[setup]['wins'] += 1
        else:
            self.winning_setups[setup]['losses'] += 1
            
            # Add to losing setups if consistently bad
            if self.winning_setups[setup]['losses'] > 10 and \
               self.winning_setups[setup]['wins'] / max(1, self.winning_setups[setup]['losses']) < 0.3:
                self.losing_setups[setup] = True
        
        # Update average profit
        total_trades = self.winning_setups[setup]['wins'] + self.winning_setups[setup]['losses']
        self.winning_setups[setup]['avg_profit'] = (
            (self.winning_setups[setup]['avg_profit'] * (total_trades - 1) + profit) / total_trades
        )
        
        self.save_setups()

# ============================================================================
# MAIN PROFITABLE TRADER
# ============================================================================

class ProfitableTrader:
    """Main trading system focused on profitability"""
    
    def __init__(self):
        self.strategy = ProfitableStrategy()
        self.positions = {}
        self.daily_stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'profit': 0,
            'max_drawdown': 0
        }
        self.start_balance = 0
        self.peak_balance = 0
        
    def initialize(self):
        """Initialize MT5"""
        
        if not mt5.initialize():
            print("[ERROR] Failed to initialize MT5")
            return False
        
        if not mt5.login(Config.LOGIN, Config.PASSWORD, Config.SERVER):
            print("[ERROR] Failed to login")
            mt5.shutdown()
            return False
        
        account = mt5.account_info()
        if account:
            self.start_balance = account.balance
            self.peak_balance = account.balance
            
            print(f"\n{'='*70}")
            print(f"MT5 PROFITABLE TRADER")
            print(f"{'='*70}")
            print(f"Balance: ${account.balance:.2f}")
            print(f"Strategy: Quality over Quantity")
            print(f"Min Confidence: {Config.MIN_CONFIDENCE*100:.0f}%")
            print(f"Min Signals: {Config.MIN_SIGNALS}")
            print(f"Risk/Reward: 1:{Config.TAKE_PROFIT_PIPS/Config.STOP_LOSS_PIPS:.1f}")
            print(f"{'='*70}\n")
            
            return True
        
        return False
    
    def should_trade_now(self):
        """Check if current time is good for trading"""
        
        current_hour = datetime.now().hour
        
        # Avoid low liquidity hours
        if current_hour in Config.AVOID_HOURS:
            return False
        
        # Best during London/NY overlap
        if Config.NY_OPEN <= current_hour <= Config.LONDON_CLOSE:
            return True
        
        # OK during main sessions
        if Config.LONDON_OPEN <= current_hour <= Config.NY_CLOSE:
            return True
        
        return False
    
    def execute_trade(self, symbol, signal):
        """Execute a quality trade"""
        
        # Get symbol info
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        
        if not info or not tick:
            return False
        
        # Check spread
        spread = (tick.ask - tick.bid) / info.point
        if spread > 20:  # Max 2 pips spread
            print(f"[SKIP] {symbol} spread too high: {spread/10:.1f} pips")
            return False
        
        # Setup trade parameters
        if signal['type'] == 'BUY':
            price = tick.ask
            sl = price - Config.STOP_LOSS_PIPS * info.point * 10
            tp = price + Config.TAKE_PROFIT_PIPS * info.point * 10
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl = price + Config.STOP_LOSS_PIPS * info.point * 10
            tp = price - Config.TAKE_PROFIT_PIPS * info.point * 10
            order_type = mt5.ORDER_TYPE_SELL
        
        # Prepare request
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': Config.BASE_LOT,
            'type': order_type,
            'price': price,
            'sl': round(sl, info.digits),
            'tp': round(tp, info.digits),
            'deviation': 10,
            'magic': Config.MAGIC,
            'comment': f"{signal['setup']}_{signal['confidence']:.2f}",
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.daily_stats['trades'] += 1
            
            # Store position info
            self.positions[result.order] = {
                'symbol': symbol,
                'type': signal['type'],
                'setup': signal['setup'],
                'entry': price,
                'sl': sl,
                'tp': tp,
                'time': datetime.now()
            }
            
            print(f"\n[TRADE] {signal['type']} {symbol}")
            print(f"  Setup: {signal['setup']}")
            print(f"  Confidence: {signal['confidence']:.0%}")
            print(f"  Entry: {price:.5f}")
            print(f"  SL: {sl:.5f} ({Config.STOP_LOSS_PIPS} pips)")
            print(f"  TP: {tp:.5f} ({Config.TAKE_PROFIT_PIPS} pips)")
            print(f"  Reasons: {', '.join(signal['reasons'][:2])}")
            
            return True
        
        return False
    
    def manage_positions(self):
        """Manage open positions with trailing stop"""
        
        positions = mt5.positions_get(magic=Config.MAGIC)
        if not positions:
            return
        
        for pos in positions:
            # Get current price
            tick = mt5.symbol_info_tick(pos.symbol)
            info = mt5.symbol_info(pos.symbol)
            
            if not tick or not info:
                continue
            
            # Calculate profit in pips
            if pos.type == 0:  # BUY
                pips = (tick.bid - pos.price_open) / (info.point * 10)
            else:  # SELL
                pips = (pos.price_open - tick.ask) / (info.point * 10)
            
            # Trailing stop
            if pips >= Config.TRAILING_START_PIPS:
                if pos.type == 0:  # BUY
                    new_sl = tick.bid - Config.TRAILING_DISTANCE_PIPS * info.point * 10
                    if new_sl > pos.sl:
                        request = {
                            'action': mt5.TRADE_ACTION_SLTP,
                            'symbol': pos.symbol,
                            'position': pos.ticket,
                            'sl': round(new_sl, info.digits),
                            'tp': pos.tp,
                            'magic': Config.MAGIC
                        }
                        result = mt5.order_send(request)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            print(f"[TRAIL] {pos.symbol} SL moved to {new_sl:.5f} (+{pips:.1f} pips)")
                
                else:  # SELL
                    new_sl = tick.ask + Config.TRAILING_DISTANCE_PIPS * info.point * 10
                    if new_sl < pos.sl:
                        request = {
                            'action': mt5.TRADE_ACTION_SLTP,
                            'symbol': pos.symbol,
                            'position': pos.ticket,
                            'sl': round(new_sl, info.digits),
                            'tp': pos.tp,
                            'magic': Config.MAGIC
                        }
                        result = mt5.order_send(request)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            print(f"[TRAIL] {pos.symbol} SL moved to {new_sl:.5f} (+{pips:.1f} pips)")
            
            # Update stats when position closes
            if pos.ticket not in self.positions:
                # Position was opened before bot started
                self.positions[pos.ticket] = {
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == 0 else 'SELL',
                    'setup': 'unknown',
                    'entry': pos.price_open
                }
    
    def check_closed_positions(self):
        """Check recently closed positions and update stats"""
        
        # Get today's deals
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        deals = mt5.history_deals_get(today, datetime.now())
        
        if deals:
            for deal in deals:
                if deal.magic == Config.MAGIC and deal.entry == 1:  # Exit deal
                    # Find the position in our records
                    for pos_id, pos_info in list(self.positions.items()):
                        if deal.position_id in self.positions:
                            # Update performance
                            if deal.profit > 0:
                                self.daily_stats['wins'] += 1
                                print(f"[WIN] {pos_info['symbol']} +${deal.profit:.2f}")
                            else:
                                self.daily_stats['losses'] += 1
                                print(f"[LOSS] {pos_info['symbol']} -${abs(deal.profit):.2f}")
                            
                            self.daily_stats['profit'] += deal.profit
                            
                            # Update strategy performance
                            self.strategy.update_performance(pos_info['setup'], deal.profit)
                            
                            # Remove from tracking
                            del self.positions[deal.position_id]
                            break
    
    def display_stats(self):
        """Display current statistics"""
        
        account = mt5.account_info()
        if not account:
            return
        
        positions = mt5.positions_get(magic=Config.MAGIC)
        open_positions = len(positions) if positions else 0
        
        # Update peak
        if account.balance > self.peak_balance:
            self.peak_balance = account.balance
        
        # Calculate stats
        total_profit = account.balance - self.start_balance
        roi = (total_profit / self.start_balance * 100) if self.start_balance > 0 else 0
        
        win_rate = 0
        if self.daily_stats['wins'] + self.daily_stats['losses'] > 0:
            win_rate = self.daily_stats['wins'] / (self.daily_stats['wins'] + self.daily_stats['losses']) * 100
        
        print(f"\n{'='*70}")
        print(f"PROFITABLE TRADER STATUS")
        print(f"{'='*70}")
        print(f"Balance: ${account.balance:.2f} | Equity: ${account.equity:.2f}")
        print(f"Session P&L: ${total_profit:+.2f} ({roi:+.1f}%)")
        print(f"Peak: ${self.peak_balance:.2f}")
        print(f"\nToday: {self.daily_stats['trades']} trades | "
              f"{self.daily_stats['wins']}W/{self.daily_stats['losses']}L | "
              f"Win Rate: {win_rate:.1f}%")
        print(f"Daily P&L: ${self.daily_stats['profit']:+.2f}")
        print(f"Open Positions: {open_positions}/{Config.MAX_POSITIONS}")
        
        # Show winning setups
        if self.strategy.winning_setups:
            print(f"\nTop Setups:")
            for setup, stats in sorted(self.strategy.winning_setups.items(), 
                                      key=lambda x: x[1]['avg_profit'], reverse=True)[:3]:
                if stats['wins'] + stats['losses'] > 0:
                    wr = stats['wins'] / (stats['wins'] + stats['losses']) * 100
                    print(f"  {setup}: {wr:.0f}% win rate, ${stats['avg_profit']:.2f} avg")
        
        print(f"{'='*70}")
    
    def run(self):
        """Main trading loop"""
        
        if not self.initialize():
            return
        
        print("Starting Profitable Trader...")
        print("Waiting for high-probability setups...\n")
        
        last_stats_time = time.time()
        
        try:
            while True:
                # Check if we should trade
                if not self.should_trade_now():
                    time.sleep(60)  # Wait a minute during off hours
                    continue
                
                # Check account
                account = mt5.account_info()
                if not account:
                    continue
                
                # Manage existing positions
                self.manage_positions()
                self.check_closed_positions()
                
                # Check position limit
                positions = mt5.positions_get(magic=Config.MAGIC)
                if positions and len(positions) >= Config.MAX_POSITIONS:
                    time.sleep(Config.SCAN_INTERVAL)
                    continue
                
                # Scan for opportunities
                for symbol in Config.SYMBOLS:
                    # Analyze market
                    signals = self.strategy.analyze_market(symbol)
                    
                    if signals:
                        # Check if we should trade
                        trade_signal = self.strategy.should_trade(signals)
                        
                        if trade_signal:
                            # Execute trade
                            if self.execute_trade(symbol, trade_signal):
                                # Wait before next trade
                                time.sleep(5)
                                break
                
                # Display stats every 30 seconds
                if time.time() - last_stats_time > 30:
                    self.display_stats()
                    last_stats_time = time.time()
                
                # Wait before next scan
                time.sleep(Config.SCAN_INTERVAL)
                
        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Closing trader...")
            self.final_report()
        finally:
            mt5.shutdown()
    
    def final_report(self):
        """Final performance report"""
        
        account = mt5.account_info()
        if account:
            total_profit = account.balance - self.start_balance
            roi = (total_profit / self.start_balance * 100) if self.start_balance > 0 else 0
            
            print(f"\n{'='*70}")
            print(f"FINAL REPORT")
            print(f"{'='*70}")
            print(f"Starting Balance: ${self.start_balance:.2f}")
            print(f"Final Balance: ${account.balance:.2f}")
            print(f"Total Profit: ${total_profit:.2f}")
            print(f"ROI: {roi:.1f}%")
            print(f"Peak Balance: ${self.peak_balance:.2f}")
            
            total_trades = self.daily_stats['wins'] + self.daily_stats['losses']
            if total_trades > 0:
                win_rate = self.daily_stats['wins'] / total_trades * 100
                print(f"\nTotal Trades: {total_trades}")
                print(f"Wins: {self.daily_stats['wins']}")
                print(f"Losses: {self.daily_stats['losses']}")
                print(f"Win Rate: {win_rate:.1f}%")
            
            print(f"{'='*70}")

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == '__main__':
    trader = ProfitableTrader()
    trader.run()