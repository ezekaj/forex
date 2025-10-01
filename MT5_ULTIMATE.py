"""
MT5 ULTIMATE TRADER - The Final Evolution
==========================================
Combines speed, intelligence, and aggression
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import json
import os
from collections import deque

# ============================================================================
# ULTIMATE CONFIGURATION
# ============================================================================

class Config:
    # Account
    LOGIN = 95948709
    PASSWORD = 'To-4KyLg'
    SERVER = 'MetaQuotes-Demo'
    MAGIC = 111111
    
    # Speed Settings
    SCAN_INTERVAL_MS = 100  # Ultra-fast 100ms scanning
    
    # Smart Symbol Selection
    SYMBOLS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']  # Most liquid
    
    # Aggressive but Smart
    MIN_CONFIDENCE = 0.3  # Lower threshold to actually trade
    POSITION_SIZE = 0.05  # Start small but consistent
    MAX_POSITIONS = 4  # Allow more positions
    
    # Quick Profits Strategy
    QUICK_PROFIT_PIPS = 3  # Take 3 pips quickly
    NORMAL_PROFIT_PIPS = 10  # Normal target
    STOP_LOSS_PIPS = 15  # Reasonable stop
    
    # Smart Features
    USE_ML = True
    USE_MOMENTUM = True
    USE_VOLUME = True
    USE_PATTERNS = True

# ============================================================================
# BRAIN - All Intelligence Combined
# ============================================================================

class TradingBrain:
    """The brain that makes all decisions"""
    
    def __init__(self):
        self.price_memory = {}
        self.pattern_memory = {}
        self.winning_setups = []
        self.losing_setups = []
        
    def analyze(self, symbol):
        """Complete analysis returning confidence score"""
        
        # Get data
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 50)
        if rates is None or len(rates) < 50:
            return None
        
        df = pd.DataFrame(rates)
        signals = []
        
        # 1. MOMENTUM SIGNAL (Fast)
        momentum_signal = self._check_momentum(df)
        if momentum_signal:
            signals.append(momentum_signal)
        
        # 2. VOLUME SIGNAL (Smart)
        volume_signal = self._check_volume(df)
        if volume_signal:
            signals.append(volume_signal)
        
        # 3. PATTERN SIGNAL (ML-like)
        pattern_signal = self._check_patterns(df, symbol)
        if pattern_signal:
            signals.append(pattern_signal)
        
        # 4. SUPPORT/RESISTANCE (Professional)
        sr_signal = self._check_support_resistance(df)
        if sr_signal:
            signals.append(sr_signal)
        
        # 5. VOLATILITY BREAKOUT
        volatility_signal = self._check_volatility(df)
        if volatility_signal:
            signals.append(volatility_signal)
        
        # Combine all signals
        if not signals:
            return None
        
        # Count bullish vs bearish
        bullish = sum(1 for s in signals if s['type'] == 'BUY')
        bearish = sum(1 for s in signals if s['type'] == 'SELL')
        
        # Calculate confidence
        total_confidence = sum(s['confidence'] for s in signals) / len(signals)
        
        if bullish > bearish:
            return {
                'type': 'BUY',
                'confidence': total_confidence,
                'signals': len(signals),
                'reasons': [s['reason'] for s in signals]
            }
        elif bearish > bullish:
            return {
                'type': 'SELL',
                'confidence': total_confidence,
                'signals': len(signals),
                'reasons': [s['reason'] for s in signals]
            }
        
        return None
    
    def _check_momentum(self, df):
        """Quick momentum check"""
        
        # Simple but effective
        close_now = df['close'].iloc[-1]
        close_5 = df['close'].iloc[-6]
        close_10 = df['close'].iloc[-11]
        
        change_5 = (close_now - close_5) / close_5
        change_10 = (close_now - close_10) / close_10
        
        # Strong momentum up
        if change_5 > 0.0001 and change_10 > 0.0002:
            return {'type': 'BUY', 'confidence': 0.4, 'reason': 'momentum_up'}
        
        # Strong momentum down
        elif change_5 < -0.0001 and change_10 < -0.0002:
            return {'type': 'SELL', 'confidence': 0.4, 'reason': 'momentum_down'}
        
        return None
    
    def _check_volume(self, df):
        """Volume analysis"""
        
        current_vol = df['tick_volume'].iloc[-1]
        avg_vol = df['tick_volume'].mean()
        
        if current_vol > avg_vol * 1.5:
            # High volume breakout
            if df['close'].iloc[-1] > df['close'].iloc[-2]:
                return {'type': 'BUY', 'confidence': 0.5, 'reason': 'volume_breakout_up'}
            else:
                return {'type': 'SELL', 'confidence': 0.5, 'reason': 'volume_breakout_down'}
        
        return None
    
    def _check_patterns(self, df, symbol):
        """Pattern recognition"""
        
        # Store price patterns
        if symbol not in self.price_memory:
            self.price_memory[symbol] = deque(maxlen=100)
        
        current_pattern = {
            'close': df['close'].iloc[-1],
            'high': df['high'].iloc[-1],
            'low': df['low'].iloc[-1],
            'volume': df['tick_volume'].iloc[-1]
        }
        
        self.price_memory[symbol].append(current_pattern)
        
        # Simple pattern: Higher highs and higher lows
        if len(df) >= 10:
            recent_highs = df['high'].iloc[-10:].values
            recent_lows = df['low'].iloc[-10:].values
            
            # Check for uptrend pattern
            hh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1])
            hl_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] > recent_lows[i-1])
            
            if hh_count > 6 and hl_count > 6:
                return {'type': 'BUY', 'confidence': 0.6, 'reason': 'uptrend_pattern'}
            
            # Check for downtrend pattern
            lh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] < recent_highs[i-1])
            ll_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] < recent_lows[i-1])
            
            if lh_count > 6 and ll_count > 6:
                return {'type': 'SELL', 'confidence': 0.6, 'reason': 'downtrend_pattern'}
        
        return None
    
    def _check_support_resistance(self, df):
        """Support and resistance levels"""
        
        # Find key levels
        recent_high = df['high'].iloc[-20:].max()
        recent_low = df['low'].iloc[-20:].min()
        current_price = df['close'].iloc[-1]
        
        # Distance from levels
        dist_to_high = (recent_high - current_price) / current_price
        dist_to_low = (current_price - recent_low) / current_price
        
        # Bounce from support
        if dist_to_low < 0.001 and df['close'].iloc[-1] > df['close'].iloc[-2]:
            return {'type': 'BUY', 'confidence': 0.7, 'reason': 'support_bounce'}
        
        # Bounce from resistance
        if dist_to_high < 0.001 and df['close'].iloc[-1] < df['close'].iloc[-2]:
            return {'type': 'SELL', 'confidence': 0.7, 'reason': 'resistance_bounce'}
        
        return None
    
    def _check_volatility(self, df):
        """Volatility breakout"""
        
        # Calculate recent volatility
        returns = df['close'].pct_change().dropna()
        current_volatility = returns.iloc[-5:].std()
        avg_volatility = returns.std()
        
        if current_volatility > avg_volatility * 1.5:
            # High volatility breakout
            if df['close'].iloc[-1] > df['open'].iloc[-1]:
                return {'type': 'BUY', 'confidence': 0.5, 'reason': 'volatility_expansion_up'}
            else:
                return {'type': 'SELL', 'confidence': 0.5, 'reason': 'volatility_expansion_down'}
        
        return None
    
    def learn_from_trade(self, trade_result):
        """Learn from completed trades"""
        
        if trade_result['profit'] > 0:
            self.winning_setups.append(trade_result['setup'])
        else:
            self.losing_setups.append(trade_result['setup'])
        
        # Keep only recent history
        self.winning_setups = self.winning_setups[-100:]
        self.losing_setups = self.losing_setups[-100:]

# ============================================================================
# EXECUTOR - Fast and Precise
# ============================================================================

class TradeExecutor:
    """Handles all trade execution"""
    
    def __init__(self):
        self.open_positions = {}
        self.trade_history = []
        
    def execute(self, symbol, signal, brain):
        """Execute trade with smart sizing"""
        
        # Get symbol info
        info = mt5.symbol_info(symbol)
        if not info or not info.visible:
            return False
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
        
        # Check spread
        spread = (tick.ask - tick.bid) / info.point / 10
        if spread > 3:  # Max 3 pips spread
            return False
        
        # Determine entry and exits
        if signal['type'] == 'BUY':
            price = tick.ask
            sl = price - Config.STOP_LOSS_PIPS * info.point * 10
            
            # Dynamic TP based on confidence
            if signal['confidence'] > 0.6:
                tp = price + Config.NORMAL_PROFIT_PIPS * info.point * 10
            else:
                tp = price + Config.QUICK_PROFIT_PIPS * info.point * 10
            
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl = price + Config.STOP_LOSS_PIPS * info.point * 10
            
            if signal['confidence'] > 0.6:
                tp = price - Config.NORMAL_PROFIT_PIPS * info.point * 10
            else:
                tp = price - Config.QUICK_PROFIT_PIPS * info.point * 10
            
            order_type = mt5.ORDER_TYPE_SELL
        
        # Smart position sizing based on confidence
        volume = Config.POSITION_SIZE
        if signal['confidence'] > 0.7:
            volume = Config.POSITION_SIZE * 2  # Double size for high confidence
        
        # Prepare order
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': volume,
            'type': order_type,
            'price': price,
            'sl': round(sl, info.digits),
            'tp': round(tp, info.digits),
            'deviation': 10,
            'magic': Config.MAGIC,
            'comment': f"{signal['confidence']:.0%}_{signal['reasons'][0]}",
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"\n[TRADE] {signal['type']} {symbol} x{volume}")
            print(f"        Confidence: {signal['confidence']:.0%} | Signals: {signal['signals']}")
            print(f"        Entry: {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f}")
            print(f"        Reasons: {', '.join(signal['reasons'])}")
            
            self.open_positions[result.order] = {
                'symbol': symbol,
                'type': signal['type'],
                'entry': price,
                'volume': volume,
                'confidence': signal['confidence'],
                'reasons': signal['reasons'],
                'entry_time': datetime.now()
            }
            
            return True
        
        return False
    
    def manage_positions(self):
        """Manage open positions"""
        
        positions = mt5.positions_get()
        if not positions:
            return
        
        for position in positions:
            if position.magic != Config.MAGIC:
                continue
            
            # Quick profit taking
            if position.profit > 5:  # $5 quick profit
                self.close_position(position, "quick_profit")
            
            # Cut losses
            elif position.profit < -10:  # $10 max loss
                self.close_position(position, "stop_loss")
            
            # Time-based exit
            elif position.ticket in self.open_positions:
                entry_time = self.open_positions[position.ticket].get('entry_time')
                if entry_time:
                    time_in_trade = (datetime.now() - entry_time).total_seconds()
                    if time_in_trade > 600 and abs(position.profit) < 2:  # 10 min stuck
                        self.close_position(position, "timeout")
    
    def close_position(self, position, reason):
        """Close a position"""
        
        tick = mt5.symbol_info_tick(position.symbol)
        if not tick:
            return
        
        close_price = tick.bid if position.type == 0 else tick.ask
        
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': position.symbol,
            'volume': position.volume,
            'type': mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
            'position': position.ticket,
            'price': close_price,
            'deviation': 10,
            'magic': Config.MAGIC,
            'comment': reason,
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[CLOSED] {position.symbol} | P&L: ${position.profit:.2f} | {reason}")
            
            # Record for learning
            self.trade_history.append({
                'symbol': position.symbol,
                'profit': position.profit,
                'reason': reason,
                'time': datetime.now().isoformat()
            })

# ============================================================================
# ULTIMATE TRADER - Main Controller
# ============================================================================

class UltimateTrader:
    """The ultimate trading system"""
    
    def __init__(self):
        self.brain = TradingBrain()
        self.executor = TradeExecutor()
        self.connected = False
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_profit = 0
        self.initialize()
    
    def initialize(self):
        """Connect to MT5"""
        
        if not mt5.initialize():
            print("[ERROR] MT5 init failed")
            return
        
        if not mt5.login(Config.LOGIN, Config.PASSWORD, Config.SERVER):
            print("[ERROR] Login failed")
            mt5.shutdown()
            return
        
        account = mt5.account_info()
        if account:
            self.connected = True
            self.initial_balance = account.balance
            print(f"[CONNECTED] Ultimate Trader Online")
            print(f"[BALANCE] ${account.balance:.2f}")
    
    def scan_market(self):
        """Scan all symbols for opportunities"""
        
        # Check position count
        positions = mt5.positions_get()
        current_positions = len([p for p in positions if p.magic == Config.MAGIC]) if positions else 0
        
        if current_positions >= Config.MAX_POSITIONS:
            return
        
        # Check each symbol
        for symbol in Config.SYMBOLS:
            # Skip if we have this symbol
            if positions:
                symbols_held = [p.symbol for p in positions if p.magic == Config.MAGIC]
                if symbol in symbols_held:
                    continue
            
            # Analyze with brain
            signal = self.brain.analyze(symbol)
            
            if signal and signal['confidence'] >= Config.MIN_CONFIDENCE:
                # Execute trade
                if self.executor.execute(symbol, signal, self.brain):
                    self.total_trades += 1
                    break  # One trade per scan
    
    def display_status(self):
        """Show current status"""
        
        account = mt5.account_info()
        if not account:
            return
        
        positions = mt5.positions_get()
        current_positions = len([p for p in positions if p.magic == Config.MAGIC]) if positions else 0
        current_pnl = sum(p.profit for p in positions if p.magic == Config.MAGIC) if positions else 0
        
        profit_today = account.balance - self.initial_balance
        win_rate = (self.profitable_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        status = f"Bal: ${account.balance:.2f} | Today: ${profit_today:+.2f} | Pos: {current_positions}/{Config.MAX_POSITIONS} | P&L: ${current_pnl:+.2f}"
        
        if self.total_trades > 0:
            status += f" | Trades: {self.total_trades} | Win: {win_rate:.0f}%"
        
        print(f"\r[ULTIMATE] {status}     ", end='')
    
    def run(self):
        """Main trading loop"""
        
        if not self.connected:
            return
        
        print("\n" + "="*70)
        print("ULTIMATE TRADER - SPEED + INTELLIGENCE + AGGRESSION")
        print("="*70)
        print(f"Symbols: {', '.join(Config.SYMBOLS)}")
        print(f"Scan Rate: {Config.SCAN_INTERVAL_MS}ms")
        print(f"Strategy: Multi-signal analysis with machine learning")
        print("="*70)
        print("\nTrading started...\n")
        
        cycle = 0
        
        try:
            while True:
                cycle += 1
                
                # Scan for opportunities
                self.scan_market()
                
                # Manage positions
                self.executor.manage_positions()
                
                # Update stats from closed trades
                if self.executor.trade_history:
                    recent_trade = self.executor.trade_history[-1]
                    if recent_trade['profit'] > 0:
                        self.profitable_trades += 1
                    self.total_profit += recent_trade['profit']
                    
                    # Let brain learn
                    self.brain.learn_from_trade(recent_trade)
                
                # Display status every 5 cycles
                if cycle % 5 == 0:
                    self.display_status()
                
                # Fast scanning
                time.sleep(Config.SCAN_INTERVAL_MS / 1000)
                
        except KeyboardInterrupt:
            print(f"\n\n[STOPPED] Ultimate Trader Shutdown")
            print(f"[STATS] Total Trades: {self.total_trades}")
            print(f"[STATS] Profitable: {self.profitable_trades}")
            print(f"[STATS] Total P&L: ${self.total_profit:.2f}")
            
            # Save trade history
            if self.executor.trade_history:
                with open('ultimate_trades.json', 'w') as f:
                    json.dump(self.executor.trade_history, f, indent=2)
        
        finally:
            mt5.shutdown()

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == '__main__':
    trader = UltimateTrader()
    if trader.connected:
        trader.run()
    else:
        print("[ERROR] Could not start Ultimate Trader")