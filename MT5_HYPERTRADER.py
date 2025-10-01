"""
MT5 HYPERTRADER - Ultra-Aggressive 24/7 Trading System
========================================================
Trades continuously with advanced features
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
import threading
from collections import deque, defaultdict
import random

# ============================================================================
# HYPER CONFIGURATION
# ============================================================================

class Config:
    # Account
    LOGIN = 95948709
    PASSWORD = 'To-4KyLg'
    SERVER = 'MetaQuotes-Demo'
    MAGIC = 444444
    
    # ULTRA AGGRESSIVE SETTINGS
    SCAN_INTERVAL_MS = 10  # 10ms ultra-fast
    MAX_POSITIONS = 10  # Allow more positions
    POSITION_SIZE = 0.01  # Small but many
    
    # Trade Every Market Condition
    TRADE_RANGING = True
    TRADE_TRENDING = True
    TRADE_VOLATILE = True
    TRADE_QUIET = True
    
    # Profit Targets
    SCALP_PROFIT = 2  # 2 pips quick profit
    NORMAL_PROFIT = 5  # 5 pips normal
    RUNNER_PROFIT = 20  # 20 pips for runners
    
    # All Symbols
    SYMBOLS = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
        'NZDUSD', 'USDCHF', 'EURJPY', 'GBPJPY', 'AUDJPY',
        'EURAUD', 'EURGBP', 'AUDNZD', 'NZDJPY', 'CADJPY'
    ]
    
    # Strategy Mix
    USE_SCALPING = True
    USE_MOMENTUM = True
    USE_REVERSAL = True
    USE_BREAKOUT = True
    USE_GRID = True
    USE_MARTINGALE = False  # Risky but profitable

# ============================================================================
# PROFIT MAXIMIZER
# ============================================================================

class ProfitMaximizer:
    """Maximize profits with aggressive strategies"""
    
    def __init__(self):
        self.profit_targets = {}
        self.loss_recovery = {}
        self.winning_streaks = defaultdict(int)
        self.pyramid_positions = {}
    
    def calculate_dynamic_size(self, symbol, confidence, streak=0):
        """Dynamic position sizing based on multiple factors"""
        
        base_size = Config.POSITION_SIZE
        
        # Increase size on winning streak
        if streak > 3:
            base_size *= 2
        elif streak > 5:
            base_size *= 3
        elif streak > 10:
            base_size *= 5
        
        # Increase size on high confidence
        if confidence > 0.8:
            base_size *= 2
        elif confidence > 0.9:
            base_size *= 3
        
        # Time-based sizing (more aggressive during good sessions)
        hour = datetime.now().hour
        if 13 <= hour <= 17:  # London/NY overlap
            base_size *= 1.5
        
        return min(base_size, 1.0)  # Max 1 lot
    
    def pyramid_trade(self, symbol, direction, current_profit):
        """Add to winning positions"""
        
        if current_profit > 5:  # If profitable, add more
            return {
                'action': 'pyramid',
                'size_multiplier': 1.5,
                'reason': 'adding_to_winner'
            }
        return None
    
    def martingale_recovery(self, symbol, last_loss):
        """Double down after losses (risky!)"""
        
        if Config.USE_MARTINGALE and last_loss < 0:
            return {
                'action': 'martingale',
                'size_multiplier': 2,
                'reason': 'loss_recovery'
            }
        return None

# ============================================================================
# MULTI-STRATEGY ENGINE
# ============================================================================

class MultiStrategy:
    """Run multiple strategies simultaneously"""
    
    def __init__(self):
        self.strategies = {
            'scalper': self.scalp_strategy,
            'momentum': self.momentum_strategy,
            'reversal': self.reversal_strategy,
            'breakout': self.breakout_strategy,
            'grid': self.grid_strategy
        }
        self.active_strategies = []
    
    def get_all_signals(self, symbol):
        """Get signals from all strategies"""
        
        signals = []
        
        for name, strategy in self.strategies.items():
            signal = strategy(symbol)
            if signal:
                signal['strategy'] = name
                signals.append(signal)
        
        return signals
    
    def scalp_strategy(self, symbol):
        """Ultra-fast scalping"""
        
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 5)
        if rates is None or len(rates) < 5:
            return None
        
        # Quick momentum
        change = (rates[-1]['close'] - rates[-2]['close']) / rates[-2]['close']
        
        if abs(change) > 0.00005:  # Tiny movement
            return {
                'type': 'BUY' if change > 0 else 'SELL',
                'confidence': 0.6,
                'tp_pips': Config.SCALP_PROFIT,
                'sl_pips': 5
            }
        return None
    
    def momentum_strategy(self, symbol):
        """Follow strong momentum"""
        
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 20)
        if rates is None or len(rates) < 20:
            return None
        
        df = pd.DataFrame(rates)
        
        # Strong momentum
        momentum = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
        
        if abs(momentum) > 0.001:  # 0.1% move
            return {
                'type': 'BUY' if momentum > 0 else 'SELL',
                'confidence': 0.7,
                'tp_pips': Config.NORMAL_PROFIT,
                'sl_pips': 10
            }
        return None
    
    def reversal_strategy(self, symbol):
        """Catch reversals at extremes"""
        
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 30)
        if rates is None or len(rates) < 30:
            return None
        
        df = pd.DataFrame(rates)
        
        # RSI for overbought/oversold
        rsi = self._calculate_rsi(df['close'])
        
        if rsi > 70:
            return {
                'type': 'SELL',
                'confidence': 0.65,
                'tp_pips': Config.NORMAL_PROFIT,
                'sl_pips': 10
            }
        elif rsi < 30:
            return {
                'type': 'BUY',
                'confidence': 0.65,
                'tp_pips': Config.NORMAL_PROFIT,
                'sl_pips': 10
            }
        return None
    
    def breakout_strategy(self, symbol):
        """Trade breakouts"""
        
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 20)
        if rates is None or len(rates) < 20:
            return None
        
        df = pd.DataFrame(rates)
        
        # Recent high/low
        recent_high = df['high'].iloc[-10:].max()
        recent_low = df['low'].iloc[-10:].min()
        current = df['close'].iloc[-1]
        
        if current > recent_high:
            return {
                'type': 'BUY',
                'confidence': 0.7,
                'tp_pips': Config.RUNNER_PROFIT,
                'sl_pips': 10
            }
        elif current < recent_low:
            return {
                'type': 'SELL',
                'confidence': 0.7,
                'tp_pips': Config.RUNNER_PROFIT,
                'sl_pips': 10
            }
        return None
    
    def grid_strategy(self, symbol):
        """Grid trading system"""
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return None
        
        # Random grid levels (simplified)
        if random.random() > 0.7:  # 30% chance
            return {
                'type': 'BUY' if random.random() > 0.5 else 'SELL',
                'confidence': 0.5,
                'tp_pips': Config.SCALP_PROFIT,
                'sl_pips': 10
            }
        return None
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs)).iloc[-1]

# ============================================================================
# TRADE EXECUTOR
# ============================================================================

class HyperExecutor:
    """Execute trades at maximum speed"""
    
    def __init__(self):
        self.pending_orders = []
        self.executed_trades = []
        self.failed_trades = []
    
    def execute_signal(self, symbol, signal):
        """Execute trade signal immediately"""
        
        info = mt5.symbol_info(symbol)
        if not info:
            return False
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
        
        # Check spread
        spread = (tick.ask - tick.bid) / info.point / 10
        if spread > 5:  # Max 5 pip spread
            return False
        
        # Determine parameters
        if signal['type'] == 'BUY':
            price = tick.ask
            sl = price - signal['sl_pips'] * info.point * 10
            tp = price + signal['tp_pips'] * info.point * 10
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl = price + signal['sl_pips'] * info.point * 10
            tp = price - signal['tp_pips'] * info.point * 10
            order_type = mt5.ORDER_TYPE_SELL
        
        # Dynamic sizing
        volume = Config.POSITION_SIZE
        if signal['confidence'] > 0.7:
            volume *= 2
        
        volume = min(volume, 0.5)  # Safety limit
        
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': volume,
            'type': order_type,
            'price': price,
            'sl': round(sl, info.digits),
            'tp': round(tp, info.digits),
            'deviation': 20,
            'magic': Config.MAGIC,
            'comment': signal.get('strategy', 'hyper'),
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.executed_trades.append({
                'time': datetime.now(),
                'symbol': symbol,
                'type': signal['type'],
                'volume': volume,
                'strategy': signal.get('strategy', 'unknown')
            })
            return True
        else:
            self.failed_trades.append({
                'symbol': symbol,
                'error': result.comment if result else 'Unknown'
            })
        
        return False

# ============================================================================
# PERFORMANCE TRACKER
# ============================================================================

class PerformanceTracker:
    """Track and display performance"""
    
    def __init__(self):
        self.start_balance = 0
        self.peak_balance = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0
        self.strategies_profit = defaultdict(float)
    
    def update(self, account, positions):
        """Update performance metrics"""
        
        if self.start_balance == 0:
            self.start_balance = account.balance
            self.peak_balance = account.balance
        
        if account.balance > self.peak_balance:
            self.peak_balance = account.balance
        
        current_profit = account.balance - self.start_balance
        self.total_profit = current_profit
        
        return {
            'start_balance': self.start_balance,
            'current_balance': account.balance,
            'peak_balance': self.peak_balance,
            'total_profit': current_profit,
            'drawdown': (self.peak_balance - account.balance),
            'roi': (current_profit / self.start_balance * 100) if self.start_balance > 0 else 0
        }
    
    def display_stats(self, stats):
        """Display performance statistics"""
        
        print(f"\r[HYPER] Bal: ${stats['current_balance']:.2f} | "
              f"P&L: ${stats['total_profit']:+.2f} | "
              f"ROI: {stats['roi']:+.1f}% | "
              f"Peak: ${stats['peak_balance']:.2f} | "
              f"DD: ${stats['drawdown']:.2f}     ", end='')

# ============================================================================
# HYPERTRADER MAIN CLASS
# ============================================================================

class HyperTrader:
    """Main 24/7 trading system"""
    
    def __init__(self):
        self.connected = False
        self.profit_maximizer = ProfitMaximizer()
        self.multi_strategy = MultiStrategy()
        self.executor = HyperExecutor()
        self.tracker = PerformanceTracker()
        
        self.last_trade_time = {}
        self.symbol_performance = defaultdict(float)
        
        self.initialize()
    
    def initialize(self):
        """Initialize MT5"""
        
        if not mt5.initialize():
            print("[ERROR] MT5 initialization failed")
            return
        
        if not mt5.login(Config.LOGIN, Config.PASSWORD, Config.SERVER):
            print("[ERROR] Login failed")
            mt5.shutdown()
            return
        
        account = mt5.account_info()
        if account:
            self.connected = True
            print(f"\n[HYPERTRADER] System Online")
            print(f"[ACCOUNT] Starting Balance: ${account.balance:.2f}")
            print(f"[MODE] 24/7 Ultra-Aggressive Trading")
    
    def scan_all_markets(self):
        """Scan all symbols for opportunities"""
        
        opportunities = []
        
        for symbol in Config.SYMBOLS:
            # Skip if too recent trade
            if symbol in self.last_trade_time:
                if (datetime.now() - self.last_trade_time[symbol]).total_seconds() < 30:
                    continue
            
            # Get all strategy signals
            signals = self.multi_strategy.get_all_signals(symbol)
            
            for signal in signals:
                signal['symbol'] = symbol
                opportunities.append(signal)
        
        # Sort by confidence
        opportunities.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return opportunities
    
    def manage_positions(self):
        """Aggressive position management"""
        
        positions = mt5.positions_get()
        if not positions:
            return
        
        for pos in positions:
            if pos.magic != Config.MAGIC:
                continue
            
            # Quick profit taking
            if pos.profit > 2:  # $2 profit
                self.close_position(pos, "quick_profit")
            
            # Add to winners (pyramiding)
            elif pos.profit > 5:
                pyramid = self.profit_maximizer.pyramid_trade(
                    pos.symbol, 
                    'BUY' if pos.type == 0 else 'SELL',
                    pos.profit
                )
                if pyramid:
                    # Open additional position
                    signal = {
                        'type': 'BUY' if pos.type == 0 else 'SELL',
                        'confidence': 0.8,
                        'tp_pips': Config.RUNNER_PROFIT,
                        'sl_pips': 10,
                        'strategy': 'pyramid'
                    }
                    self.executor.execute_signal(pos.symbol, signal)
            
            # Cut losses
            elif pos.profit < -10:  # $10 loss
                self.close_position(pos, "stop_loss")
    
    def close_position(self, position, reason):
        """Close position"""
        
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
            'deviation': 20,
            'magic': Config.MAGIC,
            'comment': reason,
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK
        }
        
        mt5.order_send(request)
        
        # Update tracking
        self.symbol_performance[position.symbol] += position.profit
        if position.profit > 0:
            self.tracker.winning_trades += 1
            self.profit_maximizer.winning_streaks[position.symbol] += 1
        else:
            self.profit_maximizer.winning_streaks[position.symbol] = 0
    
    def run(self):
        """Main 24/7 trading loop"""
        
        if not self.connected:
            return
        
        print("\n" + "="*70)
        print("HYPERTRADER - 24/7 MAXIMUM PROFIT MODE")
        print("="*70)
        print(f"• Trading {len(Config.SYMBOLS)} symbols")
        print(f"• 5 strategies running simultaneously")
        print(f"• {Config.SCAN_INTERVAL_MS}ms scan rate")
        print(f"• Pyramiding + Grid + Scalping + Momentum")
        print("="*70)
        print("\nTrading non-stop...\n")
        
        cycle = 0
        
        try:
            while True:
                cycle += 1
                
                # Get account info
                account = mt5.account_info()
                if not account:
                    continue
                
                positions = mt5.positions_get()
                current_positions = len([p for p in positions if p.magic == Config.MAGIC]) if positions else 0
                
                # Find opportunities
                if current_positions < Config.MAX_POSITIONS:
                    opportunities = self.scan_all_markets()
                    
                    for opp in opportunities[:3]:  # Take top 3 signals
                        if current_positions >= Config.MAX_POSITIONS:
                            break
                        
                        success = self.executor.execute_signal(opp['symbol'], opp)
                        if success:
                            self.last_trade_time[opp['symbol']] = datetime.now()
                            self.tracker.total_trades += 1
                            current_positions += 1
                
                # Manage positions
                self.manage_positions()
                
                # Update performance
                if cycle % 10 == 0:
                    stats = self.tracker.update(account, positions)
                    self.tracker.display_stats(stats)
                
                # Show executed trades
                if cycle % 100 == 0 and self.executor.executed_trades:
                    recent = self.executor.executed_trades[-1]
                    print(f"\n[TRADE] {recent['time'].strftime('%H:%M:%S')} | "
                          f"{recent['symbol']} | {recent['type']} | "
                          f"Strategy: {recent['strategy']}")
                
                # Ultra-fast scanning
                time.sleep(Config.SCAN_INTERVAL_MS / 1000)
                
        except KeyboardInterrupt:
            self._shutdown()
        finally:
            mt5.shutdown()
    
    def _shutdown(self):
        """Shutdown with report"""
        
        print(f"\n\n[SHUTDOWN] HyperTrader Offline")
        print(f"\n[FINAL REPORT]")
        print(f"Total Trades Executed: {self.tracker.total_trades}")
        print(f"Winning Trades: {self.tracker.winning_trades}")
        
        if self.tracker.total_trades > 0:
            win_rate = self.tracker.winning_trades / self.tracker.total_trades * 100
            print(f"Win Rate: {win_rate:.1f}%")
        
        print(f"\n[SYMBOL PERFORMANCE]")
        for symbol, profit in sorted(self.symbol_performance.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"{symbol}: ${profit:+.2f}")
        
        print(f"\n[STRATEGY BREAKDOWN]")
        print(f"Executed Trades: {len(self.executor.executed_trades)}")
        print(f"Failed Trades: {len(self.executor.failed_trades)}")

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("INITIALIZING HYPERTRADER...")
    print("="*70)
    
    trader = HyperTrader()
    
    if trader.connected:
        trader.run()
    else:
        print("[ERROR] Failed to initialize HyperTrader")