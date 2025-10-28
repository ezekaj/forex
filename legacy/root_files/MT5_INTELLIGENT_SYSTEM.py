"""
MT5 INTELLIGENT SYSTEM - Trading + Learning Simultaneously
===========================================================
Real-time trading with continuous learning and adaptation
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
import threading
import queue
from collections import deque, defaultdict
import random
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SHARED CONFIGURATION
# ============================================================================

class GlobalConfig:
    """Shared configuration for all components"""
    
    # Account
    LOGIN = 95948709
    PASSWORD = 'To-4KyLg'
    SERVER = 'MetaQuotes-Demo'
    MAGIC = 111111
    
    # Files for persistence
    MEMORY_FILE = 'intelligent_memory.json'
    CONFIG_FILE = 'intelligent_config.json'
    PERFORMANCE_FILE = 'intelligent_performance.json'
    
    # Initial Settings
    BASE_LOT = 0.02
    MAX_POSITIONS = 10
    SCAN_INTERVAL = 0.5  # 500ms for fast reaction

# ============================================================================
# SHARED MEMORY (Thread-Safe)
# ============================================================================

class SharedMemory:
    """Thread-safe shared memory for all components"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.data = self.load_memory()
        
    def load_memory(self):
        """Load memory from disk"""
        if os.path.exists(GlobalConfig.MEMORY_FILE):
            try:
                with open(GlobalConfig.MEMORY_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Default structure
        return {
            'patterns': {},
            'symbols': {},
            'indicators': {
                'rsi': 1.0, 'macd': 1.0, 'ema': 1.0, 'bollinger': 1.0,
                'stochastic': 1.0, 'momentum': 1.0, 'volume': 1.0
            },
            'timeframes': {},
            'active_trades': {},
            'performance': {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'profit': 0
            },
            'config': {
                'min_confidence': 0.6,
                'min_signals': 2,
                'stop_loss': 12,
                'take_profit': 18,
                'max_spread': 3
            }
        }
    
    def update(self, key, value):
        """Thread-safe update"""
        with self.lock:
            if '.' in key:  # Nested key
                keys = key.split('.')
                ref = self.data
                for k in keys[:-1]:
                    if k not in ref:
                        ref[k] = {}
                    ref = ref[k]
                ref[keys[-1]] = value
            else:
                self.data[key] = value
    
    def get(self, key, default=None):
        """Thread-safe get"""
        with self.lock:
            if '.' in key:  # Nested key
                keys = key.split('.')
                ref = self.data
                for k in keys:
                    if k not in ref:
                        return default
                    ref = ref[k]
                return ref
            return self.data.get(key, default)
    
    def save(self):
        """Save memory to disk"""
        with self.lock:
            with open(GlobalConfig.MEMORY_FILE, 'w') as f:
                json.dump(self.data, f, indent=2, default=str)

# ============================================================================
# LEARNING ENGINE (Runs in separate thread)
# ============================================================================

class LearningEngine(threading.Thread):
    """Continuous learning engine that runs in background"""
    
    def __init__(self, memory, message_queue):
        super().__init__()
        self.memory = memory
        self.message_queue = message_queue
        self.running = True
        self.daemon = True
        
    def run(self):
        """Main learning loop"""
        print("[LEARNING] Engine started")
        
        while self.running:
            try:
                # Process messages from trading bot
                while not self.message_queue.empty():
                    msg = self.message_queue.get()
                    self.process_message(msg)
                
                # Periodic analysis
                self.analyze_performance()
                self.optimize_parameters()
                
                # Save memory
                self.memory.save()
                
                # Sleep
                time.sleep(5)  # Analyze every 5 seconds
                
            except Exception as e:
                print(f"[LEARNING] Error: {e}")
    
    def process_message(self, msg):
        """Process message from trading bot"""
        
        if msg['type'] == 'trade_opened':
            # Record new trade
            trade_id = msg['trade_id']
            self.memory.update(f'active_trades.{trade_id}', msg['data'])
            
        elif msg['type'] == 'trade_closed':
            # Learn from closed trade
            self.learn_from_trade(msg['data'])
            
        elif msg['type'] == 'signal_generated':
            # Record signal for analysis
            self.analyze_signal(msg['data'])
    
    def learn_from_trade(self, trade_data):
        """Learn from a completed trade"""
        
        # Update pattern performance
        pattern_key = f"{trade_data['setup']}_{trade_data['hour']}"
        patterns = self.memory.get('patterns', {})
        
        if pattern_key not in patterns:
            patterns[pattern_key] = {'wins': 0, 'losses': 0, 'profit': 0}
        
        if trade_data['profit'] > 0:
            patterns[pattern_key]['wins'] += 1
            self.memory.update('performance.wins', self.memory.get('performance.wins', 0) + 1)
        else:
            patterns[pattern_key]['losses'] += 1
            self.memory.update('performance.losses', self.memory.get('performance.losses', 0) + 1)
        
        patterns[pattern_key]['profit'] += trade_data['profit']
        self.memory.update('patterns', patterns)
        
        # Update symbol performance
        symbols = self.memory.get('symbols', {})
        symbol = trade_data['symbol']
        
        if symbol not in symbols:
            symbols[symbol] = {'trades': 0, 'profit': 0, 'wins': 0}
        
        symbols[symbol]['trades'] += 1
        symbols[symbol]['profit'] += trade_data['profit']
        if trade_data['profit'] > 0:
            symbols[symbol]['wins'] += 1
        
        self.memory.update('symbols', symbols)
        
        # Update indicator weights
        if 'indicators' in trade_data:
            indicators = self.memory.get('indicators', {})
            
            for indicator in trade_data['indicators']:
                if indicator in indicators:
                    # Adjust weight based on success
                    if trade_data['profit'] > 0:
                        indicators[indicator] = min(2.0, indicators[indicator] * 1.02)
                    else:
                        indicators[indicator] = max(0.5, indicators[indicator] * 0.98)
            
            self.memory.update('indicators', indicators)
        
        # Update timeframe performance
        timeframes = self.memory.get('timeframes', {})
        hour = str(trade_data['hour'])
        
        if hour not in timeframes:
            timeframes[hour] = {'trades': 0, 'profit': 0}
        
        timeframes[hour]['trades'] += 1
        timeframes[hour]['profit'] += trade_data['profit']
        
        self.memory.update('timeframes', timeframes)
        
        # Update overall performance
        self.memory.update('performance.total_trades', 
                          self.memory.get('performance.total_trades', 0) + 1)
        self.memory.update('performance.profit', 
                          self.memory.get('performance.profit', 0) + trade_data['profit'])
        
        print(f"[LEARNING] Learned from {trade_data['symbol']} trade: "
              f"{'WIN' if trade_data['profit'] > 0 else 'LOSS'} ${trade_data['profit']:.2f}")
    
    def analyze_signal(self, signal_data):
        """Analyze generated signals for patterns"""
        # Store signal patterns for future analysis
        pass
    
    def analyze_performance(self):
        """Analyze overall performance and identify issues"""
        
        performance = self.memory.get('performance', {})
        
        if performance.get('total_trades', 0) < 10:
            return  # Need more data
        
        win_rate = performance.get('wins', 0) / performance['total_trades']
        
        # Print learning status
        if performance['total_trades'] % 10 == 0:
            print(f"[LEARNING] Performance: {performance['wins']}W/{performance.get('losses', 0)}L "
                  f"({win_rate:.1%}) | P&L: ${performance.get('profit', 0):.2f}")
    
    def optimize_parameters(self):
        """Optimize trading parameters based on performance"""
        
        performance = self.memory.get('performance', {})
        config = self.memory.get('config', {})
        
        if performance.get('total_trades', 0) < 20:
            return  # Need more data
        
        win_rate = performance.get('wins', 0) / performance['total_trades']
        
        # Adjust confidence requirement
        if win_rate < 0.35:
            # Losing too much - increase requirements
            config['min_confidence'] = min(0.8, config.get('min_confidence', 0.6) + 0.05)
            config['min_signals'] = min(4, config.get('min_signals', 2) + 1)
            print(f"[LEARNING] Tightening requirements - Win rate: {win_rate:.1%}")
            
        elif win_rate > 0.6:
            # Doing well - can loosen slightly
            config['min_confidence'] = max(0.55, config.get('min_confidence', 0.6) - 0.02)
            print(f"[LEARNING] Optimizing for more trades - Win rate: {win_rate:.1%}")
        
        # Adjust risk/reward based on results
        if performance.get('profit', 0) < -50:
            # Losing money - adjust stops
            config['stop_loss'] = max(8, config.get('stop_loss', 12) - 2)
            config['take_profit'] = min(25, config.get('take_profit', 18) + 2)
            print(f"[LEARNING] Adjusting risk/reward for better R:R")
        
        self.memory.update('config', config)
    
    def stop(self):
        """Stop the learning engine"""
        self.running = False

# ============================================================================
# TRADING BOT (Main thread)
# ============================================================================

class IntelligentTrader:
    """Main trading bot that uses learned knowledge"""
    
    def __init__(self, memory, message_queue):
        self.memory = memory
        self.message_queue = message_queue
        self.start_balance = 0
        self.trade_counter = 0
        
        # All forex pairs
        self.symbols = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
            'NZDUSD', 'USDCHF', 'EURJPY', 'GBPJPY', 'EURGBP',
            'EURAUD', 'EURCAD', 'GBPAUD', 'AUDNZD', 'CADJPY'
        ]
    
    def initialize(self):
        """Initialize MT5"""
        
        if not mt5.initialize():
            return False
        
        if not mt5.login(GlobalConfig.LOGIN, GlobalConfig.PASSWORD, GlobalConfig.SERVER):
            mt5.shutdown()
            return False
        
        account = mt5.account_info()
        if account:
            self.start_balance = account.balance
            return True
        
        return False
    
    def analyze_symbol(self, symbol):
        """Analyze symbol using learned weights"""
        
        # Get market data
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
        if rates is None or len(rates) < 100:
            return None
        
        df = pd.DataFrame(rates)
        
        # Get indicator weights from memory
        weights = self.memory.get('indicators', {})
        
        signals = []
        indicators_used = []
        
        # Calculate indicators with weights
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        if rsi < 30:
            signals.append({'type': 'BUY', 'strength': (30-rsi)/30 * weights.get('rsi', 1.0)})
            indicators_used.append('rsi')
        elif rsi > 70:
            signals.append({'type': 'SELL', 'strength': (rsi-70)/30 * weights.get('rsi', 1.0)})
            indicators_used.append('rsi')
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        
        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            signals.append({'type': 'BUY', 'strength': 0.7 * weights.get('macd', 1.0)})
            indicators_used.append('macd')
        elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            signals.append({'type': 'SELL', 'strength': 0.7 * weights.get('macd', 1.0)})
            indicators_used.append('macd')
        
        # EMA Cross
        if ema12.iloc[-1] > ema26.iloc[-1]:
            signals.append({'type': 'BUY', 'strength': 0.6 * weights.get('ema', 1.0)})
            indicators_used.append('ema')
        else:
            signals.append({'type': 'SELL', 'strength': 0.6 * weights.get('ema', 1.0)})
            indicators_used.append('ema')
        
        # Bollinger Bands
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        upper_band = sma20 + 2 * std20
        lower_band = sma20 - 2 * std20
        
        if df['close'].iloc[-1] < lower_band.iloc[-1]:
            signals.append({'type': 'BUY', 'strength': 0.7 * weights.get('bollinger', 1.0)})
            indicators_used.append('bollinger')
        elif df['close'].iloc[-1] > upper_band.iloc[-1]:
            signals.append({'type': 'SELL', 'strength': 0.7 * weights.get('bollinger', 1.0)})
            indicators_used.append('bollinger')
        
        # Momentum
        momentum = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
        if abs(momentum) > 0.001:
            if momentum > 0:
                signals.append({'type': 'BUY', 'strength': min(momentum*100, 1.0) * weights.get('momentum', 1.0)})
            else:
                signals.append({'type': 'SELL', 'strength': min(abs(momentum)*100, 1.0) * weights.get('momentum', 1.0)})
            indicators_used.append('momentum')
        
        # Volume
        vol_ratio = df['tick_volume'].iloc[-1] / df['tick_volume'].rolling(20).mean().iloc[-1]
        if vol_ratio > 1.5:
            if df['close'].iloc[-1] > df['open'].iloc[-1]:
                signals.append({'type': 'BUY', 'strength': 0.5 * weights.get('volume', 1.0)})
            else:
                signals.append({'type': 'SELL', 'strength': 0.5 * weights.get('volume', 1.0)})
            indicators_used.append('volume')
        
        # Aggregate signals
        if not signals:
            return None
        
        buy_signals = [s for s in signals if s['type'] == 'BUY']
        sell_signals = [s for s in signals if s['type'] == 'SELL']
        
        # Get config from memory
        config = self.memory.get('config', {})
        min_signals = config.get('min_signals', 2)
        
        # Check pattern confidence
        hour = datetime.now().hour
        pattern_key = f"trend_{hour}"  # Simplified pattern key
        patterns = self.memory.get('patterns', {})
        
        pattern_confidence = 0.5  # Default
        if pattern_key in patterns:
            p = patterns[pattern_key]
            if p['wins'] + p['losses'] > 0:
                pattern_confidence = p['wins'] / (p['wins'] + p['losses'])
        
        # Get symbol performance
        symbols = self.memory.get('symbols', {})
        symbol_confidence = 1.0
        if symbol in symbols:
            s = symbols[symbol]
            if s['trades'] > 5:
                symbol_confidence = 0.5 + (s['wins'] / s['trades'])
        
        # Determine signal
        if len(buy_signals) >= min_signals:
            total_strength = sum(s['strength'] for s in buy_signals)
            confidence = (total_strength / len(signals)) * pattern_confidence * symbol_confidence
            
            return {
                'type': 'BUY',
                'confidence': min(confidence, 0.95),
                'setup': 'trend',
                'indicators': indicators_used,
                'num_signals': len(buy_signals)
            }
        
        elif len(sell_signals) >= min_signals:
            total_strength = sum(s['strength'] for s in sell_signals)
            confidence = (total_strength / len(signals)) * pattern_confidence * symbol_confidence
            
            return {
                'type': 'SELL',
                'confidence': min(confidence, 0.95),
                'setup': 'trend',
                'indicators': indicators_used,
                'num_signals': len(sell_signals)
            }
        
        return None
    
    def execute_trade(self, symbol, signal):
        """Execute trade and notify learning engine"""
        
        # Get config from memory
        config = self.memory.get('config', {})
        min_confidence = config.get('min_confidence', 0.6)
        
        if signal['confidence'] < min_confidence:
            return False
        
        # Check spread
        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        
        if not tick or not info:
            return False
        
        spread_points = tick.ask - tick.bid
        spread_pips = spread_points / (info.point * 10)
        
        max_spread = config.get('max_spread', 3)
        if 'JPY' in symbol:
            max_spread *= 1.5
        
        if spread_pips > max_spread:
            return False
        
        # Setup trade
        if signal['type'] == 'BUY':
            price = tick.ask
            sl = price - config.get('stop_loss', 12) * info.point * 10
            tp = price + config.get('take_profit', 18) * info.point * 10
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl = price + config.get('stop_loss', 12) * info.point * 10
            tp = price - config.get('take_profit', 18) * info.point * 10
            order_type = mt5.ORDER_TYPE_SELL
        
        # Dynamic lot size based on performance
        performance = self.memory.get('performance', {})
        lot_size = GlobalConfig.BASE_LOT
        
        if performance.get('total_trades', 0) > 10:
            win_rate = performance.get('wins', 0) / performance['total_trades']
            if win_rate > 0.6 and signal['confidence'] > 0.75:
                lot_size = min(lot_size * 1.5, 0.05)
            elif win_rate < 0.35:
                lot_size = max(lot_size * 0.5, 0.01)
        
        lot_size = round(lot_size, 2)
        
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
            'magic': GlobalConfig.MAGIC,
            'comment': f"{signal['setup']}_{signal['confidence']:.2f}",
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.trade_counter += 1
            
            # Notify learning engine
            trade_data = {
                'trade_id': result.order,
                'symbol': symbol,
                'type': signal['type'],
                'setup': signal['setup'],
                'entry_price': price,
                'entry_time': datetime.now(),
                'hour': datetime.now().hour,
                'confidence': signal['confidence'],
                'indicators': signal.get('indicators', []),
                'lot_size': lot_size
            }
            
            self.message_queue.put({
                'type': 'trade_opened',
                'trade_id': result.order,
                'data': trade_data
            })
            
            # Update performance counter
            perf = self.memory.get('performance', {})
            perf['total_trades'] = perf.get('total_trades', 0) + 1
            self.memory.update('performance', perf)
            
            print(f"\n[TRADE #{self.trade_counter}] {signal['type']} {symbol}")
            print(f"  Confidence: {signal['confidence']:.1%} | Signals: {signal['num_signals']}")
            print(f"  Lot: {lot_size} | Entry: {price:.5f}")
            
            return True
        
        return False
    
    def manage_positions(self):
        """Manage open positions and learn from closed ones"""
        
        positions = mt5.positions_get(magic=GlobalConfig.MAGIC)
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
            if pips >= 8:
                if pos.type == 0:  # BUY
                    new_sl = tick.bid - 5 * info.point * 10
                    if new_sl > pos.sl:
                        request = {
                            'action': mt5.TRADE_ACTION_SLTP,
                            'symbol': pos.symbol,
                            'position': pos.ticket,
                            'sl': round(new_sl, info.digits),
                            'tp': pos.tp,
                            'magic': GlobalConfig.MAGIC
                        }
                        mt5.order_send(request)
                else:  # SELL
                    new_sl = tick.ask + 5 * info.point * 10
                    if new_sl < pos.sl:
                        request = {
                            'action': mt5.TRADE_ACTION_SLTP,
                            'symbol': pos.symbol,
                            'position': pos.ticket,
                            'sl': round(new_sl, info.digits),
                            'tp': pos.tp,
                            'magic': GlobalConfig.MAGIC
                        }
                        mt5.order_send(request)
    
    def check_closed_trades(self):
        """Check for closed trades and send to learning engine"""
        
        # Get recent history
        from_date = datetime.now() - timedelta(hours=1)
        deals = mt5.history_deals_get(from_date, datetime.now())
        
        if not deals:
            return
        
        active_trades = self.memory.get('active_trades', {})
        
        for deal in deals:
            if deal.magic != GlobalConfig.MAGIC or deal.entry != 1:
                continue
            
            # Check if this is a closing deal
            trade_id = str(deal.order)
            if trade_id in active_trades:
                # Get trade info
                trade_info = active_trades[trade_id]
                
                # Calculate profit and pips
                profit = deal.profit
                symbol = deal.symbol
                info = mt5.symbol_info(symbol)
                
                if info and 'entry_price' in trade_info:
                    if trade_info['type'] == 'BUY':
                        pips = (deal.price - trade_info['entry_price']) / (info.point * 10)
                    else:
                        pips = (trade_info['entry_price'] - deal.price) / (info.point * 10)
                else:
                    pips = 0
                
                # Send to learning engine
                trade_result = {
                    'symbol': symbol,
                    'type': trade_info.get('type', 'UNKNOWN'),
                    'setup': trade_info.get('setup', 'unknown'),
                    'profit': profit,
                    'pips': pips,
                    'entry_time': trade_info.get('entry_time', datetime.now()),
                    'exit_time': datetime.now(),
                    'hour': trade_info.get('hour', datetime.now().hour),
                    'indicators': trade_info.get('indicators', [])
                }
                
                self.message_queue.put({
                    'type': 'trade_closed',
                    'data': trade_result
                })
                
                # Remove from active trades
                active_trades.pop(trade_id, None)
                self.memory.update('active_trades', active_trades)
                
                # Print result
                if profit > 0:
                    print(f"[CLOSE] WIN {symbol} +${profit:.2f} (+{pips:.1f} pips)")
                else:
                    print(f"[CLOSE] LOSS {symbol} ${profit:.2f} ({pips:.1f} pips)")
    
    def run(self):
        """Main trading loop"""
        
        if not self.initialize():
            return False
        
        account = mt5.account_info()
        print(f"\n{'='*70}")
        print(f"INTELLIGENT SYSTEM STARTED")
        print(f"{'='*70}")
        print(f"Balance: ${account.balance:.2f}")
        print(f"Trading + Learning Simultaneously")
        print(f"{'='*70}\n")
        
        last_status = time.time()
        
        try:
            while True:
                # Check positions
                self.manage_positions()
                self.check_closed_trades()
                
                # Check position limit
                positions = mt5.positions_get(magic=GlobalConfig.MAGIC)
                if positions and len(positions) >= GlobalConfig.MAX_POSITIONS:
                    time.sleep(GlobalConfig.SCAN_INTERVAL)
                    continue
                
                # Scan symbols
                for symbol in self.symbols:
                    # Check limit again
                    positions = mt5.positions_get(magic=GlobalConfig.MAGIC)
                    if positions and len(positions) >= GlobalConfig.MAX_POSITIONS:
                        break
                    
                    # Analyze
                    signal = self.analyze_symbol(symbol)
                    
                    if signal:
                        # Send signal to learning engine for analysis
                        self.message_queue.put({
                            'type': 'signal_generated',
                            'data': signal
                        })
                        
                        # Try to execute
                        if self.execute_trade(symbol, signal):
                            time.sleep(2)  # Brief pause after trade
                
                # Display status
                if time.time() - last_status > 30:
                    self.display_status()
                    last_status = time.time()
                
                # Sleep
                time.sleep(GlobalConfig.SCAN_INTERVAL)
                
        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Stopping system...")
            return True
        
        return True
    
    def display_status(self):
        """Display current status"""
        
        account = mt5.account_info()
        if not account:
            return
        
        positions = mt5.positions_get(magic=GlobalConfig.MAGIC)
        performance = self.memory.get('performance', {})
        
        profit = account.balance - self.start_balance
        roi = (profit / self.start_balance * 100) if self.start_balance > 0 else 0
        
        win_rate = 0
        if performance.get('total_trades', 0) > 0:
            win_rate = performance.get('wins', 0) / performance['total_trades'] * 100
        
        print(f"\n{'='*70}")
        print(f"INTELLIGENT SYSTEM STATUS")
        print(f"{'='*70}")
        print(f"Balance: ${account.balance:.2f} | P&L: ${profit:+.2f} ({roi:+.1f}%)")
        print(f"Trades: {performance.get('total_trades', 0)} | "
              f"Win Rate: {win_rate:.1f}%")
        print(f"Open: {len(positions) if positions else 0}/{GlobalConfig.MAX_POSITIONS}")
        
        # Show learned patterns
        patterns = self.memory.get('patterns', {})
        if patterns:
            top_patterns = sorted(patterns.items(), 
                                key=lambda x: x[1].get('wins', 0) / max(1, x[1].get('wins', 0) + x[1].get('losses', 0)),
                                reverse=True)[:3]
            
            if top_patterns:
                print(f"\nTop Patterns:")
                for pattern, stats in top_patterns:
                    total = stats.get('wins', 0) + stats.get('losses', 0)
                    if total > 0:
                        wr = stats.get('wins', 0) / total * 100
                        print(f"  {pattern}: {wr:.0f}% ({stats.get('wins', 0)}W/{stats.get('losses', 0)}L)")
        
        print(f"{'='*70}")

# ============================================================================
# MAIN SYSTEM CONTROLLER
# ============================================================================

class IntelligentSystem:
    """Main system that coordinates trading and learning"""
    
    def __init__(self):
        self.memory = SharedMemory()
        self.message_queue = queue.Queue()
        self.learning_engine = None
        self.trader = None
    
    def start(self):
        """Start the complete system"""
        
        print("\n" + "="*70)
        print("MT5 INTELLIGENT SYSTEM")
        print("Real-time Trading + Continuous Learning")
        print("="*70)
        
        # Start learning engine in background
        self.learning_engine = LearningEngine(self.memory, self.message_queue)
        self.learning_engine.start()
        
        # Start trading bot in main thread
        self.trader = IntelligentTrader(self.memory, self.message_queue)
        
        try:
            # Run trading bot
            if self.trader.run():
                # Trading completed normally
                self.shutdown()
        except KeyboardInterrupt:
            print("\n[SYSTEM] Shutting down...")
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the system gracefully"""
        
        # Stop learning engine
        if self.learning_engine:
            self.learning_engine.stop()
            self.learning_engine.join(timeout=5)
        
        # Save memory
        self.memory.save()
        
        # Final report
        self.final_report()
        
        # Shutdown MT5
        mt5.shutdown()
    
    def final_report(self):
        """Generate final report"""
        
        performance = self.memory.get('performance', {})
        patterns = self.memory.get('patterns', {})
        symbols = self.memory.get('symbols', {})
        
        print(f"\n{'='*70}")
        print(f"INTELLIGENT SYSTEM - FINAL REPORT")
        print(f"{'='*70}")
        
        print(f"\nPerformance:")
        print(f"  Total Trades: {performance.get('total_trades', 0)}")
        print(f"  Wins: {performance.get('wins', 0)}")
        print(f"  Losses: {performance.get('losses', 0)}")
        
        if performance.get('total_trades', 0) > 0:
            win_rate = performance.get('wins', 0) / performance['total_trades'] * 100
            print(f"  Win Rate: {win_rate:.1f}%")
        
        print(f"  Total P&L: ${performance.get('profit', 0):.2f}")
        
        print(f"\nLearning Summary:")
        print(f"  Patterns Learned: {len(patterns)}")
        print(f"  Symbols Analyzed: {len(symbols)}")
        
        # Top performing symbols
        if symbols:
            top_symbols = sorted(symbols.items(), 
                               key=lambda x: x[1].get('profit', 0),
                               reverse=True)[:3]
            
            print(f"\nTop Symbols:")
            for symbol, stats in top_symbols:
                if stats.get('trades', 0) > 0:
                    avg = stats.get('profit', 0) / stats['trades']
                    print(f"  {symbol}: ${stats.get('profit', 0):.2f} "
                          f"({stats.get('trades', 0)} trades, ${avg:.2f} avg)")
        
        print(f"\n{'='*70}")
        print("All learning data saved for next session!")
        print("The system will be smarter next time!")
        print(f"{'='*70}")

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == '__main__':
    system = IntelligentSystem()
    system.start()