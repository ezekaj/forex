"""
MT5 SUPREME ENHANCED TRADER - Ultra Performance Edition
========================================================
Enhanced version with better balance, smarter position sizing, and improved learning
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
import hashlib
import requests
from collections import deque, defaultdict
import random
import pickle
import warnings
warnings.filterwarnings('ignore')

# Advanced imports
try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_AVAILABLE = True
except:
    VOICE_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential, clone_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    import tensorflow as tf
    NEURAL_AVAILABLE = True
except:
    NEURAL_AVAILABLE = False

# ============================================================================
# ENHANCED SUPREME CONFIGURATION
# ============================================================================

class Config:
    # Account
    LOGIN = 95948709
    PASSWORD = 'To-4KyLg'
    SERVER = 'MetaQuotes-Demo'
    MAGIC = 777777  # New magic number for enhanced version
    
    # Ultra Performance
    SCAN_INTERVAL_MS = 50  # Balanced speed (was 10ms)
    MAX_POSITIONS = 30  # Increased from 20
    BASE_LOT = 0.02  # Keep profitable 0.02 lot size
    MAX_LOT = 0.10  # Conservative max
    
    # Risk Management
    MAX_DAILY_LOSS = 100  # Max $100 daily loss
    MAX_DAILY_PROFIT = 500  # Take profit at $500/day
    TRAILING_STOP_PIPS = 5  # Trail stop after 5 pips profit
    
    # All Trading Pairs (Forex + Metals + Indices if available)
    FOREX_SYMBOLS = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'USDCHF',
        'EURJPY', 'GBPJPY', 'AUDJPY', 'EURAUD', 'EURGBP', 'AUDNZD', 'NZDJPY',
        'CADJPY', 'CHFJPY', 'AUDCAD', 'EURCAD', 'GBPAUD', 'GBPNZD'
    ]
    
    METAL_SYMBOLS = ['XAUUSD', 'XAGUSD']  # Gold and Silver if available
    
    # Neural Evolution
    POPULATION_SIZE = 100  # Increased from 50
    MUTATION_RATE = 0.15
    CROSSOVER_RATE = 0.8
    GENERATIONS = 10000
    
    # Social Trading
    TOP_TRADERS_TO_COPY = 10  # Increased from 5
    COPY_CONFIDENCE_THRESHOLD = 0.65  # Lowered from 0.7
    
    # Enhanced Settings
    ADAPTIVE_SIZING = True  # Dynamic position sizing
    CORRELATION_TRADING = True  # Trade correlated pairs
    MARKET_MAKING = True  # Act as market maker
    SCALPING_MODE = True  # Quick profits
    NEWS_AWARE = True  # React to news
    
    # Balance Control
    BUY_SELL_BALANCE = 0.5  # Target 50/50 buy/sell ratio
    REBALANCE_INTERVAL = 300  # Rebalance every 5 minutes

# ============================================================================
# POSITION BALANCER - Ensures balanced BUY/SELL
# ============================================================================

class PositionBalancer:
    """Maintains balanced buy/sell positions"""
    
    def __init__(self):
        self.position_counts = {'BUY': 0, 'SELL': 0}
        self.position_history = deque(maxlen=1000)
        self.last_rebalance = time.time()
        
    def update_counts(self):
        """Update current position counts"""
        positions = mt5.positions_get(magic=Config.MAGIC)
        
        if positions:
            self.position_counts['BUY'] = sum(1 for p in positions if p.type == 0)
            self.position_counts['SELL'] = sum(1 for p in positions if p.type == 1)
        else:
            self.position_counts = {'BUY': 0, 'SELL': 0}
            
        return self.position_counts
    
    def get_bias_adjustment(self):
        """Calculate bias adjustment to maintain balance"""
        total = self.position_counts['BUY'] + self.position_counts['SELL']
        
        if total == 0:
            return 0  # No bias
        
        buy_ratio = self.position_counts['BUY'] / total
        
        # Calculate adjustment (-1 to 1)
        # Negative = favor SELL, Positive = favor BUY
        if buy_ratio > 0.6:  # Too many BUYs
            return -((buy_ratio - 0.5) * 2)  # Favor SELL
        elif buy_ratio < 0.4:  # Too many SELLs
            return ((0.5 - buy_ratio) * 2)  # Favor BUY
        
        return 0  # Balanced
    
    def should_open_position(self, signal_type):
        """Check if we should open this position type"""
        self.update_counts()
        
        # If no positions, always allow
        if self.position_counts['BUY'] + self.position_counts['SELL'] == 0:
            return True
        
        # Check balance
        if signal_type == 'BUY':
            if self.position_counts['BUY'] > self.position_counts['SELL'] + 5:
                return False  # Too many BUYs
        else:
            if self.position_counts['SELL'] > self.position_counts['BUY'] + 5:
                return False  # Too many SELLs
        
        return True
    
    def force_rebalance(self):
        """Force close positions to maintain balance"""
        positions = mt5.positions_get(magic=Config.MAGIC)
        if not positions:
            return
        
        buy_positions = [p for p in positions if p.type == 0]
        sell_positions = [p for p in positions if p.type == 1]
        
        # Close excess positions
        if len(buy_positions) > len(sell_positions) + 5:
            # Close worst performing BUY
            worst_buy = min(buy_positions, key=lambda p: p.profit)
            self._close_position(worst_buy, "rebalance")
            
        elif len(sell_positions) > len(buy_positions) + 5:
            # Close worst performing SELL
            worst_sell = min(sell_positions, key=lambda p: p.profit)
            self._close_position(worst_sell, "rebalance")
    
    def _close_position(self, position, reason):
        """Close a specific position"""
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
            'comment': f"close_{reason}",
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK
        }
        
        mt5.order_send(request)

# ============================================================================
# ENHANCED NEURAL EVOLUTION with Learning Memory
# ============================================================================

class EnhancedNeuralEvolution:
    """Improved neural network evolution with memory"""
    
    def __init__(self):
        self.population = []
        self.generation = 0
        self.best_network = None
        self.best_fitness = 0
        self.trade_memory = deque(maxlen=10000)  # Remember trades
        self.pattern_memory = {}  # Remember successful patterns
        
        if NEURAL_AVAILABLE:
            self.initialize_population()
            self.load_memory()
    
    def initialize_population(self):
        """Create initial population with diverse strategies"""
        
        for i in range(Config.POPULATION_SIZE):
            strategy_type = random.choice(['scalper', 'swing', 'trend', 'reversal', 'balanced'])
            model = self.create_specialized_network(strategy_type)
            
            self.population.append({
                'id': i,
                'model': model,
                'strategy_type': strategy_type,
                'fitness': 0,
                'trades': 0,
                'profit': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'win_rate': 0,
                'generation_born': 0
            })
    
    def create_specialized_network(self, strategy_type):
        """Create network specialized for strategy type"""
        
        # Different architectures for different strategies
        if strategy_type == 'scalper':
            layers = [64, 32, 16]
        elif strategy_type == 'swing':
            layers = [128, 64, 32]
        elif strategy_type == 'trend':
            layers = [100, 50, 25]
        elif strategy_type == 'reversal':
            layers = [80, 40, 20]
        else:  # balanced
            layers = [96, 48, 24]
        
        model = Sequential()
        
        # Input layer
        model.add(Dense(layers[0], activation='relu', input_shape=(30,)))  # More features
        model.add(Dropout(0.2))
        
        # Hidden layers
        for neurons in layers[1:]:
            model.add(Dense(neurons, activation='relu'))
            model.add(Dropout(0.2))
        
        # Output layer (BUY, SELL, HOLD with confidence)
        model.add(Dense(3, activation='softmax'))
        
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        return model
    
    def predict_balanced(self, features, balancer):
        """Get predictions with balance adjustment"""
        
        if not NEURAL_AVAILABLE or not self.population:
            return None
        
        predictions = []
        bias_adjustment = balancer.get_bias_adjustment()
        
        # Use top performing networks
        top_networks = sorted(self.population, key=lambda x: x['fitness'], reverse=True)[:20]
        
        for individual in top_networks:
            try:
                pred = individual['model'].predict(features.reshape(1, -1), verbose=0)
                action_idx = np.argmax(pred[0])
                confidence = float(pred[0][action_idx])
                
                # Apply balance adjustment
                if action_idx == 0:  # BUY
                    confidence *= (1 + bias_adjustment)
                elif action_idx == 1:  # SELL
                    confidence *= (1 - bias_adjustment)
                
                predictions.append({
                    'network_id': individual['id'],
                    'action': ['BUY', 'SELL', 'HOLD'][action_idx],
                    'confidence': min(max(confidence, 0), 1),
                    'fitness': individual['fitness'],
                    'strategy': individual['strategy_type']
                })
            except:
                pass
        
        if not predictions:
            return None
        
        # Weighted voting with strategy diversity
        buy_score = sum(p['confidence'] * (p['fitness'] + 1) for p in predictions if p['action'] == 'BUY')
        sell_score = sum(p['confidence'] * (p['fitness'] + 1) for p in predictions if p['action'] == 'SELL')
        
        # Normalize scores
        buy_score /= len(predictions)
        sell_score /= len(predictions)
        
        if buy_score > sell_score and buy_score > 0.5:
            return {'type': 'BUY', 'confidence': min(buy_score, 0.95)}
        elif sell_score > buy_score and sell_score > 0.5:
            return {'type': 'SELL', 'confidence': min(sell_score, 0.95)}
        
        return None
    
    def evolve_with_balance(self):
        """Evolve population with balance consideration"""
        
        if not NEURAL_AVAILABLE:
            return
        
        # Calculate balanced fitness
        for individual in self.population:
            if individual['trades'] > 0:
                # Penalize unbalanced trading
                total_trades = individual['buy_trades'] + individual['sell_trades']
                if total_trades > 0:
                    balance_ratio = min(individual['buy_trades'], individual['sell_trades']) / max(individual['buy_trades'], individual['sell_trades'])
                else:
                    balance_ratio = 0
                
                # Fitness combines profit, win rate, and balance
                individual['fitness'] = (
                    individual['profit'] * 0.4 +
                    individual['win_rate'] * 100 * 0.4 +
                    balance_ratio * 100 * 0.2
                )
        
        # Sort by fitness
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Elite selection
        elite_size = Config.POPULATION_SIZE // 5
        new_population = self.population[:elite_size]
        
        # Generate new individuals
        while len(new_population) < Config.POPULATION_SIZE:
            if random.random() < Config.CROSSOVER_RATE:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                child = self.crossover(parent1, parent2)
            else:
                parent = self.tournament_selection()
                child = self.mutate(parent)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        # Save best network
        if self.population[0]['fitness'] > self.best_fitness:
            self.best_fitness = self.population[0]['fitness']
            self.best_network = self.population[0]
            self.save_memory()
    
    def tournament_selection(self, tournament_size=5):
        """Select individual via tournament"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x['fitness'])
    
    def crossover(self, parent1, parent2):
        """Create child from two parents"""
        child = {
            'id': random.randint(10000, 99999),
            'model': clone_model(parent1['model']),
            'strategy_type': random.choice([parent1['strategy_type'], parent2['strategy_type']]),
            'fitness': 0,
            'trades': 0,
            'profit': 0,
            'buy_trades': 0,
            'sell_trades': 0,
            'win_rate': 0,
            'generation_born': self.generation
        }
        
        child['model'].build((None, 30))
        
        # Mix weights
        for i, layer in enumerate(child['model'].layers):
            if hasattr(layer, 'kernel'):
                w1 = parent1['model'].layers[i].get_weights()
                w2 = parent2['model'].layers[i].get_weights()
                
                new_weights = []
                for weight1, weight2 in zip(w1, w2):
                    mask = np.random.random(weight1.shape) > 0.5
                    new_weight = np.where(mask, weight1, weight2)
                    new_weights.append(new_weight)
                
                layer.set_weights(new_weights)
        
        return child
    
    def mutate(self, parent):
        """Mutate an individual"""
        child = {
            'id': random.randint(10000, 99999),
            'model': clone_model(parent['model']),
            'strategy_type': parent['strategy_type'],
            'fitness': 0,
            'trades': 0,
            'profit': 0,
            'buy_trades': 0,
            'sell_trades': 0,
            'win_rate': 0,
            'generation_born': self.generation
        }
        
        child['model'].build((None, 30))
        child['model'].set_weights(parent['model'].get_weights())
        
        # Mutate weights
        for layer in child['model'].layers:
            if hasattr(layer, 'kernel'):
                weights = layer.get_weights()
                new_weights = []
                
                for w in weights:
                    if random.random() < Config.MUTATION_RATE:
                        mutation = np.random.randn(*w.shape) * 0.1
                        w = w + mutation
                    new_weights.append(w)
                
                layer.set_weights(new_weights)
        
        # Chance to change strategy
        if random.random() < 0.1:
            child['strategy_type'] = random.choice(['scalper', 'swing', 'trend', 'reversal', 'balanced'])
        
        return child
    
    def update_fitness(self, network_id, profit, trade_type):
        """Update network fitness with trade result"""
        
        for individual in self.population:
            if individual['id'] == network_id:
                individual['trades'] += 1
                individual['profit'] += profit
                
                if trade_type == 'BUY':
                    individual['buy_trades'] += 1
                else:
                    individual['sell_trades'] += 1
                
                if profit > 0:
                    individual['win_rate'] = (individual['win_rate'] * (individual['trades'] - 1) + 1) / individual['trades']
                else:
                    individual['win_rate'] = (individual['win_rate'] * (individual['trades'] - 1)) / individual['trades']
                
                # Store in memory
                self.trade_memory.append({
                    'network_id': network_id,
                    'profit': profit,
                    'trade_type': trade_type,
                    'timestamp': datetime.now()
                })
                
                break
    
    def save_memory(self):
        """Save learning memory to disk"""
        memory = {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'trade_memory': list(self.trade_memory)[-1000:],  # Last 1000 trades
            'pattern_memory': self.pattern_memory
        }
        
        with open('neural_memory.json', 'w') as f:
            json.dump(memory, f, default=str)
    
    def load_memory(self):
        """Load learning memory from disk"""
        if os.path.exists('neural_memory.json'):
            try:
                with open('neural_memory.json', 'r') as f:
                    memory = json.load(f)
                    self.generation = memory.get('generation', 0)
                    self.best_fitness = memory.get('best_fitness', 0)
                    self.pattern_memory = memory.get('pattern_memory', {})
                    print(f"[NEURAL] Loaded memory from generation {self.generation}")
            except:
                pass

# ============================================================================
# ENHANCED SUPREME TRADER - MASTER CONTROLLER
# ============================================================================

class SupremeEnhancedTrader:
    """Enhanced ultimate trading system with perfect balance"""
    
    def __init__(self):
        self.connected = False
        
        # Initialize ALL systems
        self.balancer = PositionBalancer()
        self.neural_evolution = EnhancedNeuralEvolution()
        
        # Trading state
        self.performance = {
            'start_balance': 0,
            'peak_balance': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0,
            'buy_trades': 0,
            'sell_trades': 0,
            'daily_profit': 0,
            'daily_loss': 0,
            'session_start': datetime.now()
        }
        
        self.positions = {}
        self.symbol_performance = defaultdict(lambda: {'profit': 0, 'trades': 0})
        
        # Initialize MT5
        self.initialize()
    
    def initialize(self):
        """Initialize MT5 and all systems"""
        
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
            self.performance['start_balance'] = account.balance
            self.performance['peak_balance'] = account.balance
            
            print(f"\n{'='*80}")
            print(f"SUPREME ENHANCED TRADER - ULTRA PERFORMANCE EDITION")
            print(f"{'='*80}")
            print(f"Account Balance: ${account.balance:.2f}")
            print(f"Leverage: 1:{account.leverage}")
            print(f"\nENHANCED FEATURES:")
            print(f"+ Position Balancer (50/50 BUY/SELL)")
            print(f"+ {Config.POPULATION_SIZE} Neural Networks")
            print(f"+ Adaptive Position Sizing")
            print(f"+ Memory-based Learning")
            print(f"+ Daily P&L Limits")
            print(f"+ {len(Config.FOREX_SYMBOLS)} Currency Pairs")
            print(f"+ Trailing Stop Loss")
            print(f"+ Correlation Trading")
            print(f"{'='*80}\n")
    
    def calculate_dynamic_lot_size(self, symbol, confidence):
        """Calculate lot size based on confidence and performance"""
        
        base_lot = Config.BASE_LOT
        
        # Adjust based on confidence
        if confidence > 0.85:
            lot_multiplier = 2.5
        elif confidence > 0.75:
            lot_multiplier = 2.0
        elif confidence > 0.65:
            lot_multiplier = 1.5
        else:
            lot_multiplier = 1.0
        
        # Adjust based on symbol performance
        symbol_perf = self.symbol_performance[symbol]
        if symbol_perf['trades'] > 10:
            if symbol_perf['profit'] > 50:
                lot_multiplier *= 1.5
            elif symbol_perf['profit'] < -50:
                lot_multiplier *= 0.5
        
        # Adjust based on daily performance
        if self.performance['daily_profit'] > 100:
            lot_multiplier *= 1.2
        elif self.performance['daily_loss'] > 50:
            lot_multiplier *= 0.7
        
        # Calculate final lot
        lot_size = base_lot * lot_multiplier
        lot_size = min(lot_size, Config.MAX_LOT)
        
        # Ensure proper stepping
        info = mt5.symbol_info(symbol)
        if info:
            lot_size = round(lot_size / info.volume_step) * info.volume_step
            lot_size = max(info.volume_min, min(lot_size, info.volume_max))
        
        return lot_size
    
    def get_enhanced_signals(self, symbol):
        """Get signals from all enhanced systems"""
        
        all_signals = []
        
        # 1. Neural Network Prediction (with balance)
        if NEURAL_AVAILABLE:
            features = self.extract_enhanced_features(symbol)
            if features is not None:
                neural_signal = self.neural_evolution.predict_balanced(features, self.balancer)
                if neural_signal:
                    neural_signal['source'] = 'neural'
                    all_signals.append(neural_signal)
        
        # 2. Technical Analysis
        tech_signal = self.enhanced_technical_analysis(symbol)
        if tech_signal:
            tech_signal['source'] = 'technical'
            all_signals.append(tech_signal)
        
        # 3. Momentum Trading
        momentum_signal = self.momentum_analysis(symbol)
        if momentum_signal:
            momentum_signal['source'] = 'momentum'
            all_signals.append(momentum_signal)
        
        # 4. Support/Resistance
        sr_signal = self.support_resistance_analysis(symbol)
        if sr_signal:
            sr_signal['source'] = 'support_resistance'
            all_signals.append(sr_signal)
        
        # 5. Pattern Recognition
        pattern_signal = self.pattern_recognition(symbol)
        if pattern_signal:
            pattern_signal['source'] = 'pattern'
            all_signals.append(pattern_signal)
        
        return all_signals
    
    def enhanced_technical_analysis(self, symbol):
        """Enhanced technical analysis with multiple timeframes"""
        
        # Get multiple timeframes
        rates_m1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 100)
        rates_m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 50)
        rates_m15 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 30)
        
        if rates_m5 is None or len(rates_m5) < 50:
            return None
        
        df = pd.DataFrame(rates_m5)
        
        # Calculate indicators
        df['sma20'] = df['close'].rolling(20).mean()
        df['sma50'] = df['close'].rolling(50).mean()
        df['ema12'] = df['close'].ewm(span=12).mean()
        df['ema26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9).mean()
        
        # Current values
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Scoring system
        buy_score = 0
        sell_score = 0
        
        # Trend
        if current['close'] > current['sma20'] > current['sma50']:
            buy_score += 2
        elif current['close'] < current['sma20'] < current['sma50']:
            sell_score += 2
        
        # RSI
        if current['rsi'] < 30:
            buy_score += 1.5
        elif current['rsi'] > 70:
            sell_score += 1.5
        
        # MACD
        if current['macd'] > current['signal'] and prev['macd'] <= prev['signal']:
            buy_score += 2
        elif current['macd'] < current['signal'] and prev['macd'] >= prev['signal']:
            sell_score += 2
        
        # Decision with balance adjustment
        bias = self.balancer.get_bias_adjustment()
        buy_score *= (1 + bias)
        sell_score *= (1 - bias)
        
        if buy_score > sell_score and buy_score > 2:
            return {'type': 'BUY', 'confidence': min(buy_score / 6, 0.95)}
        elif sell_score > buy_score and sell_score > 2:
            return {'type': 'SELL', 'confidence': min(sell_score / 6, 0.95)}
        
        return None
    
    def momentum_analysis(self, symbol):
        """Momentum-based signals"""
        
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 20)
        if rates is None or len(rates) < 20:
            return None
        
        df = pd.DataFrame(rates)
        
        # Calculate momentum
        momentum = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
        volume_ratio = df['tick_volume'].iloc[-1] / df['tick_volume'].mean()
        
        # Strong momentum with volume
        if momentum > 0.001 and volume_ratio > 1.5:
            return {'type': 'BUY', 'confidence': min(0.7 + momentum * 100, 0.9)}
        elif momentum < -0.001 and volume_ratio > 1.5:
            return {'type': 'SELL', 'confidence': min(0.7 + abs(momentum) * 100, 0.9)}
        
        return None
    
    def support_resistance_analysis(self, symbol):
        """Support and resistance levels"""
        
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
        if rates is None or len(rates) < 100:
            return None
        
        df = pd.DataFrame(rates)
        
        # Find recent highs and lows
        recent_high = df['high'].rolling(20).max().iloc[-1]
        recent_low = df['low'].rolling(20).min().iloc[-1]
        current = df['close'].iloc[-1]
        
        # Distance to levels
        dist_to_resistance = (recent_high - current) / current
        dist_to_support = (current - recent_low) / current
        
        # Near support = BUY, Near resistance = SELL
        if dist_to_support < 0.0005 and dist_to_support > 0:
            return {'type': 'BUY', 'confidence': 0.75}
        elif dist_to_resistance < 0.0005 and dist_to_resistance > 0:
            return {'type': 'SELL', 'confidence': 0.75}
        
        return None
    
    def pattern_recognition(self, symbol):
        """Recognize chart patterns"""
        
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 30)
        if rates is None or len(rates) < 30:
            return None
        
        df = pd.DataFrame(rates)
        
        # Simple pattern: Double bottom/top
        lows = df['low'].rolling(5).min()
        highs = df['high'].rolling(5).max()
        
        # Check for double bottom (bullish)
        if len(lows) > 20:
            bottom1 = lows.iloc[-20:-15].min()
            bottom2 = lows.iloc[-5:].min()
            
            if abs(bottom1 - bottom2) / bottom1 < 0.0005:  # Similar lows
                if df['close'].iloc[-1] > df['close'].iloc[-15]:  # Price rising
                    return {'type': 'BUY', 'confidence': 0.8}
        
        # Check for double top (bearish)
        if len(highs) > 20:
            top1 = highs.iloc[-20:-15].max()
            top2 = highs.iloc[-5:].max()
            
            if abs(top1 - top2) / top1 < 0.0005:  # Similar highs
                if df['close'].iloc[-1] < df['close'].iloc[-15]:  # Price falling
                    return {'type': 'SELL', 'confidence': 0.8}
        
        return None
    
    def extract_enhanced_features(self, symbol):
        """Extract 30 features for neural network"""
        
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
        if rates is None or len(rates) < 100:
            return None
        
        df = pd.DataFrame(rates)
        features = []
        
        # Price features (10)
        features.extend([
            df['close'].pct_change().mean(),
            df['close'].pct_change().std(),
            (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5],
            (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20],
            (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1],
            df['close'].iloc[-1] / df['close'].rolling(20).mean().iloc[-1] - 1,
            df['close'].iloc[-1] / df['close'].rolling(50).mean().iloc[-1] - 1,
            (df['high'].max() - df['close'].iloc[-1]) / df['close'].iloc[-1],
            (df['close'].iloc[-1] - df['low'].min()) / df['close'].iloc[-1],
            df['close'].diff().iloc[-1] / df['close'].iloc[-2]
        ])
        
        # Volume features (5)
        features.extend([
            df['tick_volume'].iloc[-1] / df['tick_volume'].mean(),
            df['tick_volume'].std() / df['tick_volume'].mean(),
            df['tick_volume'].iloc[-1] / df['tick_volume'].iloc[-2] - 1,
            df['tick_volume'].rolling(5).mean().iloc[-1] / df['tick_volume'].rolling(20).mean().iloc[-1],
            (df['tick_volume'] * df['close'].pct_change()).sum()  # Volume-weighted momentum
        ])
        
        # Technical indicators (10)
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi.iloc[-1] / 100)
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        features.append((macd.iloc[-1] - signal.iloc[-1]) / df['close'].iloc[-1])
        
        # Bollinger Bands
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        upper_band = sma20 + 2 * std20
        lower_band = sma20 - 2 * std20
        features.append((df['close'].iloc[-1] - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1]))
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean()
        features.append(atr.iloc[-1] / df['close'].iloc[-1])
        
        # Stochastic
        low_min = df['low'].rolling(14).min()
        high_max = df['high'].rolling(14).max()
        stoch = 100 * ((df['close'] - low_min) / (high_max - low_min))
        features.append(stoch.iloc[-1] / 100)
        
        # More indicators
        features.extend([
            df['close'].rolling(10).mean().iloc[-1] / df['close'].rolling(30).mean().iloc[-1] - 1,
            df['close'].rolling(5).std().iloc[-1] / df['close'].rolling(20).std().iloc[-1],
            (df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1],
            (df['close'].iloc[-1] - df['low'].iloc[-1]) / (df['high'].iloc[-1] - df['low'].iloc[-1]) if df['high'].iloc[-1] != df['low'].iloc[-1] else 0.5,
            df['close'].diff().rolling(5).mean().iloc[-1] / df['close'].iloc[-1]
        ])
        
        # Market microstructure (5)
        bid_ask_spread = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]
        features.extend([
            bid_ask_spread,
            df['tick_volume'].iloc[-5:].std() / df['tick_volume'].iloc[-5:].mean() if df['tick_volume'].iloc[-5:].mean() > 0 else 0,
            1 if datetime.now().hour in [8, 9, 14, 15] else 0,  # Trading session
            1 if datetime.now().weekday() in [1, 2, 3] else 0,  # Mid-week
            self.balancer.get_bias_adjustment()  # Current position balance
        ])
        
        # Ensure exactly 30 features
        while len(features) < 30:
            features.append(0)
        
        return np.array(features[:30])
    
    def execute_enhanced_trade(self, symbol, signals):
        """Execute trade with enhanced logic"""
        
        if not signals:
            return False
        
        # Check daily limits
        if self.performance['daily_profit'] >= Config.MAX_DAILY_PROFIT:
            print("[LIMIT] Daily profit target reached")
            return False
        
        if self.performance['daily_loss'] >= Config.MAX_DAILY_LOSS:
            print("[LIMIT] Daily loss limit reached")
            return False
        
        # Combine signals
        buy_signals = [s for s in signals if s.get('type') == 'BUY']
        sell_signals = [s for s in signals if s.get('type') == 'SELL']
        
        if not buy_signals and not sell_signals:
            return False
        
        # Determine direction
        if len(buy_signals) > len(sell_signals):
            direction = 'BUY'
            confidence = np.mean([s.get('confidence', 0.5) for s in buy_signals])
        else:
            direction = 'SELL'
            confidence = np.mean([s.get('confidence', 0.5) for s in sell_signals])
        
        # Check position balance
        if not self.balancer.should_open_position(direction):
            return False
        
        # Minimum confidence
        if confidence < 0.6:
            return False
        
        # Get symbol info
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        
        if not info or not tick:
            return False
        
        # Dynamic lot sizing
        volume = self.calculate_dynamic_lot_size(symbol, confidence)
        
        # Setup trade
        if direction == 'BUY':
            price = tick.ask
            sl = price - 10 * info.point * 10  # 10 pips
            tp = price + 15 * info.point * 10  # 15 pips
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl = price + 10 * info.point * 10
            tp = price - 15 * info.point * 10
            order_type = mt5.ORDER_TYPE_SELL
        
        # Prepare request
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
            'comment': f"enhanced_{signals[0].get('source', 'multi')}",
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK
        }
        
        # Execute
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.performance['total_trades'] += 1
            
            if direction == 'BUY':
                self.performance['buy_trades'] += 1
            else:
                self.performance['sell_trades'] += 1
            
            sources = list(set(s.get('source', 'unknown') for s in signals))
            
            # Update balancer
            self.balancer.update_counts()
            
            print(f"[TRADE] {direction} {symbol} x{volume:.2f} | "
                  f"Conf: {confidence:.0%} | "
                  f"Sources: {', '.join(sources[:3])} | "
                  f"Balance: B{self.balancer.position_counts['BUY']}/S{self.balancer.position_counts['SELL']}")
            
            return True
        
        return False
    
    def manage_positions_enhanced(self):
        """Enhanced position management with trailing stops"""
        
        positions = mt5.positions_get(magic=Config.MAGIC)
        if not positions:
            return
        
        for pos in positions:
            # Calculate profit in pips
            info = mt5.symbol_info(pos.symbol)
            if not info:
                continue
            
            pip_value = info.point * 10
            pips_profit = pos.profit / (pos.volume * info.trade_tick_value * pip_value) if info.trade_tick_value > 0 else 0
            
            # Trailing stop
            if pips_profit > Config.TRAILING_STOP_PIPS:
                tick = mt5.symbol_info_tick(pos.symbol)
                if tick:
                    if pos.type == 0:  # BUY
                        new_sl = tick.bid - Config.TRAILING_STOP_PIPS * pip_value
                        if new_sl > pos.sl:
                            request = {
                                'action': mt5.TRADE_ACTION_SLTP,
                                'symbol': pos.symbol,
                                'position': pos.ticket,
                                'sl': round(new_sl, info.digits),
                                'tp': pos.tp,
                                'magic': Config.MAGIC
                            }
                            mt5.order_send(request)
                    else:  # SELL
                        new_sl = tick.ask + Config.TRAILING_STOP_PIPS * pip_value
                        if new_sl < pos.sl:
                            request = {
                                'action': mt5.TRADE_ACTION_SLTP,
                                'symbol': pos.symbol,
                                'position': pos.ticket,
                                'sl': round(new_sl, info.digits),
                                'tp': pos.tp,
                                'magic': Config.MAGIC
                            }
                            mt5.order_send(request)
            
            # Update performance
            self.symbol_performance[pos.symbol]['profit'] = pos.profit
            
            # Quick profit taking
            if pos.profit > 3:  # $3 profit
                self.close_position(pos, "quick_profit")
                self.performance['winning_trades'] += 1
                
                # Update neural network
                if NEURAL_AVAILABLE:
                    self.neural_evolution.update_fitness(0, pos.profit, 'BUY' if pos.type == 0 else 'SELL')
            
            # Stop loss
            elif pos.profit < -5:  # $5 loss
                self.close_position(pos, "stop_loss")
                
                # Update neural network
                if NEURAL_AVAILABLE:
                    self.neural_evolution.update_fitness(0, pos.profit, 'BUY' if pos.type == 0 else 'SELL')
    
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
            'deviation': 20,
            'magic': Config.MAGIC,
            'comment': reason,
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            # Update performance
            self.performance['total_profit'] += position.profit
            
            if position.profit > 0:
                self.performance['daily_profit'] += position.profit
            else:
                self.performance['daily_loss'] += abs(position.profit)
            
            # Update symbol performance
            self.symbol_performance[position.symbol]['trades'] += 1
            self.symbol_performance[position.symbol]['profit'] += position.profit
    
    def update_daily_stats(self):
        """Reset daily stats at midnight"""
        
        if datetime.now().hour == 0 and datetime.now().minute == 0:
            self.performance['daily_profit'] = 0
            self.performance['daily_loss'] = 0
    
    def display_enhanced_status(self):
        """Display enhanced status with balance info"""
        
        account = mt5.account_info()
        if not account:
            return
        
        positions = mt5.positions_get(magic=Config.MAGIC)
        self.balancer.update_counts()
        
        profit = account.balance - self.performance['start_balance']
        roi = (profit / self.performance['start_balance'] * 100) if self.performance['start_balance'] > 0 else 0
        
        # Calculate win rate
        win_rate = 0
        if self.performance['total_trades'] > 0:
            win_rate = (self.performance['winning_trades'] / self.performance['total_trades']) * 100
        
        # Position balance
        total_positions = self.balancer.position_counts['BUY'] + self.balancer.position_counts['SELL']
        buy_pct = (self.balancer.position_counts['BUY'] / total_positions * 100) if total_positions > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"SUPREME ENHANCED STATUS - Generation {self.neural_evolution.generation}")
        print(f"{'='*80}")
        print(f"Balance: ${account.balance:.2f} | Equity: ${account.equity:.2f}")
        print(f"P&L: ${profit:+.2f} ({roi:+.1f}%) | Peak: ${self.performance['peak_balance']:.2f}")
        print(f"Today: +${self.performance['daily_profit']:.2f} / -${self.performance['daily_loss']:.2f}")
        print(f"\nPositions: {total_positions}/{Config.MAX_POSITIONS}")
        print(f"BUY: {self.balancer.position_counts['BUY']} ({buy_pct:.0f}%)")
        print(f"SELL: {self.balancer.position_counts['SELL']} ({100-buy_pct:.0f}%)")
        print(f"\nTrades: {self.performance['total_trades']} | Win Rate: {win_rate:.1f}%")
        
        if NEURAL_AVAILABLE:
            print(f"Neural Fitness: {self.neural_evolution.best_fitness:.2f}")
        
        # Top performing symbols
        top_symbols = sorted(self.symbol_performance.items(), key=lambda x: x[1]['profit'], reverse=True)[:3]
        if top_symbols:
            print(f"\nTop Symbols:")
            for symbol, perf in top_symbols:
                if perf['trades'] > 0:
                    print(f"  {symbol}: ${perf['profit']:.2f} ({perf['trades']} trades)")
        
        print(f"{'='*80}")
    
    def run(self):
        """Main enhanced trading loop"""
        
        if not self.connected:
            return
        
        print("Supreme Enhanced Trader Active...\n")
        
        cycle = 0
        last_evolution = time.time()
        last_rebalance = time.time()
        last_status = time.time()
        
        try:
            while True:
                cycle += 1
                
                # Update daily stats
                self.update_daily_stats()
                
                # Check account
                account = mt5.account_info()
                if not account:
                    continue
                
                # Update peak balance
                if account.balance > self.performance['peak_balance']:
                    self.performance['peak_balance'] = account.balance
                
                # Get current positions
                positions = mt5.positions_get(magic=Config.MAGIC)
                current_positions = len(positions) if positions else 0
                
                # Find and execute trades
                if current_positions < Config.MAX_POSITIONS:
                    for symbol in Config.FOREX_SYMBOLS:
                        if current_positions >= Config.MAX_POSITIONS:
                            break
                        
                        # Check if symbol is available
                        if not mt5.symbol_info(symbol):
                            continue
                        
                        # Get signals
                        signals = self.get_enhanced_signals(symbol)
                        
                        if signals:
                            if self.execute_enhanced_trade(symbol, signals):
                                current_positions += 1
                
                # Manage positions
                self.manage_positions_enhanced()
                
                # Rebalance positions
                if time.time() - last_rebalance > Config.REBALANCE_INTERVAL:
                    self.balancer.force_rebalance()
                    last_rebalance = time.time()
                
                # Evolve neural networks
                if time.time() - last_evolution > 300 and NEURAL_AVAILABLE:  # Every 5 minutes
                    self.neural_evolution.evolve_with_balance()
                    last_evolution = time.time()
                
                # Display status
                if time.time() - last_status > 10:  # Every 10 seconds
                    self.display_enhanced_status()
                    last_status = time.time()
                
                # Sleep
                time.sleep(Config.SCAN_INTERVAL_MS / 1000)
                
        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Closing all positions...")
            self.close_all_positions()
            self.final_report()
        finally:
            mt5.shutdown()
    
    def close_all_positions(self):
        """Close all open positions"""
        
        positions = mt5.positions_get(magic=Config.MAGIC)
        if positions:
            for pos in positions:
                self.close_position(pos, "shutdown")
    
    def final_report(self):
        """Generate final performance report"""
        
        print(f"\n{'='*80}")
        print(f"SUPREME ENHANCED - FINAL REPORT")
        print(f"{'='*80}")
        
        account = mt5.account_info()
        if account:
            profit = account.balance - self.performance['start_balance']
            roi = (profit / self.performance['start_balance'] * 100) if self.performance['start_balance'] > 0 else 0
            
            print(f"Starting Balance: ${self.performance['start_balance']:.2f}")
            print(f"Final Balance: ${account.balance:.2f}")
            print(f"Total Profit: ${profit:.2f}")
            print(f"ROI: {roi:.1f}%")
            print(f"Peak Balance: ${self.performance['peak_balance']:.2f}")
            
            runtime = datetime.now() - self.performance['session_start']
            print(f"\nRuntime: {runtime}")
            print(f"Total Trades: {self.performance['total_trades']}")
            print(f"BUY Trades: {self.performance['buy_trades']}")
            print(f"SELL Trades: {self.performance['sell_trades']}")
            
            if self.performance['total_trades'] > 0:
                win_rate = (self.performance['winning_trades'] / self.performance['total_trades']) * 100
                print(f"Win Rate: {win_rate:.1f}%")
                
                buy_pct = (self.performance['buy_trades'] / self.performance['total_trades']) * 100
                print(f"BUY/SELL Ratio: {buy_pct:.0f}% / {100-buy_pct:.0f}%")
        
        if NEURAL_AVAILABLE:
            print(f"\nNeural Evolution:")
            print(f"Final Generation: {self.neural_evolution.generation}")
            print(f"Best Fitness: {self.neural_evolution.best_fitness:.2f}")
            self.neural_evolution.save_memory()
        
        print(f"{'='*80}")

# ============================================================================
# LAUNCH ENHANCED SYSTEM
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("INITIALIZING SUPREME ENHANCED TRADER...")
    print("="*80)
    
    trader = SupremeEnhancedTrader()
    
    if trader.connected:
        trader.run()
    else:
        print("[ERROR] Failed to initialize Supreme Enhanced Trader")