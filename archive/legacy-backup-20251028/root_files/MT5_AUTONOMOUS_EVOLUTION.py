"""
MT5 AUTONOMOUS EVOLUTION SYSTEM
================================
The Ultimate Self-Evolving, Fully Autonomous Trading System

Features:
1. FIXED: Balanced BUY/SELL trading (no more directional bias)
2. Fully autonomous operation with smart start/stop
3. Continuous self-evolution and improvement
4. Real-time learning and adaptation
5. Multi-dimensional strategy evolution
"""

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import json
import os
import time
import random
from datetime import datetime, timedelta
import hashlib
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try imports for advanced features
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False

print("="*70)
print("MT5 AUTONOMOUS EVOLUTION SYSTEM")
print("Self-Evolving | Self-Improving | Fully Autonomous")
print("="*70)

# =====================================
# EVOLUTION CONFIGURATION
# =====================================
class EvolutionConfig:
    # Core Settings
    SCAN_INTERVAL_MS = 100  # Ultra-fast scanning
    EVOLUTION_INTERVAL = 300  # Evolve every 5 minutes
    
    # Position Management
    MAX_POSITIONS = 20
    MIN_CONFIDENCE = 0.55  # Lower threshold for more opportunities
    
    # Risk Management  
    RISK_PER_TRADE = 0.01  # 1% risk per trade
    MAX_DAILY_LOSS = 0.05  # 5% max daily loss
    
    # Evolution Parameters
    POPULATION_SIZE = 100  # 100 strategies competing
    MUTATION_RATE = 0.2
    CROSSOVER_RATE = 0.7
    ELITE_SIZE = 10
    
    # Autonomous Control
    AUTO_START_HOUR = 0  # Start at midnight
    AUTO_STOP_HOUR = 24  # Trade 24/7
    MIN_SPREAD_PIPS = 0.5
    MAX_SPREAD_PIPS = 3.0
    
    # Learning Parameters
    MEMORY_SIZE = 10000  # Remember last 10k trades
    LEARNING_RATE = 0.01
    ADAPTATION_SPEED = 0.1

# =====================================
# AUTONOMOUS BRAIN
# =====================================
class AutonomousBrain:
    def __init__(self):
        self.evolution_state = self.load_evolution_state()
        self.trade_memory = []
        self.performance_history = []
        self.active_strategies = self.initialize_strategies()
        self.market_regime = "UNKNOWN"
        self.trading_enabled = True
        self.evolution_generation = 0
        self.buy_sell_ratio = {"buy": 0, "sell": 0}
        
    def initialize_strategies(self):
        """Initialize diverse population of strategies"""
        strategies = []
        for i in range(EvolutionConfig.POPULATION_SIZE):
            strategy = {
                'id': i,
                'genes': {
                    'rsi_period': random.randint(5, 30),
                    'rsi_oversold': random.uniform(20, 40),
                    'rsi_overbought': random.uniform(60, 80),
                    'ma_fast': random.randint(5, 20),
                    'ma_slow': random.randint(20, 100),
                    'atr_multiplier': random.uniform(1.0, 3.0),
                    'volume_threshold': random.uniform(0.5, 2.0),
                    'trend_strength': random.uniform(0.3, 0.8),
                    'pattern_weight': random.uniform(0.1, 0.5),
                    'sentiment_weight': random.uniform(0.1, 0.4),
                    'buy_bias': random.uniform(0.3, 0.7),  # BALANCED BIAS
                    'sell_bias': random.uniform(0.3, 0.7),  # BALANCED BIAS
                    'time_preference': random.randint(0, 23),
                    'volatility_preference': random.uniform(0.5, 2.0),
                    'correlation_threshold': random.uniform(0.3, 0.8),
                    'ml_weight': random.uniform(0.1, 0.5) if ML_AVAILABLE else 0
                },
                'fitness': 0,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'total_profit': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'last_evolved': datetime.now()
            }
            strategies.append(strategy)
        return strategies
    
    def should_trade_now(self):
        """Autonomous decision on whether to trade"""
        current_hour = datetime.now().hour
        
        # Check time window
        if EvolutionConfig.AUTO_START_HOUR <= current_hour < EvolutionConfig.AUTO_STOP_HOUR:
            
            # Check market conditions
            account = mt5.account_info()
            if account:
                # Check daily loss limit
                daily_loss = self.calculate_daily_loss()
                if daily_loss > EvolutionConfig.MAX_DAILY_LOSS:
                    print("[AUTONOMOUS] Daily loss limit reached - PAUSING")
                    return False
                
                # Check account health
                if account.margin_level and account.margin_level < 200:
                    print("[AUTONOMOUS] Low margin level - PAUSING")
                    return False
                
                # Check spread conditions
                symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
                high_spread_count = 0
                
                for symbol in symbols:
                    tick = mt5.symbol_info_tick(symbol)
                    if tick:
                        spread = (tick.ask - tick.bid) / mt5.symbol_info(symbol).point
                        if spread > EvolutionConfig.MAX_SPREAD_PIPS * 10:
                            high_spread_count += 1
                
                if high_spread_count >= 2:
                    print("[AUTONOMOUS] High spreads detected - WAITING")
                    return False
                
                return True
        
        return False
    
    def evolve_strategies(self):
        """Evolve strategies based on performance"""
        print("\n[EVOLUTION] Generation", self.evolution_generation)
        
        # Sort by fitness
        self.active_strategies.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Keep elite
        elite = self.active_strategies[:EvolutionConfig.ELITE_SIZE]
        
        # Create new generation
        new_generation = elite.copy()
        
        while len(new_generation) < EvolutionConfig.POPULATION_SIZE:
            # Selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover
            if random.random() < EvolutionConfig.CROSSOVER_RATE:
                child = self.crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            if random.random() < EvolutionConfig.MUTATION_RATE:
                child = self.mutate(child)
            
            # Reset child stats
            child['trades'] = 0
            child['wins'] = 0
            child['losses'] = 0
            child['total_profit'] = 0
            child['fitness'] = 0
            
            new_generation.append(child)
        
        self.active_strategies = new_generation
        self.evolution_generation += 1
        
        # Save evolution state
        self.save_evolution_state()
        
        print(f"[EVOLUTION] Best fitness: {elite[0]['fitness']:.2f}")
        print(f"[EVOLUTION] Best win rate: {elite[0]['win_rate']:.2%}")
        print(f"[EVOLUTION] Buy/Sell ratio: {self.buy_sell_ratio['buy']}/{self.buy_sell_ratio['sell']}")
    
    def tournament_selection(self, tournament_size=5):
        """Select strategy using tournament selection"""
        tournament = random.sample(self.active_strategies, tournament_size)
        return max(tournament, key=lambda x: x['fitness'])
    
    def crossover(self, parent1, parent2):
        """Create child from two parents"""
        child = parent1.copy()
        child['genes'] = {}
        
        for gene in parent1['genes']:
            if random.random() < 0.5:
                child['genes'][gene] = parent1['genes'][gene]
            else:
                child['genes'][gene] = parent2['genes'][gene]
        
        return child
    
    def mutate(self, strategy):
        """Mutate strategy genes"""
        mutated = strategy.copy()
        
        for gene in mutated['genes']:
            if random.random() < 0.1:  # 10% chance per gene
                if isinstance(mutated['genes'][gene], int):
                    mutated['genes'][gene] = int(mutated['genes'][gene] * random.uniform(0.8, 1.2))
                else:
                    mutated['genes'][gene] = mutated['genes'][gene] * random.uniform(0.8, 1.2)
        
        # Ensure balanced buy/sell bias
        mutated['genes']['buy_bias'] = max(0.3, min(0.7, mutated['genes']['buy_bias']))
        mutated['genes']['sell_bias'] = max(0.3, min(0.7, mutated['genes']['sell_bias']))
        
        return mutated
    
    def calculate_fitness(self, strategy):
        """Calculate strategy fitness"""
        if strategy['trades'] == 0:
            return 0
        
        # Multi-objective fitness
        win_rate = strategy['wins'] / max(1, strategy['trades'])
        avg_profit = strategy['total_profit'] / max(1, strategy['trades'])
        
        # Sharpe ratio component
        if len(self.performance_history) > 10:
            returns = [p['profit'] for p in self.performance_history[-10:]]
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns)
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        # Balance component (reward balanced trading)
        buy_ratio = self.buy_sell_ratio['buy'] / max(1, self.buy_sell_ratio['buy'] + self.buy_sell_ratio['sell'])
        balance_score = 1.0 - abs(0.5 - buy_ratio) * 2  # Max score at 50/50
        
        # Combine components
        fitness = (
            win_rate * 40 +
            min(avg_profit * 10, 30) +  # Cap profit component
            sharpe * 10 +
            balance_score * 20 +  # Balance is important
            (1 - strategy.get('max_drawdown', 0)) * 10
        )
        
        return fitness
    
    def select_strategy(self):
        """Select best strategy for current conditions"""
        # Update fitness for all strategies
        for strategy in self.active_strategies:
            strategy['fitness'] = self.calculate_fitness(strategy)
        
        # Sort by fitness
        self.active_strategies.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Use top 10 strategies with weighted random selection
        top_strategies = self.active_strategies[:10]
        weights = [s['fitness'] for s in top_strategies]
        
        if sum(weights) > 0:
            return random.choices(top_strategies, weights=weights)[0]
        else:
            return random.choice(self.active_strategies)
    
    def analyze_with_strategy(self, symbol, strategy):
        """Analyze market with specific strategy genes"""
        genes = strategy['genes']
        
        # Get market data
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 100)
        if rates is None or len(rates) < 50:
            return None
        
        df = pd.DataFrame(rates)
        
        # Calculate indicators based on genes
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'], genes['rsi_period'])
        
        # Moving averages
        df['ma_fast'] = df['close'].rolling(genes['ma_fast']).mean()
        df['ma_slow'] = df['close'].rolling(genes['ma_slow']).mean()
        
        # ATR
        df['atr'] = self.calculate_atr(df, 14) * genes['atr_multiplier']
        
        # Volume
        df['volume_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
        
        current = df.iloc[-1]
        
        # BALANCED SIGNAL GENERATION
        buy_signals = 0
        sell_signals = 0
        
        # RSI signals
        if current['rsi'] < genes['rsi_oversold']:
            buy_signals += 1
        if current['rsi'] > genes['rsi_overbought']:
            sell_signals += 1
        
        # MA crossover
        if current['ma_fast'] > current['ma_slow']:
            buy_signals += genes['trend_strength']
        else:
            sell_signals += genes['trend_strength']
        
        # Volume confirmation
        if current['volume_ratio'] > genes['volume_threshold']:
            if current['close'] > df['close'].iloc[-2]:
                buy_signals += genes['pattern_weight']
            else:
                sell_signals += genes['pattern_weight']
        
        # Time preference
        if datetime.now().hour == genes['time_preference']:
            buy_signals += 0.1
            sell_signals += 0.1
        
        # Apply bias (BALANCED)
        buy_score = buy_signals * genes['buy_bias']
        sell_score = sell_signals * genes['sell_bias']
        
        # Determine signal
        if buy_score > sell_score and buy_score > 1.0:
            signal = 'BUY'
            confidence = min(buy_score / 3.0, 0.95)
        elif sell_score > buy_score and sell_score > 1.0:
            signal = 'SELL'
            confidence = min(sell_score / 3.0, 0.95)
        else:
            signal = 'NEUTRAL'
            confidence = 0
        
        if signal != 'NEUTRAL':
            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'strategy_id': strategy['id'],
                'entry': current['close'],
                'sl': current['close'] - current['atr'] if signal == 'BUY' else current['close'] + current['atr'],
                'tp': current['close'] + current['atr'] * 2 if signal == 'BUY' else current['close'] - current['atr'] * 2
            }
        
        return None
    
    def record_trade_result(self, strategy_id, profit, is_win):
        """Record trade result and update strategy"""
        for strategy in self.active_strategies:
            if strategy['id'] == strategy_id:
                strategy['trades'] += 1
                strategy['total_profit'] += profit
                
                if is_win:
                    strategy['wins'] += 1
                else:
                    strategy['losses'] += 1
                
                strategy['win_rate'] = strategy['wins'] / max(1, strategy['trades'])
                strategy['avg_profit'] = strategy['total_profit'] / max(1, strategy['trades'])
                
                # Update fitness
                strategy['fitness'] = self.calculate_fitness(strategy)
                break
        
        # Save to memory
        self.trade_memory.append({
            'timestamp': datetime.now(),
            'strategy_id': strategy_id,
            'profit': profit,
            'is_win': is_win
        })
        
        # Trim memory
        if len(self.trade_memory) > EvolutionConfig.MEMORY_SIZE:
            self.trade_memory = self.trade_memory[-EvolutionConfig.MEMORY_SIZE:]
    
    def calculate_daily_loss(self):
        """Calculate today's loss percentage"""
        today_trades = [t for t in self.trade_memory 
                       if t['timestamp'].date() == datetime.now().date()]
        
        if today_trades:
            total_profit = sum(t['profit'] for t in today_trades)
            account = mt5.account_info()
            if account and account.balance > 0:
                return -total_profit / account.balance if total_profit < 0 else 0
        
        return 0
    
    def calculate_rsi(self, prices, period):
        """Calculate RSI"""
        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, df, period=14):
        """Calculate ATR"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        return true_range.rolling(period).mean()
    
    def save_evolution_state(self):
        """Save evolution state to disk"""
        state = {
            'generation': self.evolution_generation,
            'strategies': self.active_strategies,
            'buy_sell_ratio': self.buy_sell_ratio,
            'trade_memory': self.trade_memory[-1000:],  # Save last 1000 trades
            'timestamp': datetime.now().isoformat()
        }
        
        with open('evolution_state.json', 'w') as f:
            json.dump(state, f, default=str)
    
    def load_evolution_state(self):
        """Load evolution state from disk"""
        if os.path.exists('evolution_state.json'):
            try:
                with open('evolution_state.json', 'r') as f:
                    state = json.load(f)
                    
                self.evolution_generation = state.get('generation', 0)
                self.buy_sell_ratio = state.get('buy_sell_ratio', {"buy": 0, "sell": 0})
                
                # Convert trade memory timestamps
                trade_memory = state.get('trade_memory', [])
                for trade in trade_memory:
                    if isinstance(trade['timestamp'], str):
                        trade['timestamp'] = datetime.fromisoformat(trade['timestamp'])
                
                self.trade_memory = trade_memory
                
                print(f"[EVOLUTION] Loaded generation {self.evolution_generation}")
                print(f"[EVOLUTION] Historical ratio: BUY {self.buy_sell_ratio['buy']} / SELL {self.buy_sell_ratio['sell']}")
                
                return state
            except Exception as e:
                print(f"[EVOLUTION] Could not load state: {e}")
        
        return None

# =====================================
# TRADE EXECUTOR
# =====================================
class TradeExecutor:
    def __init__(self):
        self.magic = 777777  # Lucky sevens for evolution
        self.positions = {}
        
    def execute_trade(self, signal, strategy_id):
        """Execute trade with balanced buy/sell"""
        symbol = signal['symbol']
        
        # Check if we already have a position
        positions = mt5.positions_get(symbol=symbol)
        if positions and len(positions) >= 2:
            return False
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info or not symbol_info.visible:
            mt5.symbol_select(symbol, True)
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return False
        
        # Calculate position size
        account = mt5.account_info()
        if not account:
            return False
        
        risk_amount = account.balance * EvolutionConfig.RISK_PER_TRADE
        
        # Calculate stop loss distance
        sl_distance = abs(signal['entry'] - signal['sl'])
        if sl_distance == 0:
            sl_distance = 10 * symbol_info.point
        
        # Calculate lot size
        tick_value = symbol_info.trade_tick_value
        if tick_value == 0:
            tick_value = 1
        
        lot_size = risk_amount / (sl_distance / symbol_info.point * tick_value)
        lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
        lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step
        
        # Prepare request
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': lot_size,
            'type': mt5.ORDER_TYPE_BUY if signal['signal'] == 'BUY' else mt5.ORDER_TYPE_SELL,
            'price': signal['entry'],
            'sl': round(signal['sl'], symbol_info.digits),
            'tp': round(signal['tp'], symbol_info.digits),
            'deviation': 20,
            'magic': self.magic,
            'comment': f"EVO_{strategy_id}_{signal['signal']}",
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[TRADE] {signal['signal']} {symbol} - Strategy {strategy_id} - Confidence {signal['confidence']:.2%}")
            
            # Track position
            self.positions[result.order] = {
                'symbol': symbol,
                'signal': signal['signal'],
                'strategy_id': strategy_id,
                'entry': signal['entry'],
                'sl': signal['sl'],
                'tp': signal['tp'],
                'time': datetime.now()
            }
            
            return True
        
        return False
    
    def monitor_positions(self, brain):
        """Monitor and update positions"""
        positions = mt5.positions_get(magic=self.magic)
        
        if positions:
            for pos in positions:
                # Update strategy performance
                if pos.comment and 'EVO_' in pos.comment:
                    try:
                        strategy_id = int(pos.comment.split('_')[1])
                        
                        # Record result if position closed
                        if pos.ticket not in self.positions:
                            is_win = pos.profit > 0
                            brain.record_trade_result(strategy_id, pos.profit, is_win)
                            
                            # Update buy/sell ratio
                            if 'BUY' in pos.comment:
                                brain.buy_sell_ratio['buy'] += 1
                            else:
                                brain.buy_sell_ratio['sell'] += 1
                    except:
                        pass
                
                # Adaptive stop loss
                if pos.profit > 10:  # If profit > $10, tighten stop
                    current_price = mt5.symbol_info_tick(pos.symbol).bid if pos.type == 0 else mt5.symbol_info_tick(pos.symbol).ask
                    new_sl = current_price - 5 * mt5.symbol_info(pos.symbol).point if pos.type == 0 else current_price + 5 * mt5.symbol_info(pos.symbol).point
                    
                    if (pos.type == 0 and new_sl > pos.sl) or (pos.type == 1 and new_sl < pos.sl):
                        request = {
                            'action': mt5.TRADE_ACTION_SLTP,
                            'symbol': pos.symbol,
                            'position': pos.ticket,
                            'sl': round(new_sl, mt5.symbol_info(pos.symbol).digits),
                            'tp': pos.tp,
                            'magic': self.magic
                        }
                        mt5.order_send(request)

# =====================================
# MAIN EVOLUTION SYSTEM
# =====================================
class EvolutionSystem:
    def __init__(self):
        self.brain = AutonomousBrain()
        self.executor = TradeExecutor()
        self.last_evolution = datetime.now()
        self.symbols = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
            'NZDUSD', 'USDCHF', 'EURJPY', 'GBPJPY', 'EURGBP'
        ]
        
    def run(self):
        """Main evolution loop"""
        print("\n[SYSTEM] Starting Autonomous Evolution System...")
        print("[SYSTEM] Generation:", self.brain.evolution_generation)
        print("[SYSTEM] Strategies:", len(self.brain.active_strategies))
        
        iteration = 0
        
        while True:
            try:
                # Check if we should trade
                if not self.brain.should_trade_now():
                    print("\r[AUTONOMOUS] Trading paused - Waiting for better conditions...", end='')
                    time.sleep(5)
                    continue
                
                # Evolution check
                if (datetime.now() - self.last_evolution).seconds > EvolutionConfig.EVOLUTION_INTERVAL:
                    self.brain.evolve_strategies()
                    self.last_evolution = datetime.now()
                
                # Monitor existing positions
                self.executor.monitor_positions(self.brain)
                
                # Select strategy for this iteration
                strategy = self.brain.select_strategy()
                
                # Scan all symbols
                for symbol in self.symbols:
                    # Check position limit
                    positions = mt5.positions_get()
                    if positions and len(positions) >= EvolutionConfig.MAX_POSITIONS:
                        break
                    
                    # Analyze with selected strategy
                    signal = self.brain.analyze_with_strategy(symbol, strategy)
                    
                    if signal and signal['confidence'] > EvolutionConfig.MIN_CONFIDENCE:
                        # Execute trade
                        if self.executor.execute_trade(signal, strategy['id']):
                            strategy['trades'] += 1
                
                # Display status
                if iteration % 10 == 0:
                    self.display_status()
                
                iteration += 1
                time.sleep(EvolutionConfig.SCAN_INTERVAL_MS / 1000)
                
            except KeyboardInterrupt:
                print("\n[SYSTEM] Shutting down...")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
                time.sleep(1)
    
    def display_status(self):
        """Display system status"""
        account = mt5.account_info()
        positions = mt5.positions_get(magic=self.executor.magic)
        
        if account:
            # Calculate statistics
            buy_count = sum(1 for p in positions if p.type == 0) if positions else 0
            sell_count = sum(1 for p in positions if p.type == 1) if positions else 0
            
            # Get top strategy
            top_strategy = self.brain.active_strategies[0] if self.brain.active_strategies else None
            
            print(f"\n{'='*50}")
            print(f"AUTONOMOUS EVOLUTION STATUS")
            print(f"{'='*50}")
            print(f"Generation: {self.brain.evolution_generation}")
            print(f"Balance: ${account.balance:.2f}")
            print(f"Equity: ${account.equity:.2f}")
            print(f"Positions: {len(positions) if positions else 0} (BUY: {buy_count}, SELL: {sell_count})")
            
            if self.brain.buy_sell_ratio['buy'] + self.brain.buy_sell_ratio['sell'] > 0:
                buy_pct = self.brain.buy_sell_ratio['buy'] / (self.brain.buy_sell_ratio['buy'] + self.brain.buy_sell_ratio['sell']) * 100
                print(f"Historical: BUY {buy_pct:.1f}% / SELL {100-buy_pct:.1f}%")
            
            if top_strategy:
                print(f"\nTop Strategy #{top_strategy['id']}:")
                print(f"  Fitness: {top_strategy['fitness']:.2f}")
                print(f"  Win Rate: {top_strategy['win_rate']:.2%}")
                print(f"  Trades: {top_strategy['trades']}")
                print(f"  Profit: ${top_strategy['total_profit']:.2f}")
            
            print(f"{'='*50}")

# =====================================
# MAIN EXECUTION
# =====================================
def main():
    # Initialize MT5
    if not mt5.initialize():
        print("[ERROR] Failed to initialize MT5")
        return
    
    # Login
    if not mt5.login(95948709, 'To-4KyLg', 'MetaQuotes-Demo'):
        print("[ERROR] Failed to login")
        mt5.shutdown()
        return
    
    print("[SUCCESS] Connected to MT5")
    
    # Show account info
    account = mt5.account_info()
    if account:
        print(f"[ACCOUNT] Balance: ${account.balance:.2f}")
        print(f"[ACCOUNT] Leverage: 1:{account.leverage}")
    
    # Start evolution system
    system = EvolutionSystem()
    
    try:
        system.run()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Evolution stopped by user")
    finally:
        # Save final state
        system.brain.save_evolution_state()
        mt5.shutdown()
        print("[SYSTEM] MT5 connection closed")

if __name__ == "__main__":
    main()