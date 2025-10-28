"""
MT5 MASTER AI - The Ultimate Trading System with Everything
============================================================
Dashboard + Deep Learning + Position Optimizer + Market Maker + Arbitrage
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
import warnings
warnings.filterwarnings('ignore')

# For Dashboard
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("[INFO] Rich not installed. Dashboard will use simple display")

# For Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    print("[INFO] TensorFlow not installed. Deep Learning disabled")

try:
    from sklearn.preprocessing import MinMaxScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[INFO] scikit-learn not installed. Some ML features disabled")

# ============================================================================
# MASTER CONFIGURATION
# ============================================================================

class Config:
    # Account
    LOGIN = 95948709
    PASSWORD = 'To-4KyLg'
    SERVER = 'MetaQuotes-Demo'
    MAGIC = 333333
    
    # Speed Settings
    SCAN_INTERVAL_MS = 50
    DASHBOARD_UPDATE_MS = 1000
    
    # Symbols (Extended for arbitrage)
    PRIMARY_SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    CORRELATION_PAIRS = {
        'EURUSD': ['GBPUSD', 'USDCHF'],
        'GBPUSD': ['EURUSD', 'GBPJPY'],
        'USDJPY': ['EURJPY', 'GBPJPY'],
        'AUDUSD': ['NZDUSD', 'AUDNZD']
    }
    
    # Position Management
    BASE_LOT = 0.01
    MAX_LOT = 1.0
    MAX_POSITIONS = 8
    
    # Market Making
    MARKET_MAKER_SPREAD = 5  # pips
    MARKET_MAKER_DEPTH = 3   # levels
    
    # Deep Learning
    LSTM_LOOKBACK = 60
    LSTM_FEATURES = 10
    PREDICTION_THRESHOLD = 0.65

# ============================================================================
# REAL-TIME DASHBOARD
# ============================================================================

class TradingDashboard:
    """Real-time trading dashboard with Rich"""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.layout = Layout() if RICH_AVAILABLE else None
        self.stats = {
            'balance': 0,
            'equity': 0,
            'positions': 0,
            'total_pnl': 0,
            'today_pnl': 0,
            'win_rate': 0,
            'total_trades': 0,
            'active_strategies': []
        }
        self.positions = []
        self.recent_trades = deque(maxlen=10)
        self.market_data = {}
        
        if RICH_AVAILABLE:
            self._setup_layout()
    
    def _setup_layout(self):
        """Setup dashboard layout"""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        self.layout["main"].split_row(
            Layout(name="positions"),
            Layout(name="stats"),
            Layout(name="signals")
        )
    
    def update(self, data):
        """Update dashboard with new data"""
        self.stats.update(data.get('stats', {}))
        self.positions = data.get('positions', [])
        self.recent_trades = data.get('recent_trades', self.recent_trades)
        self.market_data = data.get('market_data', {})
        
        if RICH_AVAILABLE:
            self._render()
        else:
            self._simple_display()
    
    def _render(self):
        """Render Rich dashboard"""
        # Header
        header = Panel(
            f"[bold cyan]MASTER AI TRADING SYSTEM[/bold cyan]\n"
            f"Balance: ${self.stats['balance']:.2f} | "
            f"Equity: ${self.stats['equity']:.2f} | "
            f"P&L: ${self.stats['total_pnl']:+.2f}",
            style="bold white on blue"
        )
        self.layout["header"].update(header)
        
        # Positions table
        positions_table = Table(title="Active Positions")
        positions_table.add_column("Symbol", style="cyan")
        positions_table.add_column("Type", style="green")
        positions_table.add_column("Volume")
        positions_table.add_column("P&L", style="yellow")
        positions_table.add_column("Strategy")
        
        for pos in self.positions:
            positions_table.add_row(
                pos['symbol'],
                pos['type'],
                str(pos['volume']),
                f"${pos['pnl']:+.2f}",
                pos['strategy']
            )
        
        self.layout["positions"].update(Panel(positions_table))
        
        # Stats panel
        stats_text = (
            f"[bold]Performance Stats[/bold]\n\n"
            f"Today P&L: ${self.stats['today_pnl']:+.2f}\n"
            f"Total Trades: {self.stats['total_trades']}\n"
            f"Win Rate: {self.stats['win_rate']:.1f}%\n"
            f"Active Strategies: {', '.join(self.stats['active_strategies'])}"
        )
        self.layout["stats"].update(Panel(stats_text))
        
        # Signals panel
        signals_text = "[bold]Recent Signals[/bold]\n\n"
        for trade in self.recent_trades:
            signals_text += f"{trade['time']} | {trade['symbol']} | {trade['action']}\n"
        
        self.layout["signals"].update(Panel(signals_text))
        
        # Footer
        footer = Panel(
            f"Scan Rate: {Config.SCAN_INTERVAL_MS}ms | "
            f"Positions: {self.stats['positions']}/{Config.MAX_POSITIONS} | "
            f"[bold green]● ONLINE[/bold green]",
            style="dim"
        )
        self.layout["footer"].update(footer)
        
        self.console.clear()
        self.console.print(self.layout)
    
    def _simple_display(self):
        """Simple console display"""
        print(f"\r[MASTER AI] Bal: ${self.stats['balance']:.2f} | "
              f"Eq: ${self.stats['equity']:.2f} | "
              f"P&L: ${self.stats['total_pnl']:+.2f} | "
              f"Pos: {self.stats['positions']} | "
              f"Win: {self.stats['win_rate']:.1f}%     ", end='')

# ============================================================================
# DEEP LEARNING LSTM MODEL
# ============================================================================

class DeepLearningPredictor:
    """LSTM-based price prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.prediction_history = deque(maxlen=100)
        
        if ML_AVAILABLE:
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        if DL_AVAILABLE:
            self.build_model()
    
    def build_model(self):
        """Build LSTM model"""
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(Config.LSTM_LOOKBACK, Config.LSTM_FEATURES)),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # Binary classification: up/down
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def prepare_data(self, symbol):
        """Prepare data for LSTM"""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, Config.LSTM_LOOKBACK + 50)
        if rates is None or len(rates) < Config.LSTM_LOOKBACK + 50:
            return None, None
        
        df = pd.DataFrame(rates)
        
        # Create features
        features = pd.DataFrame()
        features['returns'] = df['close'].pct_change()
        features['volume_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
        features['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        features['close_open_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Technical indicators
        features['sma_ratio'] = df['close'] / df['close'].rolling(20).mean()
        features['rsi'] = self._calculate_rsi(df['close'])
        features['macd'] = self._calculate_macd(df['close'])
        
        # Volatility
        features['volatility'] = df['close'].pct_change().rolling(20).std()
        
        # Time features
        features['hour'] = pd.to_datetime(df['time'], unit='s').dt.hour / 24
        features['minute'] = pd.to_datetime(df['time'], unit='s').dt.minute / 60
        
        # Clean data
        features = features.fillna(0)
        features = features.iloc[50:]  # Remove initial NaN rows
        
        # Scale features
        if self.scaler:
            scaled_features = self.scaler.fit_transform(features)
        else:
            # Simple normalization if scaler not available
            scaled_features = (features - features.mean()) / (features.std() + 1e-7)
            scaled_features = scaled_features.values
        
        # Create sequences
        X = []
        for i in range(Config.LSTM_LOOKBACK, len(scaled_features)):
            X.append(scaled_features[i-Config.LSTM_LOOKBACK:i])
        
        return np.array(X), df['close'].iloc[-1]
    
    def predict(self, symbol):
        """Make prediction using LSTM"""
        if not DL_AVAILABLE or self.model is None:
            return None
        
        X, current_price = self.prepare_data(symbol)
        if X is None or len(X) == 0:
            return None
        
        try:
            # Make prediction
            prediction = self.model.predict(X[-1:], verbose=0)
            probability = float(prediction[0][0])
            
            # Store prediction for later validation
            self.prediction_history.append({
                'symbol': symbol,
                'prediction': probability,
                'price': current_price,
                'time': datetime.now()
            })
            
            if probability > Config.PREDICTION_THRESHOLD:
                return {
                    'type': 'BUY',
                    'confidence': probability,
                    'model': 'LSTM',
                    'reason': f'lstm_bullish_{probability:.0%}'
                }
            elif probability < (1 - Config.PREDICTION_THRESHOLD):
                return {
                    'type': 'SELL',
                    'confidence': 1 - probability,
                    'model': 'LSTM',
                    'reason': f'lstm_bearish_{(1-probability):.0%}'
                }
        
        except Exception as e:
            pass
        
        return None
    
    def train_online(self, symbol, outcome):
        """Online training with new data"""
        if not DL_AVAILABLE or self.model is None:
            return
        
        X, _ = self.prepare_data(symbol)
        if X is None or len(X) < 10:
            return
        
        # Create labels based on outcome
        y = np.array([1 if outcome > 0 else 0] * len(X))
        
        try:
            # Quick training on recent data
            self.model.fit(X, y, epochs=1, batch_size=1, verbose=0)
            self.is_trained = True
        except:
            pass
    
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices):
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd - signal

# ============================================================================
# POSITION OPTIMIZER
# ============================================================================

class PositionOptimizer:
    """Optimizes current open positions"""
    
    def __init__(self):
        self.optimization_history = {}
        self.trailing_stops = {}
        
    def optimize_position(self, position):
        """Optimize a single position"""
        
        symbol = position.symbol
        ticket = position.ticket
        
        # Get current market data
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return None
        
        info = mt5.symbol_info(symbol)
        if not info:
            return None
        
        # Calculate current profit in pips
        if position.type == 0:  # Buy
            current_price = tick.bid
            pips = (current_price - position.price_open) / info.point / 10
        else:  # Sell
            current_price = tick.ask
            pips = (position.price_open - current_price) / info.point / 10
        
        optimization = None
        
        # 1. BREAKEVEN MANAGEMENT
        if pips >= 5 and position.sl != position.price_open:
            optimization = {
                'action': 'move_to_breakeven',
                'new_sl': position.price_open,
                'reason': 'secure_breakeven'
            }
        
        # 2. PARTIAL PROFIT TAKING
        elif pips >= 10 and position.volume > Config.BASE_LOT:
            optimization = {
                'action': 'partial_close',
                'close_volume': position.volume * 0.5,
                'reason': 'take_partial_profit'
            }
        
        # 3. TRAILING STOP
        elif pips >= 7:
            trail_distance = 3 * info.point * 10
            
            if position.type == 0:  # Buy
                new_sl = current_price - trail_distance
                if new_sl > position.sl:
                    optimization = {
                        'action': 'trail_stop',
                        'new_sl': new_sl,
                        'reason': f'trail_{pips:.1f}_pips'
                    }
            else:  # Sell
                new_sl = current_price + trail_distance
                if new_sl < position.sl:
                    optimization = {
                        'action': 'trail_stop',
                        'new_sl': new_sl,
                        'reason': f'trail_{pips:.1f}_pips'
                    }
        
        # 4. LOSS MANAGEMENT
        elif pips <= -10:
            optimization = {
                'action': 'reduce_position',
                'close_volume': position.volume * 0.5,
                'reason': 'reduce_loss'
            }
        
        # 5. TIME-BASED OPTIMIZATION
        if ticket not in self.optimization_history:
            self.optimization_history[ticket] = datetime.now()
        
        time_in_position = (datetime.now() - self.optimization_history[ticket]).total_seconds() / 60
        
        if time_in_position > 30 and abs(pips) < 3:
            optimization = {
                'action': 'close',
                'reason': 'timeout_flat'
            }
        
        return optimization
    
    def execute_optimization(self, position, optimization):
        """Execute the optimization action"""
        
        if optimization['action'] == 'move_to_breakeven':
            return self._modify_position(position, optimization['new_sl'], position.tp)
        
        elif optimization['action'] == 'trail_stop':
            return self._modify_position(position, optimization['new_sl'], position.tp)
        
        elif optimization['action'] == 'partial_close':
            return self._partial_close(position, optimization['close_volume'])
        
        elif optimization['action'] == 'reduce_position':
            return self._partial_close(position, optimization['close_volume'])
        
        elif optimization['action'] == 'close':
            return self._close_position(position)
        
        return False
    
    def _modify_position(self, position, new_sl, new_tp):
        """Modify position stops"""
        info = mt5.symbol_info(position.symbol)
        
        request = {
            'action': mt5.TRADE_ACTION_SLTP,
            'symbol': position.symbol,
            'position': position.ticket,
            'sl': round(new_sl, info.digits),
            'tp': round(new_tp, info.digits),
            'magic': Config.MAGIC
        }
        
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE
    
    def _partial_close(self, position, volume):
        """Partially close position"""
        tick = mt5.symbol_info_tick(position.symbol)
        if not tick:
            return False
        
        close_price = tick.bid if position.type == 0 else tick.ask
        
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': position.symbol,
            'volume': round(volume, 2),
            'type': mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
            'position': position.ticket,
            'price': close_price,
            'deviation': 10,
            'magic': Config.MAGIC,
            'comment': 'partial_close',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK
        }
        
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE
    
    def _close_position(self, position):
        """Close entire position"""
        tick = mt5.symbol_info_tick(position.symbol)
        if not tick:
            return False
        
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
            'comment': 'optimized_close',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK
        }
        
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE

# ============================================================================
# VOLUME PROFILE ANALYZER
# ============================================================================

class VolumeProfileAnalyzer:
    """Analyzes volume at price levels"""
    
    def __init__(self):
        self.volume_profiles = {}
        self.poc_levels = {}  # Point of Control
        self.value_areas = {}
        
    def analyze_volume_profile(self, symbol):
        """Create volume profile for symbol"""
        
        # Get historical data
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 288)  # 24 hours
        if rates is None or len(rates) < 100:
            return None
        
        df = pd.DataFrame(rates)
        
        # Calculate price levels
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_range = price_max - price_min
        
        if price_range == 0:
            return None
        
        # Create price bins
        num_bins = 50
        bins = np.linspace(price_min, price_max, num_bins)
        
        # Calculate volume at each price level
        volume_profile = {}
        
        for i in range(len(bins) - 1):
            level_min = bins[i]
            level_max = bins[i + 1]
            
            # Volume that traded in this price range
            mask = (df['low'] <= level_max) & (df['high'] >= level_min)
            volume_at_level = df[mask]['tick_volume'].sum()
            
            if volume_at_level > 0:
                volume_profile[(level_min + level_max) / 2] = volume_at_level
        
        if not volume_profile:
            return None
        
        # Find Point of Control (highest volume level)
        poc = max(volume_profile, key=volume_profile.get)
        
        # Calculate Value Area (70% of volume)
        total_volume = sum(volume_profile.values())
        target_volume = total_volume * 0.7
        
        sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        accumulated_volume = 0
        value_area_levels = []
        
        for level, volume in sorted_levels:
            accumulated_volume += volume
            value_area_levels.append(level)
            if accumulated_volume >= target_volume:
                break
        
        value_area_high = max(value_area_levels)
        value_area_low = min(value_area_levels)
        
        # Store results
        self.volume_profiles[symbol] = volume_profile
        self.poc_levels[symbol] = poc
        self.value_areas[symbol] = {
            'high': value_area_high,
            'low': value_area_low,
            'poc': poc
        }
        
        # Generate trading signal
        current_price = df['close'].iloc[-1]
        
        # Price near POC is strong support/resistance
        if abs(current_price - poc) / current_price < 0.001:
            if df['close'].iloc[-1] > df['close'].iloc[-2]:
                return {
                    'type': 'BUY',
                    'confidence': 0.7,
                    'reason': f'poc_bounce_{poc:.5f}',
                    'levels': self.value_areas[symbol]
                }
            else:
                return {
                    'type': 'SELL',
                    'confidence': 0.7,
                    'reason': f'poc_rejection_{poc:.5f}',
                    'levels': self.value_areas[symbol]
                }
        
        # Value area breakout
        if current_price > value_area_high:
            return {
                'type': 'BUY',
                'confidence': 0.6,
                'reason': 'value_area_breakout_up',
                'levels': self.value_areas[symbol]
            }
        elif current_price < value_area_low:
            return {
                'type': 'SELL',
                'confidence': 0.6,
                'reason': 'value_area_breakout_down',
                'levels': self.value_areas[symbol]
            }
        
        return None

# ============================================================================
# MARKET MAKER STRATEGY
# ============================================================================

class MarketMaker:
    """Places limit orders to capture spread"""
    
    def __init__(self):
        self.active_orders = {}
        self.filled_orders = []
        self.spread_profits = 0
        
    def place_maker_orders(self, symbol):
        """Place buy and sell limit orders"""
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return []
        
        info = mt5.symbol_info(symbol)
        if not info:
            return []
        
        # Calculate order levels
        spread = (tick.ask - tick.bid)
        mid_price = (tick.ask + tick.bid) / 2
        
        orders = []
        
        # Place multiple levels
        for level in range(1, Config.MARKET_MAKER_DEPTH + 1):
            distance = Config.MARKET_MAKER_SPREAD * level * info.point * 10
            
            # Buy limit below
            buy_price = mid_price - distance
            buy_request = {
                'action': mt5.TRADE_ACTION_PENDING,
                'symbol': symbol,
                'volume': Config.BASE_LOT * level,  # Increase size at better prices
                'type': mt5.ORDER_TYPE_BUY_LIMIT,
                'price': round(buy_price, info.digits),
                'sl': round(buy_price - 20 * info.point * 10, info.digits),
                'tp': round(buy_price + Config.MARKET_MAKER_SPREAD * info.point * 10, info.digits),
                'deviation': 10,
                'magic': Config.MAGIC,
                'comment': f'maker_buy_{level}',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_RETURN
            }
            
            # Sell limit above
            sell_price = mid_price + distance
            sell_request = {
                'action': mt5.TRADE_ACTION_PENDING,
                'symbol': symbol,
                'volume': Config.BASE_LOT * level,
                'type': mt5.ORDER_TYPE_SELL_LIMIT,
                'price': round(sell_price, info.digits),
                'sl': round(sell_price + 20 * info.point * 10, info.digits),
                'tp': round(sell_price - Config.MARKET_MAKER_SPREAD * info.point * 10, info.digits),
                'deviation': 10,
                'magic': Config.MAGIC,
                'comment': f'maker_sell_{level}',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_RETURN
            }
            
            orders.append(buy_request)
            orders.append(sell_request)
        
        return orders
    
    def execute_maker_strategy(self, symbol):
        """Execute market making strategy"""
        
        # Cancel old orders
        self.cancel_pending_orders(symbol)
        
        # Place new orders
        orders = self.place_maker_orders(symbol)
        
        placed = 0
        for order_request in orders:
            result = mt5.order_send(order_request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                placed += 1
                self.active_orders[result.order] = {
                    'symbol': symbol,
                    'type': order_request['type'],
                    'price': order_request['price'],
                    'volume': order_request['volume']
                }
        
        return placed > 0
    
    def cancel_pending_orders(self, symbol):
        """Cancel pending orders for symbol"""
        
        orders = mt5.orders_get(symbol=symbol)
        if not orders:
            return
        
        for order in orders:
            if order.magic == Config.MAGIC:
                request = {
                    'action': mt5.TRADE_ACTION_REMOVE,
                    'order': order.ticket
                }
                mt5.order_send(request)

# ============================================================================
# ARBITRAGE DETECTOR
# ============================================================================

class ArbitrageDetector:
    """Detects arbitrage opportunities"""
    
    def __init__(self):
        self.price_differences = {}
        self.arbitrage_history = []
        
    def detect_triangular_arbitrage(self):
        """Detect triangular arbitrage in forex"""
        
        # Example: EUR/USD * USD/JPY = EUR/JPY
        opportunities = []
        
        # Get prices
        eurusd = mt5.symbol_info_tick('EURUSD')
        usdjpy = mt5.symbol_info_tick('USDJPY')
        eurjpy = mt5.symbol_info_tick('EURJPY')
        
        if not all([eurusd, usdjpy, eurjpy]):
            return opportunities
        
        # Calculate synthetic EURJPY
        synthetic_eurjpy_bid = eurusd.bid * usdjpy.bid
        synthetic_eurjpy_ask = eurusd.ask * usdjpy.ask
        
        # Check for arbitrage
        # Buy EURJPY directly, sell synthetically
        if eurjpy.bid > synthetic_eurjpy_ask:
            profit_pips = (eurjpy.bid - synthetic_eurjpy_ask) / mt5.symbol_info('EURJPY').point / 10
            if profit_pips > 1:  # Minimum 1 pip profit
                opportunities.append({
                    'type': 'triangular',
                    'action': 'buy_direct_sell_synthetic',
                    'pairs': ['EURJPY', 'EURUSD', 'USDJPY'],
                    'profit_pips': profit_pips,
                    'confidence': min(0.9, profit_pips / 10)
                })
        
        # Sell EURJPY directly, buy synthetically
        elif synthetic_eurjpy_bid > eurjpy.ask:
            profit_pips = (synthetic_eurjpy_bid - eurjpy.ask) / mt5.symbol_info('EURJPY').point / 10
            if profit_pips > 1:
                opportunities.append({
                    'type': 'triangular',
                    'action': 'sell_direct_buy_synthetic',
                    'pairs': ['EURJPY', 'EURUSD', 'USDJPY'],
                    'profit_pips': profit_pips,
                    'confidence': min(0.9, profit_pips / 10)
                })
        
        return opportunities
    
    def detect_correlation_divergence(self):
        """Detect divergence in correlated pairs"""
        
        opportunities = []
        
        for base_pair, corr_pairs in Config.CORRELATION_PAIRS.items():
            base_rates = mt5.copy_rates_from_pos(base_pair, mt5.TIMEFRAME_M5, 0, 20)
            if base_rates is None:
                continue
            
            base_df = pd.DataFrame(base_rates)
            base_return = (base_df['close'].iloc[-1] - base_df['close'].iloc[-10]) / base_df['close'].iloc[-10]
            
            for corr_pair in corr_pairs:
                corr_rates = mt5.copy_rates_from_pos(corr_pair, mt5.TIMEFRAME_M5, 0, 20)
                if corr_rates is None:
                    continue
                
                corr_df = pd.DataFrame(corr_rates)
                corr_return = (corr_df['close'].iloc[-1] - corr_df['close'].iloc[-10]) / corr_df['close'].iloc[-10]
                
                # Check for divergence
                divergence = abs(base_return - corr_return)
                
                if divergence > 0.002:  # 0.2% divergence
                    if base_return > corr_return:
                        opportunities.append({
                            'type': 'divergence',
                            'pair1': {'symbol': base_pair, 'action': 'SELL'},
                            'pair2': {'symbol': corr_pair, 'action': 'BUY'},
                            'divergence': divergence,
                            'confidence': min(0.8, divergence * 100)
                        })
                    else:
                        opportunities.append({
                            'type': 'divergence',
                            'pair1': {'symbol': base_pair, 'action': 'BUY'},
                            'pair2': {'symbol': corr_pair, 'action': 'SELL'},
                            'divergence': divergence,
                            'confidence': min(0.8, divergence * 100)
                        })
        
        return opportunities

# ============================================================================
# MASTER AI CONTROLLER
# ============================================================================

class MasterAI:
    """Master controller integrating all systems"""
    
    def __init__(self):
        self.connected = False
        
        # Initialize all components
        self.dashboard = TradingDashboard()
        self.deep_learning = DeepLearningPredictor()
        self.optimizer = PositionOptimizer()
        self.volume_analyzer = VolumeProfileAnalyzer()
        self.market_maker = MarketMaker()
        self.arbitrage = ArbitrageDetector()
        
        # Trading state
        self.performance = {
            'balance_start': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0
        }
        
        self.active_strategies = []
        self.trade_history = []
        
        self.initialize()
    
    def initialize(self):
        """Initialize MT5 connection"""
        
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
            self.performance['balance_start'] = account.balance
            print(f"\n[MASTER AI] System Online")
            print(f"[ACCOUNT] Balance: ${account.balance:.2f}")
            print(f"[FEATURES] Dashboard | Deep Learning | Optimizer | Volume Profile | Market Maker | Arbitrage")
    
    def analyze_all_systems(self, symbol):
        """Combine all analysis systems"""
        
        signals = []
        
        # 1. Deep Learning Prediction
        if DL_AVAILABLE:
            dl_signal = self.deep_learning.predict(symbol)
            if dl_signal:
                signals.append(dl_signal)
                if 'DeepLearning' not in self.active_strategies:
                    self.active_strategies.append('DeepLearning')
        
        # 2. Volume Profile Analysis
        volume_signal = self.volume_analyzer.analyze_volume_profile(symbol)
        if volume_signal:
            signals.append(volume_signal)
            if 'VolumeProfile' not in self.active_strategies:
                self.active_strategies.append('VolumeProfile')
        
        # 3. Arbitrage Detection
        arb_opportunities = self.arbitrage.detect_triangular_arbitrage()
        if arb_opportunities:
            for opp in arb_opportunities:
                if symbol in opp.get('pairs', []):
                    signals.append({
                        'type': 'BUY' if 'buy' in opp['action'] else 'SELL',
                        'confidence': opp['confidence'],
                        'reason': f"arbitrage_{opp['profit_pips']:.1f}pips"
                    })
                    if 'Arbitrage' not in self.active_strategies:
                        self.active_strategies.append('Arbitrage')
        
        # Combine signals
        if not signals:
            return None
        
        # Vote on direction
        buy_votes = sum(1 for s in signals if s['type'] == 'BUY')
        sell_votes = sum(1 for s in signals if s['type'] == 'SELL')
        
        if buy_votes == sell_votes:
            return None
        
        direction = 'BUY' if buy_votes > sell_votes else 'SELL'
        avg_confidence = sum(s['confidence'] for s in signals) / len(signals)
        
        return {
            'symbol': symbol,
            'type': direction,
            'confidence': avg_confidence,
            'signals': len(signals),
            'strategies': [s.get('reason', '') for s in signals]
        }
    
    def execute_master_trade(self, signal):
        """Execute trade with all optimizations"""
        
        symbol = signal['symbol']
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        
        if not info or not tick:
            return False
        
        # Dynamic position sizing
        volume = Config.BASE_LOT * (1 + signal['confidence'])
        volume = min(volume, Config.MAX_LOT)
        
        # Setup trade
        if signal['type'] == 'BUY':
            price = tick.ask
            sl = price - 15 * info.point * 10
            tp = price + 20 * info.point * 10
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl = price + 15 * info.point * 10
            tp = price - 20 * info.point * 10
            order_type = mt5.ORDER_TYPE_SELL
        
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': round(volume, 2),
            'type': order_type,
            'price': price,
            'sl': round(sl, info.digits),
            'tp': round(tp, info.digits),
            'deviation': 10,
            'magic': Config.MAGIC,
            'comment': f"AI_{signal['confidence']:.0%}",
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"\n[MASTER] {signal['type']} {symbol} x{volume:.2f}")
            print(f"         Confidence: {signal['confidence']:.0%}")
            print(f"         Strategies: {', '.join(signal['strategies'][:2])}")
            
            self.performance['total_trades'] += 1
            
            # Add to dashboard
            self.dashboard.recent_trades.append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'symbol': symbol,
                'action': f"{signal['type']} {volume:.2f}"
            })
            
            return True
        
        return False
    
    def optimize_all_positions(self):
        """Optimize all open positions"""
        
        positions = mt5.positions_get()
        if not positions:
            return
        
        for position in positions:
            if position.magic != Config.MAGIC:
                continue
            
            # Get optimization suggestion
            optimization = self.optimizer.optimize_position(position)
            
            if optimization:
                success = self.optimizer.execute_optimization(position, optimization)
                if success:
                    print(f"[OPTIMIZED] {position.symbol} - {optimization['reason']}")
    
    def update_dashboard(self):
        """Update dashboard with current data"""
        
        account = mt5.account_info()
        positions = mt5.positions_get()
        
        if not account:
            return
        
        # Prepare dashboard data
        dashboard_data = {
            'stats': {
                'balance': account.balance,
                'equity': account.equity,
                'positions': len([p for p in positions if p.magic == Config.MAGIC]) if positions else 0,
                'total_pnl': account.equity - self.performance['balance_start'],
                'today_pnl': account.balance - self.performance['balance_start'],
                'win_rate': (self.performance['winning_trades'] / self.performance['total_trades'] * 100) if self.performance['total_trades'] > 0 else 0,
                'total_trades': self.performance['total_trades'],
                'active_strategies': self.active_strategies
            },
            'positions': []
        }
        
        if positions:
            for pos in positions:
                if pos.magic == Config.MAGIC:
                    dashboard_data['positions'].append({
                        'symbol': pos.symbol,
                        'type': 'BUY' if pos.type == 0 else 'SELL',
                        'volume': pos.volume,
                        'pnl': pos.profit,
                        'strategy': pos.comment
                    })
        
        self.dashboard.update(dashboard_data)
    
    def run(self):
        """Main AI loop"""
        
        if not self.connected:
            return
        
        print("\n" + "="*70)
        print("MASTER AI TRADING SYSTEM")
        print("="*70)
        print("• Real-time Dashboard")
        print("• Deep Learning LSTM")
        print("• Position Optimizer")
        print("• Volume Profile Analysis")
        print("• Market Making")
        print("• Arbitrage Detection")
        print("="*70)
        
        cycle = 0
        last_dashboard_update = time.time()
        
        try:
            while True:
                cycle += 1
                
                # Check positions
                positions = mt5.positions_get()
                current_positions = len([p for p in positions if p.magic == Config.MAGIC]) if positions else 0
                
                # Scan for new trades
                if current_positions < Config.MAX_POSITIONS:
                    for symbol in Config.PRIMARY_SYMBOLS:
                        # Skip if we have this position
                        if positions:
                            if any(p.symbol == symbol and p.magic == Config.MAGIC for p in positions):
                                continue
                        
                        # Get master signal
                        signal = self.analyze_all_systems(symbol)
                        
                        if signal and signal['confidence'] > 0.5:
                            if self.execute_master_trade(signal):
                                break
                
                # Optimize positions
                if cycle % 10 == 0:
                    self.optimize_all_positions()
                
                # Market making (disabled for now to focus on directional)
                # if cycle % 100 == 0:
                #     for symbol in Config.PRIMARY_SYMBOLS[:1]:
                #         self.market_maker.execute_maker_strategy(symbol)
                
                # Check arbitrage
                if cycle % 50 == 0:
                    arb_opps = self.arbitrage.detect_triangular_arbitrage()
                    if arb_opps:
                        print(f"[ARBITRAGE] Found {len(arb_opps)} opportunities")
                
                # Update dashboard
                if time.time() - last_dashboard_update > Config.DASHBOARD_UPDATE_MS / 1000:
                    self.update_dashboard()
                    last_dashboard_update = time.time()
                
                # Train deep learning model
                if cycle % 1000 == 0 and len(self.trade_history) > 10:
                    for trade in self.trade_history[-10:]:
                        self.deep_learning.train_online(trade['symbol'], trade['profit'])
                
                time.sleep(Config.SCAN_INTERVAL_MS / 1000)
                
        except KeyboardInterrupt:
            self._shutdown()
        finally:
            mt5.shutdown()
    
    def _shutdown(self):
        """Clean shutdown"""
        
        print(f"\n\n[SHUTDOWN] Master AI Offline")
        print(f"\n[FINAL REPORT]")
        print(f"Starting Balance: ${self.performance['balance_start']:.2f}")
        
        account = mt5.account_info()
        if account:
            print(f"Final Balance: ${account.balance:.2f}")
            print(f"Total Profit: ${account.balance - self.performance['balance_start']:.2f}")
        
        print(f"Total Trades: {self.performance['total_trades']}")
        
        if self.performance['total_trades'] > 0:
            win_rate = self.performance['winning_trades'] / self.performance['total_trades'] * 100
            print(f"Win Rate: {win_rate:.1f}%")
        
        # Save performance
        with open('master_ai_performance.json', 'w') as f:
            json.dump(self.performance, f, indent=2)
        
        print("\n[SAVED] Performance saved to master_ai_performance.json")

# ============================================================================
# LAUNCH MASTER AI
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("INITIALIZING MASTER AI SYSTEM...")
    print("="*70)
    
    master = MasterAI()
    
    if master.connected:
        master.run()
    else:
        print("[ERROR] Failed to initialize Master AI")