"""
MT5 WALL STREET TRADER - Professional Trading System
====================================================
Think like a hedge fund, trade like a quant
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import threading
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - PROFESSIONAL SETTINGS
# ============================================================================

class Config:
    # Account
    LOGIN = 95948709
    PASSWORD = 'To-4KyLg'
    SERVER = 'MetaQuotes-Demo'
    MAGIC = 777888
    
    # Risk Management (Wall Street Style)
    KELLY_FRACTION = 0.25  # Use 25% of Kelly Criterion
    MAX_PORTFOLIO_RISK = 0.02  # Max 2% portfolio risk
    MAX_CORRELATION_RISK = 0.7  # Max correlation between positions
    VAR_CONFIDENCE = 0.95  # 95% Value at Risk
    
    # Position Sizing
    MIN_POSITION = 0.01
    MAX_POSITION = 1.0
    MAX_POSITIONS = 5
    
    # Execution
    MAX_SLIPPAGE = 2  # pips
    MAX_SPREAD = 2.0  # pips
    
    # Symbols (Majors + Crosses for correlation)
    SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 
               'EURJPY', 'GBPJPY', 'EURGBP', 'AUDNZD']
    
    # Timeframes for multi-timeframe analysis
    TIMEFRAMES = {
        'SCALP': mt5.TIMEFRAME_M1,
        'INTRADAY': mt5.TIMEFRAME_M15,
        'SWING': mt5.TIMEFRAME_H1,
        'POSITION': mt5.TIMEFRAME_H4
    }
    
    # Machine Learning
    ML_LOOKBACK = 100  # Bars for ML features
    ML_CONFIDENCE_THRESHOLD = 0.65  # Min ML confidence

# ============================================================================
# MARKET MICROSTRUCTURE ANALYZER
# ============================================================================

class MarketMicrostructure:
    """Analyze order flow and market microstructure"""
    
    def __init__(self):
        self.order_flow = defaultdict(lambda: deque(maxlen=1000))
        self.volume_profile = defaultdict(dict)
        self.liquidity_map = defaultdict(list)
        
    def analyze_order_flow(self, symbol):
        """Analyze order flow imbalance using price and volume"""
        
        # Get recent rates for volume analysis
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 20)
        if rates is None or len(rates) < 20:
            return {'imbalance': 0, 'pressure': 'neutral'}
        
        df = pd.DataFrame(rates)
        
        # Analyze volume and price relationship
        # Higher volume on up moves = bullish, higher volume on down moves = bearish
        up_moves = df[df['close'] > df['open']]
        down_moves = df[df['close'] < df['open']]
        
        buy_volume = up_moves['tick_volume'].sum() if len(up_moves) > 0 else 0
        sell_volume = down_moves['tick_volume'].sum() if len(down_moves) > 0 else 0
        
        total = buy_volume + sell_volume
        if total == 0:
            return {'imbalance': 0, 'pressure': 'neutral'}
        
        imbalance = (buy_volume - sell_volume) / total
        
        # Also check recent price momentum
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
        
        # Combine volume and price signals
        if imbalance > 0.2 and price_change > 0:
            pressure = 'bullish'
        elif imbalance < -0.2 and price_change < 0:
            pressure = 'bearish'
        else:
            pressure = 'neutral'
        
        return {
            'imbalance': imbalance,
            'pressure': pressure,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume
        }
    
    def find_liquidity_pools(self, symbol):
        """Find where stop losses are likely clustered"""
        
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 50)
        if rates is None or len(rates) < 50:
            return []
        
        df = pd.DataFrame(rates)
        
        # Find recent highs and lows (liquidity pools)
        highs = df['high'].rolling(10).max()
        lows = df['low'].rolling(10).min()
        
        liquidity_zones = []
        
        # Recent highs (sell stops above)
        recent_high = df['high'].iloc[-20:].max()
        if recent_high:
            liquidity_zones.append({
                'level': recent_high,
                'type': 'sell_stops',
                'strength': 'high'
            })
        
        # Recent lows (buy stops below)
        recent_low = df['low'].iloc[-20:].min()
        if recent_low:
            liquidity_zones.append({
                'level': recent_low,
                'type': 'buy_stops',
                'strength': 'high'
            })
        
        return liquidity_zones

# ============================================================================
# QUANTITATIVE ANALYSIS ENGINE
# ============================================================================

class QuantEngine:
    """Statistical and mathematical analysis"""
    
    def __init__(self):
        self.correlations = pd.DataFrame()
        self.volatilities = {}
        self.momentum_scores = {}
        
    def calculate_volatility(self, symbol, period=20):
        """Calculate realized volatility"""
        
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, period)
        if rates is None or len(rates) < period:
            return 0
        
        df = pd.DataFrame(rates)
        returns = np.log(df['close'] / df['close'].shift(1))
        volatility = returns.std() * np.sqrt(96)  # Annualized for 15min bars
        
        self.volatilities[symbol] = volatility
        return volatility
    
    def calculate_sharpe_ratio(self, returns):
        """Calculate Sharpe ratio for strategy performance"""
        
        if len(returns) < 2:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        sharpe = mean_return / std_return * np.sqrt(252)  # Annualized
        return sharpe
    
    def detect_regime(self, symbol):
        """Detect market regime (trending/ranging/volatile)"""
        
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 48)
        if rates is None or len(rates) < 48:
            return 'unknown'
        
        df = pd.DataFrame(rates)
        
        # Calculate indicators
        df['returns'] = df['close'].pct_change()
        df['sma20'] = df['close'].rolling(20).mean()
        df['atr'] = self._calculate_atr(df)
        
        # Trend strength
        if len(df) > 20:
            recent_close = df['close'].iloc[-1]
            sma = df['sma20'].iloc[-1]
            
            if pd.isna(sma):
                trend_strength = 0
            else:
                trend_strength = abs(recent_close - sma) / sma
        else:
            trend_strength = 0
        
        # Volatility
        volatility = df['returns'].std()
        
        # Classify regime
        if trend_strength > 0.01:
            return 'trending'
        elif volatility > 0.02:
            return 'volatile'
        else:
            return 'ranging'
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def find_statistical_edge(self, symbol):
        """Find statistical arbitrage opportunities"""
        
        # Get recent data
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
        if rates is None or len(rates) < 100:
            return None
        
        df = pd.DataFrame(rates)
        
        # Calculate z-score (mean reversion)
        df['sma20'] = df['close'].rolling(20).mean()
        df['std20'] = df['close'].rolling(20).std()
        
        if df['std20'].iloc[-1] == 0 or pd.isna(df['std20'].iloc[-1]):
            return None
        
        z_score = (df['close'].iloc[-1] - df['sma20'].iloc[-1]) / df['std20'].iloc[-1]
        
        # Mean reversion signal
        if z_score > 2:
            return {'type': 'SELL', 'reason': 'overbought', 'z_score': z_score}
        elif z_score < -2:
            return {'type': 'BUY', 'reason': 'oversold', 'z_score': z_score}
        
        return None

# ============================================================================
# MACHINE LEARNING PREDICTOR
# ============================================================================

class MLPredictor:
    """Machine learning for price prediction"""
    
    def __init__(self):
        self.features = []
        self.predictions = {}
        self.model_performance = defaultdict(lambda: {'correct': 0, 'total': 0})
        
    def extract_features(self, symbol):
        """Extract ML features from market data"""
        
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, Config.ML_LOOKBACK)
        if rates is None or len(rates) < Config.ML_LOOKBACK:
            return None
        
        df = pd.DataFrame(rates)
        
        features = {}
        
        # Price features
        features['return_1'] = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100
        features['return_5'] = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100
        features['return_10'] = (df['close'].iloc[-1] / df['close'].iloc[-11] - 1) * 100
        
        # Volume features
        features['volume_ratio'] = df['tick_volume'].iloc[-1] / df['tick_volume'].mean()
        
        # Technical features
        features['rsi'] = self._calculate_rsi(df['close'])
        features['macd_signal'] = self._calculate_macd_signal(df['close'])
        
        # Volatility features
        features['volatility'] = df['close'].pct_change().std()
        
        # Market microstructure
        high_low_ratio = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]
        features['high_low_ratio'] = high_low_ratio * 100
        
        return features
    
    def predict(self, symbol):
        """Predict price movement using ensemble method"""
        
        features = self.extract_features(symbol)
        if not features:
            return None
        
        # Simple ensemble of rules (replace with real ML in production)
        signals = []
        
        # Momentum model
        if features['return_5'] > 0.1 and features['return_1'] > 0:
            signals.append(1)  # Bullish
        elif features['return_5'] < -0.1 and features['return_1'] < 0:
            signals.append(-1)  # Bearish
        
        # Mean reversion model
        if features['rsi'] > 70:
            signals.append(-1)
        elif features['rsi'] < 30:
            signals.append(1)
        
        # Volume model
        if features['volume_ratio'] > 1.5 and features['return_1'] > 0:
            signals.append(1)
        elif features['volume_ratio'] > 1.5 and features['return_1'] < 0:
            signals.append(-1)
        
        # MACD model
        if features['macd_signal'] > 0:
            signals.append(1)
        elif features['macd_signal'] < 0:
            signals.append(-1)
        
        if not signals:
            return None
        
        # Calculate ensemble prediction
        prediction = np.mean(signals)
        confidence = abs(prediction)
        
        if confidence < Config.ML_CONFIDENCE_THRESHOLD:
            return None
        
        return {
            'direction': 'BUY' if prediction > 0 else 'SELL',
            'confidence': confidence,
            'features': features
        }
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        
        if len(prices) < period + 1:
            return 50
        
        deltas = prices.diff()
        gain = deltas.where(deltas > 0, 0).rolling(period).mean()
        loss = -deltas.where(deltas < 0, 0).rolling(period).mean()
        
        if loss.iloc[-1] == 0:
            return 100
        
        rs = gain.iloc[-1] / loss.iloc[-1]
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd_signal(self, prices):
        """Calculate MACD signal"""
        
        if len(prices) < 26:
            return 0
        
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        return (macd.iloc[-1] - signal.iloc[-1])

# ============================================================================
# RISK MANAGEMENT SYSTEM
# ============================================================================

class RiskManager:
    """Professional risk management"""
    
    def __init__(self):
        self.portfolio_heat = 0
        self.daily_trades = 0
        self.daily_pnl = 0
        self.win_rate = 0.5  # Start with 50% assumption
        self.avg_win = 0
        self.avg_loss = 0
        
    def calculate_position_size(self, symbol, stop_distance):
        """Calculate position size using Kelly Criterion"""
        
        account = mt5.account_info()
        if not account:
            return Config.MIN_POSITION
        
        # Kelly formula: f = (p*b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        
        if self.avg_loss == 0:
            kelly_fraction = Config.KELLY_FRACTION * 0.5  # Conservative start
        else:
            b = abs(self.avg_win / self.avg_loss) if self.avg_loss != 0 else 1
            p = self.win_rate
            q = 1 - p
            
            kelly = (p * b - q) / b if b != 0 else 0
            kelly_fraction = min(kelly * Config.KELLY_FRACTION, Config.MAX_PORTFOLIO_RISK)
        
        # Calculate position size
        risk_amount = account.balance * max(kelly_fraction, 0.001)
        
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return Config.MIN_POSITION
        
        pip_value = symbol_info.trade_tick_value * 10
        
        if pip_value == 0 or stop_distance == 0:
            return Config.MIN_POSITION
        
        position_size = risk_amount / (stop_distance * pip_value)
        
        # Apply limits
        position_size = max(Config.MIN_POSITION, min(position_size, Config.MAX_POSITION))
        
        # Round to valid step
        position_size = round(position_size / symbol_info.volume_step) * symbol_info.volume_step
        
        return position_size
    
    def check_correlation_risk(self, symbol, existing_positions):
        """Check correlation risk with existing positions"""
        
        if not existing_positions:
            return True
        
        # Simple correlation check (enhance with real correlation matrix)
        correlated_pairs = {
            'EURUSD': ['GBPUSD', 'EURJPY', 'EURGBP'],
            'GBPUSD': ['EURUSD', 'GBPJPY', 'EURGBP'],
            'USDJPY': ['EURJPY', 'GBPJPY'],
            'AUDUSD': ['AUDNZD', 'NZDUSD']
        }
        
        for position in existing_positions:
            if symbol in correlated_pairs.get(position.symbol, []):
                return False  # Too correlated
        
        return True
    
    def update_performance(self, trade_result):
        """Update performance metrics"""
        
        if trade_result['profit'] > 0:
            self.avg_win = (self.avg_win * self.win_rate + trade_result['profit']) / (self.win_rate + 0.01)
        else:
            self.avg_loss = (self.avg_loss * (1 - self.win_rate) + abs(trade_result['profit'])) / (1 - self.win_rate + 0.01)
        
        # Update win rate
        self.daily_trades += 1
        if trade_result['profit'] > 0:
            self.win_rate = (self.win_rate * (self.daily_trades - 1) + 1) / self.daily_trades
        else:
            self.win_rate = (self.win_rate * (self.daily_trades - 1)) / self.daily_trades
        
        self.daily_pnl += trade_result['profit']

# ============================================================================
# WALL STREET TRADER - MAIN CLASS
# ============================================================================

class WallStreetTrader:
    """Main trading system orchestrator"""
    
    def __init__(self):
        self.connected = False
        self.microstructure = MarketMicrostructure()
        self.quant = QuantEngine()
        self.ml = MLPredictor()
        self.risk = RiskManager()
        self.active_trades = {}
        self.performance_log = []
        
        self.initialize()
    
    def initialize(self):
        """Initialize connection and systems"""
        
        if not mt5.initialize():
            print("[ERROR] MT5 initialization failed")
            return
        
        if not mt5.login(Config.LOGIN, Config.PASSWORD, Config.SERVER):
            print(f"[ERROR] Login failed")
            mt5.shutdown()
            return
        
        account = mt5.account_info()
        if account:
            self.connected = True
            print(f"[CONNECTED] Wall Street Trader Active")
            print(f"[ACCOUNT] Balance: ${account.balance:.2f} | Leverage: 1:{account.leverage}")
    
    def analyze_opportunity(self, symbol):
        """Complete analysis for trading opportunity"""
        
        # 1. Market Regime
        regime = self.quant.detect_regime(symbol)
        
        # 2. Order Flow
        order_flow = self.microstructure.analyze_order_flow(symbol)
        
        # 3. Statistical Edge
        stat_edge = self.quant.find_statistical_edge(symbol)
        
        # 4. ML Prediction
        ml_pred = self.ml.predict(symbol)
        
        # 5. Liquidity Analysis
        liquidity = self.microstructure.find_liquidity_pools(symbol)
        
        # Combine signals
        signals = []
        confidence = 0
        
        if order_flow['pressure'] == 'bullish':
            signals.append('BUY')
            confidence += 0.2
        elif order_flow['pressure'] == 'bearish':
            signals.append('SELL')
            confidence += 0.2
        
        if stat_edge:
            signals.append(stat_edge['type'])
            confidence += 0.3
        
        if ml_pred:
            signals.append(ml_pred['direction'])
            confidence += ml_pred['confidence'] * 0.5
        
        # Decision logic
        if not signals:
            return None
        
        # Majority vote
        buy_signals = signals.count('BUY')
        sell_signals = signals.count('SELL')
        
        if buy_signals > sell_signals and confidence > 0.5:
            return {
                'type': 'BUY',
                'confidence': confidence,
                'regime': regime,
                'reasons': {
                    'order_flow': order_flow['pressure'],
                    'statistical': stat_edge['reason'] if stat_edge else None,
                    'ml_confidence': ml_pred['confidence'] if ml_pred else 0
                }
            }
        elif sell_signals > buy_signals and confidence > 0.5:
            return {
                'type': 'SELL',
                'confidence': confidence,
                'regime': regime,
                'reasons': {
                    'order_flow': order_flow['pressure'],
                    'statistical': stat_edge['reason'] if stat_edge else None,
                    'ml_confidence': ml_pred['confidence'] if ml_pred else 0
                }
            }
        
        return None
    
    def execute_trade(self, symbol, signal):
        """Execute trade with professional risk management"""
        
        # Check correlation risk
        positions = mt5.positions_get()
        if positions:
            if not self.risk.check_correlation_risk(symbol, positions):
                print(f"[SKIP] {symbol} - Too correlated with existing positions")
                return False
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return False
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
        
        # Calculate stop distance based on ATR
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 20)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            atr = self.quant._calculate_atr(df).iloc[-1]
            stop_distance = atr * 2 / symbol_info.point / 10  # 2 ATR stop
        else:
            stop_distance = 20  # Default 20 pips
        
        # Calculate position size
        volume = self.risk.calculate_position_size(symbol, stop_distance)
        
        # Setup trade parameters
        if signal['type'] == 'BUY':
            price = tick.ask
            sl = price - stop_distance * symbol_info.point * 10
            tp = price + stop_distance * 1.5 * symbol_info.point * 10  # 1.5:1 RR
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl = price + stop_distance * symbol_info.point * 10
            tp = price - stop_distance * 1.5 * symbol_info.point * 10
            order_type = mt5.ORDER_TYPE_SELL
        
        # Send order
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': volume,
            'type': order_type,
            'price': price,
            'sl': round(sl, symbol_info.digits),
            'tp': round(tp, symbol_info.digits),
            'deviation': Config.MAX_SLIPPAGE,
            'magic': Config.MAGIC,
            'comment': f"WS_{signal['confidence']:.0%}",
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"\n[TRADE] {signal['type']} {symbol} x{volume:.2f}")
            print(f"        Confidence: {signal['confidence']:.0%} | Regime: {signal['regime']}")
            print(f"        Entry: {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f}")
            print(f"        Reasons: {signal['reasons']}")
            
            self.active_trades[result.order] = {
                'symbol': symbol,
                'type': signal['type'],
                'entry': price,
                'volume': volume,
                'confidence': signal['confidence']
            }
            
            return True
        
        return False
    
    def manage_positions(self):
        """Manage open positions with trailing stops"""
        
        positions = mt5.positions_get()
        if not positions:
            return
        
        for position in positions:
            if position.magic != Config.MAGIC:
                continue
            
            symbol_info = mt5.symbol_info(position.symbol)
            if not symbol_info:
                continue
            
            tick = mt5.symbol_info_tick(position.symbol)
            if not tick:
                continue
            
            # Calculate profit in pips
            if position.type == 0:  # Buy
                pips = (tick.bid - position.price_open) / symbol_info.point / 10
            else:  # Sell
                pips = (position.price_open - tick.ask) / symbol_info.point / 10
            
            # Dynamic trailing stop based on profit
            if pips > 20:
                # Trail tightly when in good profit
                trail_distance = 5
            elif pips > 10:
                trail_distance = 7
            elif pips > 5:
                trail_distance = 10
            else:
                continue  # Don't trail yet
            
            # Calculate new stop loss
            if position.type == 0:  # Buy
                new_sl = tick.bid - trail_distance * symbol_info.point * 10
                if new_sl > position.sl:
                    self.modify_position(position, new_sl, position.tp)
            else:  # Sell
                new_sl = tick.ask + trail_distance * symbol_info.point * 10
                if new_sl < position.sl:
                    self.modify_position(position, new_sl, position.tp)
    
    def modify_position(self, position, new_sl, new_tp):
        """Modify position stops"""
        
        symbol_info = mt5.symbol_info(position.symbol)
        
        request = {
            'action': mt5.TRADE_ACTION_SLTP,
            'symbol': position.symbol,
            'position': position.ticket,
            'sl': round(new_sl, symbol_info.digits),
            'tp': round(new_tp, symbol_info.digits),
            'magic': Config.MAGIC
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[MODIFIED] {position.symbol} SL -> {new_sl:.5f}")
    
    def run(self):
        """Main trading loop"""
        
        if not self.connected:
            return
        
        print("\n" + "="*60)
        print("WALL STREET TRADER - THINK LIKE A QUANT")
        print("="*60)
        print("Systems Active:")
        print("- Market Microstructure Analysis")
        print("- Quantitative Models")
        print("- Machine Learning Predictions")
        print("- Risk Management (Kelly Criterion)")
        print("="*60)
        
        cycle = 0
        
        try:
            while True:
                cycle += 1
                
                # Analyze all symbols
                for symbol in Config.SYMBOLS:
                    # Check if we can take more positions
                    positions = mt5.positions_get()
                    if positions and len([p for p in positions if p.magic == Config.MAGIC]) >= Config.MAX_POSITIONS:
                        break
                    
                    # Skip if we already have this symbol
                    if positions:
                        symbols_held = [p.symbol for p in positions if p.magic == Config.MAGIC]
                        if symbol in symbols_held:
                            continue
                    
                    # Analyze opportunity
                    signal = self.analyze_opportunity(symbol)
                    
                    if signal and signal['confidence'] > 0.6:
                        self.execute_trade(symbol, signal)
                
                # Manage existing positions
                self.manage_positions()
                
                # Display status
                if cycle % 10 == 0:  # Every 2 seconds
                    account = mt5.account_info()
                    positions = mt5.positions_get()
                    
                    if account:
                        open_positions = len([p for p in positions if p.magic == Config.MAGIC]) if positions else 0
                        total_pnl = sum(p.profit for p in positions if p.magic == Config.MAGIC) if positions else 0
                        
                        print(f"\r[SCAN #{cycle}] Balance: ${account.balance:.2f} | Positions: {open_positions}/{Config.MAX_POSITIONS} | P&L: ${total_pnl:.2f} | Win Rate: {self.risk.win_rate:.1%}     ", end='')
                
                time.sleep(0.2)  # 200ms scan rate
                
        except KeyboardInterrupt:
            print("\n\n[SHUTDOWN] Closing Wall Street Trader")
            
            # Log final performance
            if self.performance_log:
                with open('wall_street_performance.json', 'w') as f:
                    json.dump(self.performance_log, f, indent=2)
                
                total_trades = len(self.performance_log)
                profitable = len([t for t in self.performance_log if t['profit'] > 0])
                
                print(f"[PERFORMANCE] Total Trades: {total_trades} | Win Rate: {profitable/total_trades:.1%}")
        
        finally:
            mt5.shutdown()

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    trader = WallStreetTrader()
    if trader.connected:
        trader.run()
    else:
        print("[ERROR] Failed to initialize Wall Street Trader")