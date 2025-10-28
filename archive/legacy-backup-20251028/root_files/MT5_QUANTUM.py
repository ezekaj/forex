"""
MT5 QUANTUM TRADER - Next Generation AI Trading System
=======================================================
Combines News, Sentiment, Neural Networks, and Correlation Trading
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
import threading
import requests
from collections import deque, defaultdict
import pickle
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[WARNING] scikit-learn not installed. ML features disabled.")

# ============================================================================
# QUANTUM CONFIGURATION
# ============================================================================

class Config:
    # Account
    LOGIN = 95948709
    PASSWORD = 'To-4KyLg'
    SERVER = 'MetaQuotes-Demo'
    MAGIC = 222222
    
    # Speed
    SCAN_INTERVAL_MS = 50  # Ultra-fast 50ms
    NEWS_CHECK_INTERVAL = 60  # Check news every minute
    
    # Symbols with correlations
    SYMBOLS = {
        'EURUSD': {'corr': ['GBPUSD', 'USDCHF'], 'inverse': ['USDCHF']},
        'GBPUSD': {'corr': ['EURUSD'], 'inverse': []},
        'USDJPY': {'corr': ['EURJPY', 'GBPJPY'], 'inverse': []},
        'AUDUSD': {'corr': ['NZDUSD'], 'inverse': []},
        'USDCAD': {'corr': [], 'inverse': ['EURUSD']},  # Oil correlation
        'USDCHF': {'corr': [], 'inverse': ['EURUSD', 'GBPUSD']},
    }
    
    # Position Management
    BASE_LOT = 0.01
    MAX_LOT = 0.5
    MAX_POSITIONS = 6
    MAX_RISK = 0.02  # 2% max risk
    
    # ML Settings
    ML_CONFIDENCE_THRESHOLD = 0.65
    ML_TRAINING_SIZE = 1000
    ML_FEATURES = 20
    
    # News Impact Levels
    NEWS_HIGH_IMPACT = 3
    NEWS_MEDIUM_IMPACT = 2
    NEWS_LOW_IMPACT = 1

# ============================================================================
# NEWS TRADING ENGINE
# ============================================================================

class NewsTrader:
    """Trades based on economic news and events"""
    
    def __init__(self):
        self.upcoming_events = []
        self.high_impact_pairs = []
        self.news_positions = {}
        self.last_check = datetime.now()
        
    def fetch_economic_calendar(self):
        """Fetch economic calendar (simulated - use real API in production)"""
        
        # In production, use APIs like:
        # - ForexFactory API
        # - Investing.com API
        # - FXStreet API
        # - Bloomberg API
        
        # Simulated high-impact events
        current_hour = datetime.now().hour
        
        events = []
        
        # Simulate events based on time
        if 8 <= current_hour <= 10:  # European session
            events.append({
                'time': datetime.now() + timedelta(minutes=30),
                'currency': 'EUR',
                'event': 'ECB Rate Decision',
                'impact': Config.NEWS_HIGH_IMPACT,
                'forecast': 'hawkish',
                'pairs': ['EURUSD', 'EURGBP', 'EURJPY']
            })
        
        elif 13 <= current_hour <= 15:  # US session
            events.append({
                'time': datetime.now() + timedelta(minutes=30),
                'currency': 'USD',
                'event': 'NFP',
                'impact': Config.NEWS_HIGH_IMPACT,
                'forecast': 'positive',
                'pairs': ['EURUSD', 'GBPUSD', 'USDJPY']
            })
        
        elif 21 <= current_hour <= 23:  # Asian session
            events.append({
                'time': datetime.now() + timedelta(minutes=30),
                'currency': 'JPY',
                'event': 'BoJ Statement',
                'impact': Config.NEWS_MEDIUM_IMPACT,
                'forecast': 'dovish',
                'pairs': ['USDJPY', 'EURJPY', 'GBPJPY']
            })
        
        self.upcoming_events = events
        return events
    
    def analyze_news_impact(self, symbol):
        """Analyze how news will impact a symbol"""
        
        if not self.upcoming_events:
            return None
        
        for event in self.upcoming_events:
            # Check if event is within 30 minutes
            time_to_event = (event['time'] - datetime.now()).total_seconds() / 60
            
            if 0 < time_to_event < 30 and symbol in event['pairs']:
                # Determine trade direction based on forecast
                base_currency = symbol[:3]
                quote_currency = symbol[3:6]
                
                signal = {
                    'event': event['event'],
                    'impact': event['impact'],
                    'time_to_event': time_to_event,
                    'confidence': 0.7 if event['impact'] == Config.NEWS_HIGH_IMPACT else 0.5
                }
                
                # Logic for trade direction
                if event['currency'] == base_currency:
                    if event['forecast'] in ['hawkish', 'positive']:
                        signal['type'] = 'BUY'
                        signal['reason'] = f"{event['event']} - {base_currency} bullish"
                    else:
                        signal['type'] = 'SELL'
                        signal['reason'] = f"{event['event']} - {base_currency} bearish"
                
                elif event['currency'] == quote_currency:
                    if event['forecast'] in ['hawkish', 'positive']:
                        signal['type'] = 'SELL'
                        signal['reason'] = f"{event['event']} - {quote_currency} bullish"
                    else:
                        signal['type'] = 'BUY'
                        signal['reason'] = f"{event['event']} - {quote_currency} bearish"
                else:
                    continue
                
                return signal
        
        return None
    
    def should_avoid_trading(self, symbol):
        """Check if we should avoid trading due to upcoming news"""
        
        for event in self.upcoming_events:
            time_to_event = (event['time'] - datetime.now()).total_seconds() / 60
            
            # Avoid trading 5 minutes before high impact news
            if event['impact'] == Config.NEWS_HIGH_IMPACT:
                if 0 < time_to_event < 5 and symbol in event['pairs']:
                    return True
        
        return False

# ============================================================================
# CORRELATION MATRIX TRADER
# ============================================================================

class CorrelationTrader:
    """Trades based on currency correlations"""
    
    def __init__(self):
        self.correlation_matrix = self._build_correlation_matrix()
        self.pair_positions = {}
        
    def _build_correlation_matrix(self):
        """Build correlation matrix for currency pairs"""
        
        # Simplified correlation matrix (in production, calculate from historical data)
        correlations = {
            ('EURUSD', 'GBPUSD'): 0.85,   # Strong positive
            ('EURUSD', 'USDCHF'): -0.95,  # Strong negative
            ('GBPUSD', 'USDCHF'): -0.85,  # Strong negative
            ('AUDUSD', 'NZDUSD'): 0.90,   # Strong positive
            ('USDJPY', 'EURJPY'): 0.75,   # Positive
            ('USDCAD', 'Oil'): -0.70,     # Negative (CAD follows oil)
            ('GOLD', 'USDJPY'): -0.60,    # Negative (safe haven)
        }
        
        return correlations
    
    def find_correlation_trades(self, primary_signal):
        """Find correlated trades based on primary signal"""
        
        symbol = primary_signal['symbol']
        direction = primary_signal['type']
        
        correlation_trades = []
        
        # Check correlations
        for (pair1, pair2), correlation in self.correlation_matrix.items():
            if pair1 == symbol:
                # Strong positive correlation
                if correlation > 0.8:
                    correlation_trades.append({
                        'symbol': pair2,
                        'type': direction,  # Same direction
                        'confidence': abs(correlation) * primary_signal['confidence'] * 0.8,
                        'reason': f'correlation_{pair1}_{correlation:.2f}'
                    })
                
                # Strong negative correlation
                elif correlation < -0.8:
                    opposite = 'SELL' if direction == 'BUY' else 'BUY'
                    correlation_trades.append({
                        'symbol': pair2,
                        'type': opposite,  # Opposite direction
                        'confidence': abs(correlation) * primary_signal['confidence'] * 0.8,
                        'reason': f'inverse_correlation_{pair1}_{correlation:.2f}'
                    })
        
        return correlation_trades
    
    def check_divergence(self, symbol1, symbol2):
        """Check for divergence between correlated pairs"""
        
        # Get recent data for both symbols
        rates1 = mt5.copy_rates_from_pos(symbol1, mt5.TIMEFRAME_M5, 0, 20)
        rates2 = mt5.copy_rates_from_pos(symbol2, mt5.TIMEFRAME_M5, 0, 20)
        
        if rates1 is None or rates2 is None:
            return None
        
        df1 = pd.DataFrame(rates1)
        df2 = pd.DataFrame(rates2)
        
        # Calculate returns
        returns1 = (df1['close'].iloc[-1] - df1['close'].iloc[-10]) / df1['close'].iloc[-10]
        returns2 = (df2['close'].iloc[-1] - df2['close'].iloc[-10]) / df2['close'].iloc[-10]
        
        # Check for divergence
        correlation_key = (symbol1, symbol2)
        if correlation_key in self.correlation_matrix:
            expected_correlation = self.correlation_matrix[correlation_key]
            
            # If normally correlated but moving opposite
            if expected_correlation > 0.7:
                if returns1 > 0.001 and returns2 < -0.001:
                    return {
                        'type': 'divergence',
                        'trade1': {'symbol': symbol1, 'type': 'SELL'},  # Overbought
                        'trade2': {'symbol': symbol2, 'type': 'BUY'},   # Oversold
                        'confidence': 0.7
                    }
        
        return None

# ============================================================================
# NEURAL NETWORK PREDICTOR
# ============================================================================

class NeuralNetworkPredictor:
    """ML-based price prediction using neural networks"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.training_data = deque(maxlen=Config.ML_TRAINING_SIZE)
        self.feature_importance = {}
        self.model_accuracy = 0
        
        if ML_AVAILABLE:
            self.initialize_model()
    
    def initialize_model(self):
        """Initialize the neural network model"""
        
        # Multi-layer perceptron for classification
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        
        # Also create Random Forest for feature importance
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    def extract_features(self, symbol):
        """Extract features for ML prediction"""
        
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
        if rates is None or len(rates) < 100:
            return None
        
        df = pd.DataFrame(rates)
        features = []
        
        # Price features
        features.append(df['close'].pct_change().iloc[-1])  # Last return
        features.append(df['close'].pct_change().mean())    # Mean return
        features.append(df['close'].pct_change().std())     # Volatility
        
        # Technical indicators
        features.append(self._calculate_rsi(df['close']))
        features.append(self._calculate_macd(df['close']))
        features.append(self._calculate_bollinger_position(df['close']))
        
        # Volume features
        features.append(df['tick_volume'].iloc[-1] / df['tick_volume'].mean())
        features.append(df['tick_volume'].std() / df['tick_volume'].mean())
        
        # Pattern features
        features.append(self._detect_candlestick_pattern(df))
        features.append(self._calculate_trend_strength(df))
        
        # Market microstructure
        features.append((df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1])
        features.append((df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1])
        
        # Time features
        features.append(datetime.now().hour / 24)  # Hour of day
        features.append(datetime.now().weekday() / 7)  # Day of week
        
        # Momentum features
        for period in [5, 10, 20]:
            if len(df) > period:
                features.append((df['close'].iloc[-1] - df['close'].iloc[-period]) / df['close'].iloc[-period])
        
        # Ensure we have exactly ML_FEATURES
        while len(features) < Config.ML_FEATURES:
            features.append(0)
        
        return np.array(features[:Config.ML_FEATURES])
    
    def predict(self, symbol):
        """Predict price movement using neural network"""
        
        if not ML_AVAILABLE or self.model is None:
            return None
        
        features = self.extract_features(symbol)
        if features is None:
            return None
        
        # Add to training data for future training
        self.training_data.append({
            'features': features,
            'symbol': symbol,
            'time': datetime.now()
        })
        
        # Need enough training data first
        if len(self.training_data) < 100:
            return None
        
        try:
            # Reshape for prediction
            features_scaled = self.scaler.fit_transform(features.reshape(1, -1))
            
            # Get prediction and probability
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            confidence = max(probability)
            
            if confidence > Config.ML_CONFIDENCE_THRESHOLD:
                return {
                    'type': 'BUY' if prediction == 1 else 'SELL',
                    'confidence': confidence,
                    'model_accuracy': self.model_accuracy,
                    'reason': f'neural_network_{confidence:.0%}'
                }
        
        except Exception as e:
            # Model not trained yet
            pass
        
        return None
    
    def train_model(self, trade_history):
        """Train the model on historical trades"""
        
        if not ML_AVAILABLE or len(trade_history) < 50:
            return
        
        # Prepare training data
        X = []
        y = []
        
        for trade in trade_history:
            if 'features' in trade:
                X.append(trade['features'])
                y.append(1 if trade['profit'] > 0 else 0)
        
        if len(X) < 50:
            return
        
        try:
            # Split data
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train neural network
            self.model.fit(X_train_scaled, y_train)
            
            # Calculate accuracy
            self.model_accuracy = self.model.score(X_test_scaled, y_test)
            
            # Train random forest for feature importance
            self.rf_model.fit(X_train, y_train)
            self.feature_importance = dict(enumerate(self.rf_model.feature_importances_))
            
            print(f"\n[ML] Model trained | Accuracy: {self.model_accuracy:.2%}")
            
        except Exception as e:
            print(f"[ML] Training error: {e}")
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    
    def _calculate_macd(self, prices):
        """Calculate MACD signal"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        return (macd.iloc[-1] - signal.iloc[-1]) if len(prices) > 26 else 0
    
    def _calculate_bollinger_position(self, prices, period=20):
        """Calculate position within Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        
        current = prices.iloc[-1]
        
        if pd.isna(upper.iloc[-1]) or pd.isna(lower.iloc[-1]):
            return 0.5
        
        position = (current - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
        return max(0, min(1, position))
    
    def _detect_candlestick_pattern(self, df):
        """Detect candlestick patterns"""
        
        # Simple pattern detection
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        body = abs(last_candle['close'] - last_candle['open'])
        full_range = last_candle['high'] - last_candle['low']
        
        if full_range == 0:
            return 0
        
        # Doji
        if body / full_range < 0.1:
            return 0.5
        
        # Hammer (bullish)
        if last_candle['close'] > last_candle['open'] and body / full_range > 0.6:
            return 1
        
        # Shooting star (bearish)
        if last_candle['close'] < last_candle['open'] and body / full_range > 0.6:
            return -1
        
        return 0
    
    def _calculate_trend_strength(self, df):
        """Calculate trend strength"""
        
        if len(df) < 20:
            return 0
        
        # Simple linear regression slope
        x = np.arange(20)
        y = df['close'].iloc[-20:].values
        
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize by price
        normalized_slope = slope / df['close'].mean()
        
        return normalized_slope * 1000  # Scale up

# ============================================================================
# SENTIMENT ANALYZER
# ============================================================================

class SentimentAnalyzer:
    """Analyzes market sentiment from various sources"""
    
    def __init__(self):
        self.sentiment_scores = {}
        self.social_signals = deque(maxlen=100)
        
    def analyze_market_sentiment(self, symbol):
        """Analyze overall market sentiment"""
        
        # In production, connect to:
        # - Twitter API for forex sentiment
        # - Reddit r/forex sentiment
        # - TradingView ideas sentiment
        # - ForexFactory sentiment
        
        # Simulated sentiment based on time and market conditions
        sentiment_score = self._calculate_technical_sentiment(symbol)
        
        # Add fear & greed index simulation
        fear_greed = self._simulate_fear_greed_index()
        
        # Combine sentiments
        overall_sentiment = (sentiment_score + fear_greed) / 2
        
        self.sentiment_scores[symbol] = overall_sentiment
        
        if overall_sentiment > 0.6:
            return {
                'type': 'BUY',
                'confidence': overall_sentiment,
                'reason': f'bullish_sentiment_{overall_sentiment:.0%}'
            }
        elif overall_sentiment < -0.6:
            return {
                'type': 'SELL',
                'confidence': abs(overall_sentiment),
                'reason': f'bearish_sentiment_{overall_sentiment:.0%}'
            }
        
        return None
    
    def _calculate_technical_sentiment(self, symbol):
        """Calculate sentiment from technical indicators"""
        
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 24)
        if rates is None or len(rates) < 24:
            return 0
        
        df = pd.DataFrame(rates)
        
        bullish_signals = 0
        bearish_signals = 0
        
        # Moving average sentiment
        if df['close'].iloc[-1] > df['close'].mean():
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # Momentum sentiment
        if df['close'].iloc[-1] > df['close'].iloc[-12]:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # Volume sentiment
        if df['tick_volume'].iloc[-1] > df['tick_volume'].mean():
            if df['close'].iloc[-1] > df['open'].iloc[-1]:
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        # Calculate sentiment score
        total_signals = bullish_signals + bearish_signals
        if total_signals == 0:
            return 0
        
        sentiment = (bullish_signals - bearish_signals) / total_signals
        return sentiment
    
    def _simulate_fear_greed_index(self):
        """Simulate fear & greed index"""
        
        # In production, fetch from:
        # - CNN Fear & Greed Index
        # - Crypto Fear & Greed Index
        # - VIX for general market fear
        
        hour = datetime.now().hour
        
        # Simulate based on trading session
        if 8 <= hour <= 16:  # European/US session
            # More greed during active hours
            return np.random.uniform(0.3, 0.7)
        else:
            # More fear during off hours
            return np.random.uniform(-0.3, 0.3)

# ============================================================================
# QUANTUM TRADER - MASTER CONTROLLER
# ============================================================================

class QuantumTrader:
    """Master AI trading system combining all strategies"""
    
    def __init__(self):
        self.connected = False
        
        # Initialize all components
        self.news_trader = NewsTrader()
        self.correlation_trader = CorrelationTrader()
        self.neural_network = NeuralNetworkPredictor()
        self.sentiment = SentimentAnalyzer()
        
        # Trading state
        self.positions = {}
        self.trade_history = []
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0,
            'best_trade': 0,
            'worst_trade': 0
        }
        
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
            self.initial_balance = account.balance
            print(f"\n[QUANTUM] System Online")
            print(f"[ACCOUNT] Balance: ${account.balance:.2f}")
            print(f"[SYSTEMS] News | Correlation | Neural Network | Sentiment")
    
    def analyze_all_signals(self, symbol):
        """Combine all trading signals"""
        
        signals = []
        total_confidence = 0
        
        # 1. News Analysis
        news_signal = self.news_trader.analyze_news_impact(symbol)
        if news_signal:
            signals.append(news_signal)
            total_confidence += news_signal['confidence'] * 0.3
        
        # 2. Neural Network Prediction
        ml_signal = self.neural_network.predict(symbol)
        if ml_signal:
            signals.append(ml_signal)
            total_confidence += ml_signal['confidence'] * 0.25
        
        # 3. Sentiment Analysis
        sentiment_signal = self.sentiment.analyze_market_sentiment(symbol)
        if sentiment_signal:
            signals.append(sentiment_signal)
            total_confidence += sentiment_signal['confidence'] * 0.2
        
        # 4. Technical Analysis (from previous implementation)
        tech_signal = self._analyze_technical(symbol)
        if tech_signal:
            signals.append(tech_signal)
            total_confidence += tech_signal['confidence'] * 0.25
        
        if not signals:
            return None
        
        # Vote on direction
        buy_votes = sum(1 for s in signals if s['type'] == 'BUY')
        sell_votes = sum(1 for s in signals if s['type'] == 'SELL')
        
        if buy_votes > sell_votes:
            direction = 'BUY'
        elif sell_votes > buy_votes:
            direction = 'SELL'
        else:
            return None
        
        # Average confidence
        avg_confidence = total_confidence / len(signals)
        
        return {
            'symbol': symbol,
            'type': direction,
            'confidence': avg_confidence,
            'signals': len(signals),
            'components': [s.get('reason', '') for s in signals]
        }
    
    def _analyze_technical(self, symbol):
        """Basic technical analysis"""
        
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 20)
        if rates is None or len(rates) < 20:
            return None
        
        df = pd.DataFrame(rates)
        
        # Simple momentum
        change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
        
        if abs(change) > 0.0002:
            return {
                'type': 'BUY' if change > 0 else 'SELL',
                'confidence': min(abs(change) * 500, 0.8),
                'reason': f'technical_momentum_{change:.4f}'
            }
        
        return None
    
    def execute_quantum_trade(self, signal):
        """Execute trade with quantum logic"""
        
        symbol = signal['symbol']
        
        # Check if news is coming
        if self.news_trader.should_avoid_trading(symbol):
            print(f"[SKIP] {symbol} - High impact news coming")
            return False
        
        # Get symbol info
        info = mt5.symbol_info(symbol)
        if not info:
            return False
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
        
        # Dynamic position sizing based on confidence
        base_lot = Config.BASE_LOT
        
        if signal['confidence'] > 0.8:
            lot_multiplier = 5
        elif signal['confidence'] > 0.7:
            lot_multiplier = 3
        elif signal['confidence'] > 0.6:
            lot_multiplier = 2
        else:
            lot_multiplier = 1
        
        volume = min(base_lot * lot_multiplier, Config.MAX_LOT)
        
        # Dynamic stops based on volatility
        atr = self._calculate_atr(symbol)
        stop_distance = max(10, min(30, atr * 2))  # 10-30 pips
        
        # Setup trade
        if signal['type'] == 'BUY':
            price = tick.ask
            sl = price - stop_distance * info.point * 10
            tp = price + stop_distance * 1.5 * info.point * 10
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl = price + stop_distance * info.point * 10
            tp = price - stop_distance * 1.5 * info.point * 10
            order_type = mt5.ORDER_TYPE_SELL
        
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
            'comment': f"Q_{signal['confidence']:.0%}",
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"\n[QUANTUM] {signal['type']} {symbol} x{volume:.2f}")
            print(f"          Confidence: {signal['confidence']:.0%} | Signals: {signal['signals']}")
            print(f"          Components: {', '.join(signal['components'][:3])}")
            print(f"          Entry: {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f}")
            
            # Check for correlation trades
            correlation_trades = self.correlation_trader.find_correlation_trades(signal)
            for corr_trade in correlation_trades[:1]:  # Take best correlation trade
                if corr_trade['confidence'] > 0.6:
                    self.execute_quantum_trade(corr_trade)
            
            self.performance['total_trades'] += 1
            return True
        
        return False
    
    def _calculate_atr(self, symbol, period=14):
        """Calculate ATR for dynamic stops"""
        
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, period + 1)
        if rates is None or len(rates) < period:
            return 15  # Default
        
        df = pd.DataFrame(rates)
        
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean().iloc[-1]
        
        # Convert to pips
        info = mt5.symbol_info(symbol)
        if info:
            return atr / info.point / 10
        
        return 15
    
    def manage_quantum_positions(self):
        """Advanced position management"""
        
        positions = mt5.positions_get()
        if not positions:
            return
        
        for position in positions:
            if position.magic != Config.MAGIC:
                continue
            
            # Smart exit logic
            if position.profit > 10:  # $10 profit
                self._close_position(position, "profit_target")
            elif position.profit < -15:  # $15 loss
                self._close_position(position, "stop_loss")
            
            # Trail profitable positions
            elif position.profit > 5:
                self._trail_position(position)
    
    def _trail_position(self, position):
        """Trail stop for profitable positions"""
        
        info = mt5.symbol_info(position.symbol)
        tick = mt5.symbol_info_tick(position.symbol)
        
        if not info or not tick:
            return
        
        trail_distance = 5 * info.point * 10  # 5 pips
        
        if position.type == 0:  # Buy
            new_sl = tick.bid - trail_distance
            if new_sl > position.sl:
                self._modify_position(position, new_sl, position.tp)
        else:  # Sell
            new_sl = tick.ask + trail_distance
            if new_sl < position.sl:
                self._modify_position(position, new_sl, position.tp)
    
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
        
        mt5.order_send(request)
    
    def _close_position(self, position, reason):
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
            'deviation': 10,
            'magic': Config.MAGIC,
            'comment': reason,
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[CLOSED] {position.symbol} | ${position.profit:.2f} | {reason}")
            
            # Update performance
            self.performance['total_profit'] += position.profit
            if position.profit > 0:
                self.performance['winning_trades'] += 1
            
            # Record for ML training
            self.trade_history.append({
                'symbol': position.symbol,
                'profit': position.profit,
                'reason': reason,
                'time': datetime.now().isoformat()
            })
    
    def run(self):
        """Main quantum trading loop"""
        
        if not self.connected:
            return
        
        print("\n" + "="*70)
        print("QUANTUM TRADER - NEXT GENERATION AI SYSTEM")
        print("="*70)
        print("Active Systems:")
        print("• News Trading Engine")
        print("• Correlation Matrix")
        print("• Neural Network ML")
        print("• Sentiment Analysis")
        print("="*70)
        
        cycle = 0
        last_news_check = time.time()
        
        try:
            while True:
                cycle += 1
                
                # Check news periodically
                if time.time() - last_news_check > Config.NEWS_CHECK_INTERVAL:
                    self.news_trader.fetch_economic_calendar()
                    last_news_check = time.time()
                
                # Check positions
                positions = mt5.positions_get()
                current_positions = len([p for p in positions if p.magic == Config.MAGIC]) if positions else 0
                
                if current_positions < Config.MAX_POSITIONS:
                    # Analyze all symbols
                    for symbol in Config.SYMBOLS.keys():
                        # Skip if we have this position
                        if positions:
                            if any(p.symbol == symbol and p.magic == Config.MAGIC for p in positions):
                                continue
                        
                        # Get quantum signal
                        signal = self.analyze_all_signals(symbol)
                        
                        if signal and signal['confidence'] > 0.5:
                            if self.execute_quantum_trade(signal):
                                break
                
                # Manage positions
                self.manage_quantum_positions()
                
                # Train ML model periodically
                if cycle % 1000 == 0 and len(self.trade_history) > 50:
                    self.neural_network.train_model(self.trade_history)
                
                # Display status
                if cycle % 20 == 0:
                    self._display_status()
                
                time.sleep(Config.SCAN_INTERVAL_MS / 1000)
                
        except KeyboardInterrupt:
            self._shutdown()
        finally:
            mt5.shutdown()
    
    def _display_status(self):
        """Display trading status"""
        
        account = mt5.account_info()
        if not account:
            return
        
        positions = mt5.positions_get()
        current_positions = len([p for p in positions if p.magic == Config.MAGIC]) if positions else 0
        current_pnl = sum(p.profit for p in positions if p.magic == Config.MAGIC) if positions else 0
        
        profit_today = account.balance - self.initial_balance
        win_rate = (self.performance['winning_trades'] / self.performance['total_trades'] * 100) if self.performance['total_trades'] > 0 else 0
        
        print(f"\r[QUANTUM] Bal: ${account.balance:.2f} | Today: ${profit_today:+.2f} | Pos: {current_positions}/{Config.MAX_POSITIONS} | P&L: ${current_pnl:+.2f} | Trades: {self.performance['total_trades']} | Win: {win_rate:.0f}%     ", end='')
    
    def _shutdown(self):
        """Clean shutdown"""
        
        print(f"\n\n[SHUTDOWN] Quantum Trader Offline")
        print(f"\n[PERFORMANCE REPORT]")
        print(f"Total Trades: {self.performance['total_trades']}")
        print(f"Winning Trades: {self.performance['winning_trades']}")
        print(f"Total Profit: ${self.performance['total_profit']:.2f}")
        
        if self.performance['total_trades'] > 0:
            win_rate = self.performance['winning_trades'] / self.performance['total_trades'] * 100
            print(f"Win Rate: {win_rate:.1f}%")
        
        # Save everything
        with open('quantum_performance.json', 'w') as f:
            json.dump(self.performance, f, indent=2)
        
        with open('quantum_trades.json', 'w') as f:
            json.dump(self.trade_history, f, indent=2)
        
        print("\n[SAVED] Performance data saved to files")

# ============================================================================
# LAUNCH QUANTUM SYSTEM
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("INITIALIZING QUANTUM TRADING SYSTEM...")
    print("="*70)
    
    trader = QuantumTrader()
    
    if trader.connected:
        trader.run()
    else:
        print("[ERROR] Failed to initialize Quantum Trader")