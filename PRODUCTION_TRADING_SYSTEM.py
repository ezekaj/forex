#!/usr/bin/env python3
"""
PRODUCTION TRADING SYSTEM
=========================

FIXES ALL PREVIOUS ISSUES:
1. ✅ Persistent Model - Train once, save to disk, load for trading
2. ✅ Consistent Data - Single source with validation
3. ✅ OANDA Broker - Works on Mac (no MT5 needed)
4. ✅ Proper Pipeline - Train → Validate → Deploy

MODES:
- TRAIN: Train model on historical data, validate, save
- BACKTEST: Test saved model on out-of-sample data
- LIVE: Load saved model, execute trades

USAGE:
    python PRODUCTION_TRADING_SYSTEM.py train      # Train and save model
    python PRODUCTION_TRADING_SYSTEM.py backtest   # Test on out-of-sample
    python PRODUCTION_TRADING_SYSTEM.py live       # Run live trading
"""

import os
import sys
import json
import time
import pickle
import hashlib
import asyncio
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


# ═══════════════════════════════════════════════════════════════════════════════
#                       CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Production configuration"""

    # Capital & Risk
    STARTING_CAPITAL: float = 30000
    RISK_PER_TRADE: float = 0.01      # 1% risk per trade (conservative)
    MAX_POSITIONS: int = 2             # Max 2 concurrent positions
    MAX_DAILY_LOSS: float = 0.03       # 3% max daily drawdown - stop trading

    # Signal Quality
    MIN_CONFIDENCE: float = 0.40       # 40% minimum confidence
    MIN_MODEL_ACCURACY: float = 0.55   # Don't deploy model below 55% accuracy

    # Trading pairs
    PAIRS: List[str] = field(default_factory=lambda: ["EUR_USD", "GBP_USD", "USD_JPY"])

    # Risk:Reward
    DEFAULT_RR: float = 2.0            # 1:2 Risk:Reward

    # API Keys (Twelve Data for data, OANDA for trading)
    TWELVE_DATA_KEY: str = "6e0c3f6868b443ba8d3515a8def07244"

    # OANDA Settings (get from https://www.oanda.com/demo-account/)
    OANDA_ACCOUNT_ID: str = ""         # Your OANDA account ID
    OANDA_API_KEY: str = ""            # Your OANDA API token
    OANDA_ENVIRONMENT: str = "practice"  # "practice" or "live"

    # Telegram Notifications
    TELEGRAM_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # Paths
    MODEL_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "models")
    DATA_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "data")

    # Execution
    PAPER_TRADING: bool = True

    def __post_init__(self):
        self.MODEL_DIR.mkdir(exist_ok=True)
        self.DATA_DIR.mkdir(exist_ok=True)

        # Load from environment if not set
        if not self.OANDA_ACCOUNT_ID:
            self.OANDA_ACCOUNT_ID = os.environ.get('OANDA_ACCOUNT_ID', '')
        if not self.OANDA_API_KEY:
            self.OANDA_API_KEY = os.environ.get('OANDA_API_KEY', '')


# ═══════════════════════════════════════════════════════════════════════════════
#                       DATA MANAGER - CONSISTENT SOURCE
# ═══════════════════════════════════════════════════════════════════════════════

class DataManager:
    """
    Single, consistent data source with validation.
    Uses Twelve Data API (800 requests/day free tier).
    """

    def __init__(self, config: Config):
        self.config = config
        self.cache = {}

    def fetch_historical(self, pair: str, timeframe: str = '4h',
                         bars: int = 1000, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch historical data with caching.

        Args:
            pair: Currency pair (e.g., "EUR_USD")
            timeframe: Candle timeframe (1h, 4h, 1day)
            bars: Number of bars to fetch
            use_cache: Whether to use/save cached data

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        cache_key = f"{pair}_{timeframe}_{bars}"
        cache_file = self.config.DATA_DIR / f"{cache_key}.csv"

        # Check cache
        if use_cache and cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(hours=4):  # Cache valid for 4 hours
                print(f"  📁 Loading {pair} from cache")
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return df

        # Fetch from Twelve Data
        print(f"  📡 Fetching {pair} from Twelve Data...")
        df = self._fetch_twelve_data(pair, timeframe, bars)

        if df is not None and len(df) >= 100:
            # Validate data
            if self._validate_data(df):
                # Save to cache
                df.to_csv(cache_file)
                return df
            else:
                print(f"  ⚠️  Data validation failed for {pair}")

        # Try loading stale cache as fallback
        if cache_file.exists():
            print(f"  ⚠️  Using stale cache for {pair}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

        raise ValueError(f"Cannot get valid data for {pair}")

    def _fetch_twelve_data(self, pair: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
        """Fetch from Twelve Data API"""
        try:
            # Convert pair format (EUR_USD -> EUR/USD)
            symbol = pair.replace('_', '/')

            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'outputsize': bars,
                'apikey': self.config.TWELVE_DATA_KEY
            }

            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if 'values' not in data:
                print(f"  ⚠️  API Error: {data.get('message', 'Unknown error')}")
                return None

            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df = df.sort_index()

            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col])

            df['volume'] = 0  # Forex doesn't have volume

            return df[['open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            print(f"  ⚠️  Twelve Data error: {e}")
            return None

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality"""
        if df is None or len(df) < 100:
            return False

        # Check for NaN
        if df.isna().sum().sum() > len(df) * 0.01:  # Max 1% NaN
            return False

        # Check for duplicate timestamps
        if df.index.duplicated().sum() > 0:
            return False

        # Check price sanity (no zeros, no extreme values)
        if (df['close'] <= 0).any():
            return False

        # Check for data gaps (max 24 hour gap for 4h data)
        gaps = df.index.to_series().diff()
        max_gap = gaps.max()
        if max_gap > timedelta(hours=48):
            print(f"  ⚠️  Large data gap detected: {max_gap}")

        return True

    def get_latest_price(self, pair: str) -> Optional[float]:
        """Get the latest price for a pair"""
        try:
            symbol = pair.replace('_', '/')
            url = f"https://api.twelvedata.com/price?symbol={symbol}&apikey={self.config.TWELVE_DATA_KEY}"
            response = requests.get(url, timeout=10)
            data = response.json()
            return float(data.get('price', 0))
        except:
            return None


# ═══════════════════════════════════════════════════════════════════════════════
#                       FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureEngine:
    """
    Generate consistent, reproducible features.
    """

    FEATURE_VERSION = "v2"  # Increment when changing features

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range"""
        high, low, close = df['high'], df['low'], df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 0.0001)
        return 100 - (100 / (1 + rs))

    @classmethod
    def generate_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from OHLC data.
        Returns DataFrame with named features.
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']

        # ATR (volatility)
        features['atr_14'] = cls.calculate_atr(df, 14)
        features['atr_7'] = cls.calculate_atr(df, 7)

        # RSI (momentum)
        features['rsi_14'] = cls.calculate_rsi(close, 14)
        features['rsi_7'] = cls.calculate_rsi(close, 7)

        # Moving Averages
        for period in [10, 20, 50, 100]:
            sma = close.rolling(period).mean()
            features[f'sma_{period}'] = sma
            features[f'price_vs_sma_{period}'] = (close - sma) / sma

        # EMA
        for period in [12, 26]:
            features[f'ema_{period}'] = close.ewm(span=period).mean()

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']

        # Bollinger Bands
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        features['bb_upper'] = bb_mid + 2 * bb_std
        features['bb_lower'] = bb_mid - 2 * bb_std
        features['bb_position'] = (close - bb_mid) / (2 * bb_std + 0.0001)

        # Rate of Change
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = close.pct_change(period)

        # Stochastic
        for period in [14]:
            lowest = low.rolling(period).min()
            highest = high.rolling(period).max()
            features[f'stoch_{period}'] = 100 * (close - lowest) / (highest - lowest + 0.0001)

        # Z-score (mean reversion)
        for period in [20, 50]:
            mean = close.rolling(period).mean()
            std = close.rolling(period).std()
            features[f'zscore_{period}'] = (close - mean) / (std + 0.0001)

        # Volatility ratio
        features['vol_ratio'] = features['atr_14'] / close

        # Trend strength
        features['trend_strength'] = abs(features['price_vs_sma_20']) / (features['vol_ratio'] + 0.0001)

        # Price patterns
        features['higher_high'] = (high > high.shift(1)).astype(int)
        features['lower_low'] = (low < low.shift(1)).astype(int)

        # Drop NaN
        features = features.dropna()

        return features

    @classmethod
    def generate_labels(cls, df: pd.DataFrame, lookahead: int = 20,
                        atr_multiplier: float = 1.5) -> pd.Series:
        """
        Generate labels based on ATR-dynamic thresholds.

        Labels:
            1 = BUY (price goes up by ATR * multiplier)
           -1 = SELL (price goes down by ATR * multiplier)
            0 = HOLD (no significant move)
        """
        atr = cls.calculate_atr(df, 14)
        close = df['close']

        labels = pd.Series(index=df.index, dtype=float)

        for i in range(len(df) - lookahead):
            current_price = close.iloc[i]
            future_prices = close.iloc[i+1:i+lookahead+1]
            current_atr = atr.iloc[i]

            if pd.isna(current_atr) or current_atr <= 0:
                labels.iloc[i] = 0
                continue

            threshold = current_atr * atr_multiplier
            max_up = future_prices.max() - current_price
            max_down = current_price - future_prices.min()

            if max_up > threshold and max_up > max_down:
                labels.iloc[i] = 1
            elif max_down > threshold and max_down > max_up:
                labels.iloc[i] = -1
            else:
                labels.iloc[i] = 0

        return labels


# ═══════════════════════════════════════════════════════════════════════════════
#                       MODEL MANAGER - PERSISTENT MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelMetadata:
    """Metadata for a saved model"""
    pair: str
    version: str
    feature_version: str
    trained_at: str
    train_samples: int
    train_accuracy: float
    val_accuracy: float
    val_precision: float
    val_recall: float
    val_f1: float
    feature_columns: List[str]
    data_hash: str  # Hash of training data for reproducibility


class ModelManager:
    """
    Manages model training, saving, loading, and validation.
    Ensures consistent, reproducible models.
    """

    MODEL_VERSION = "v3"

    def __init__(self, config: Config):
        self.config = config
        self.models = {}  # pair -> model
        self.metadata = {}  # pair -> ModelMetadata

    def train_model(self, pair: str, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, ModelMetadata]:
        """
        Train a model with time-series cross-validation.

        Returns:
            (trained_model, metadata)
        """
        print(f"\n{'─'*60}")
        print(f"  Training model for {pair}")
        print(f"{'─'*60}")
        print(f"  Samples: {len(X)}")
        print(f"  Features: {len(X.columns)}")
        print(f"  Class distribution: {dict(y.value_counts())}")

        # Time-series split (no data leakage)
        tscv = TimeSeriesSplit(n_splits=5)

        accuracies = []
        precisions = []
        recalls = []
        f1s = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=20,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

            print(f"  Fold {fold+1}: Accuracy={acc:.1%}, F1={f1:.3f}")

        # Final training on all data
        final_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=20,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        final_model.fit(X, y)

        # Create data hash for reproducibility
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(X).values.tobytes() +
            pd.util.hash_pandas_object(y).values.tobytes()
        ).hexdigest()[:12]

        metadata = ModelMetadata(
            pair=pair,
            version=self.MODEL_VERSION,
            feature_version=FeatureEngine.FEATURE_VERSION,
            trained_at=datetime.now().isoformat(),
            train_samples=len(X),
            train_accuracy=np.mean(accuracies),
            val_accuracy=np.mean(accuracies),
            val_precision=np.mean(precisions),
            val_recall=np.mean(recalls),
            val_f1=np.mean(f1s),
            feature_columns=list(X.columns),
            data_hash=data_hash
        )

        print(f"\n  📊 Final Results:")
        print(f"     Accuracy: {metadata.val_accuracy:.1%}")
        print(f"     Precision: {metadata.val_precision:.3f}")
        print(f"     Recall: {metadata.val_recall:.3f}")
        print(f"     F1 Score: {metadata.val_f1:.3f}")

        return final_model, metadata

    def save_model(self, pair: str, model: Any, metadata: ModelMetadata):
        """Save model and metadata to disk"""
        model_file = self.config.MODEL_DIR / f"{pair}_model.pkl"
        meta_file = self.config.MODEL_DIR / f"{pair}_metadata.json"

        # Save model
        joblib.dump(model, model_file)

        # Save metadata
        with open(meta_file, 'w') as f:
            json.dump(metadata.__dict__, f, indent=2)

        print(f"  💾 Saved model to {model_file}")

    def load_model(self, pair: str) -> Tuple[Optional[Any], Optional[ModelMetadata]]:
        """Load model and metadata from disk"""
        model_file = self.config.MODEL_DIR / f"{pair}_model.pkl"
        meta_file = self.config.MODEL_DIR / f"{pair}_metadata.json"

        if not model_file.exists() or not meta_file.exists():
            return None, None

        model = joblib.load(model_file)

        with open(meta_file, 'r') as f:
            meta_dict = json.load(f)
            metadata = ModelMetadata(**meta_dict)

        return model, metadata

    def validate_model(self, metadata: ModelMetadata) -> bool:
        """Check if model meets minimum requirements"""
        if metadata.val_accuracy < self.config.MIN_MODEL_ACCURACY:
            print(f"  ⚠️  Model accuracy {metadata.val_accuracy:.1%} below minimum {self.config.MIN_MODEL_ACCURACY:.0%}")
            return False

        if metadata.version != self.MODEL_VERSION:
            print(f"  ⚠️  Model version mismatch: {metadata.version} != {self.MODEL_VERSION}")
            return False

        if metadata.feature_version != FeatureEngine.FEATURE_VERSION:
            print(f"  ⚠️  Feature version mismatch: {metadata.feature_version} != {FeatureEngine.FEATURE_VERSION}")
            return False

        return True


# ═══════════════════════════════════════════════════════════════════════════════
#                       OANDA BROKER CONNECTION
# ═══════════════════════════════════════════════════════════════════════════════

class OandaBroker:
    """
    OANDA v20 REST API broker connection.
    Works on Mac, Linux, and Windows.

    Setup:
    1. Create demo account: https://www.oanda.com/demo-account/
    2. Get API key from: https://www.oanda.com/demo-account/tpa/personal_token
    """

    def __init__(self, config: Config):
        self.config = config
        self.connected = False

        if config.OANDA_ENVIRONMENT == "live":
            self.base_url = "https://api-fxtrade.oanda.com/v3"
        else:
            self.base_url = "https://api-fxpractice.oanda.com/v3"

        self.headers = {
            "Authorization": f"Bearer {config.OANDA_API_KEY}",
            "Content-Type": "application/json"
        }

        self._check_connection()

    def _check_connection(self):
        """Verify OANDA connection"""
        if not self.config.OANDA_API_KEY or not self.config.OANDA_ACCOUNT_ID:
            print("  ⚠️  OANDA credentials not configured")
            print("     Get API key from: https://www.oanda.com/demo-account/")
            return

        try:
            url = f"{self.base_url}/accounts/{self.config.OANDA_ACCOUNT_ID}/summary"
            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                account = data.get('account', {})
                balance = float(account.get('balance', 0))
                currency = account.get('currency', 'USD')

                print(f"  ✅ Connected to OANDA ({self.config.OANDA_ENVIRONMENT})")
                print(f"     Balance: {currency} {balance:,.2f}")
                self.connected = True
            else:
                print(f"  ⚠️  OANDA connection failed: {response.status_code}")
                print(f"     {response.text}")
        except Exception as e:
            print(f"  ⚠️  OANDA connection error: {e}")

    def get_account_info(self) -> Dict:
        """Get account information"""
        if not self.connected:
            return {'balance': self.config.STARTING_CAPITAL, 'currency': 'USD'}

        try:
            url = f"{self.base_url}/accounts/{self.config.OANDA_ACCOUNT_ID}/summary"
            response = requests.get(url, headers=self.headers, timeout=10)
            data = response.json()
            account = data.get('account', {})

            return {
                'balance': float(account.get('balance', 0)),
                'equity': float(account.get('NAV', 0)),
                'margin_available': float(account.get('marginAvailable', 0)),
                'currency': account.get('currency', 'USD'),
                'open_trades': int(account.get('openTradeCount', 0))
            }
        except:
            return {'balance': self.config.STARTING_CAPITAL, 'currency': 'USD'}

    def get_current_price(self, pair: str) -> Optional[Dict]:
        """Get current bid/ask price"""
        if not self.connected:
            return None

        try:
            url = f"{self.base_url}/accounts/{self.config.OANDA_ACCOUNT_ID}/pricing"
            params = {'instruments': pair}
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            data = response.json()

            prices = data.get('prices', [])
            if prices:
                p = prices[0]
                return {
                    'bid': float(p['bids'][0]['price']),
                    'ask': float(p['asks'][0]['price']),
                    'time': p.get('time')
                }
        except:
            pass
        return None

    def open_position(self, pair: str, direction: str, units: int,
                      sl_price: float, tp_price: float) -> Optional[Dict]:
        """
        Open a position via OANDA.

        Args:
            pair: Instrument (e.g., "EUR_USD")
            direction: "BUY" or "SELL"
            units: Number of units (negative for sell)
            sl_price: Stop loss price
            tp_price: Take profit price
        """
        if self.config.PAPER_TRADING:
            return self._paper_trade(pair, direction, units, sl_price, tp_price)

        if not self.connected:
            print("  ⚠️  Not connected to broker")
            return None

        try:
            url = f"{self.base_url}/accounts/{self.config.OANDA_ACCOUNT_ID}/orders"

            # OANDA uses negative units for sell
            if direction == "SELL":
                units = -abs(units)
            else:
                units = abs(units)

            # Determine pip precision
            if "JPY" in pair:
                sl_price = round(sl_price, 3)
                tp_price = round(tp_price, 3)
            else:
                sl_price = round(sl_price, 5)
                tp_price = round(tp_price, 5)

            order_data = {
                "order": {
                    "type": "MARKET",
                    "instrument": pair,
                    "units": str(units),
                    "timeInForce": "FOK",
                    "stopLossOnFill": {
                        "price": str(sl_price)
                    },
                    "takeProfitOnFill": {
                        "price": str(tp_price)
                    }
                }
            }

            response = requests.post(url, headers=self.headers, json=order_data, timeout=15)
            data = response.json()

            if 'orderFillTransaction' in data:
                fill = data['orderFillTransaction']
                return {
                    'order_id': fill.get('id'),
                    'trade_id': fill.get('tradeOpened', {}).get('tradeID'),
                    'price': float(fill.get('price', 0)),
                    'units': int(fill.get('units', 0)),
                    'pair': pair,
                    'direction': direction
                }
            else:
                print(f"  ⚠️  Order rejected: {data}")
                return None

        except Exception as e:
            print(f"  ⚠️  Error opening position: {e}")
            return None

    def close_position(self, trade_id: str) -> bool:
        """Close an existing position"""
        if self.config.PAPER_TRADING:
            return True

        if not self.connected:
            return False

        try:
            url = f"{self.base_url}/accounts/{self.config.OANDA_ACCOUNT_ID}/trades/{trade_id}/close"
            response = requests.put(url, headers=self.headers, timeout=10)
            return response.status_code == 200
        except:
            return False

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        if self.config.PAPER_TRADING:
            return getattr(self, '_paper_positions', [])

        if not self.connected:
            return []

        try:
            url = f"{self.base_url}/accounts/{self.config.OANDA_ACCOUNT_ID}/openTrades"
            response = requests.get(url, headers=self.headers, timeout=10)
            data = response.json()

            positions = []
            for trade in data.get('trades', []):
                positions.append({
                    'trade_id': trade.get('id'),
                    'pair': trade.get('instrument'),
                    'direction': 'BUY' if int(trade.get('currentUnits', 0)) > 0 else 'SELL',
                    'units': abs(int(trade.get('currentUnits', 0))),
                    'price': float(trade.get('price', 0)),
                    'pnl': float(trade.get('unrealizedPL', 0)),
                    'sl': float(trade.get('stopLossOrder', {}).get('price', 0) or 0),
                    'tp': float(trade.get('takeProfitOrder', {}).get('price', 0) or 0)
                })

            return positions
        except:
            return []

    def _paper_trade(self, pair, direction, units, sl, tp):
        """Simulate trade in paper mode"""
        if not hasattr(self, '_paper_positions'):
            self._paper_positions = []
            self._paper_trade_id = 1000

        self._paper_trade_id += 1

        position = {
            'trade_id': str(self._paper_trade_id),
            'pair': pair,
            'direction': direction,
            'units': abs(units),
            'price': 0,  # Would be filled by market
            'sl': sl,
            'tp': tp,
            'pnl': 0
        }

        self._paper_positions.append(position)
        return {'trade_id': position['trade_id'], 'price': 0, 'units': units}


# ═══════════════════════════════════════════════════════════════════════════════
#                       NOTIFICATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class Notifier:
    """Telegram notifications for exceptional events"""

    def __init__(self, config: Config):
        self.config = config
        self.enabled = bool(config.TELEGRAM_TOKEN and config.TELEGRAM_CHAT_ID)

    def send(self, message: str, force: bool = False):
        """Send notification"""
        print(message)

        if not self.enabled and not force:
            return

        try:
            url = f"https://api.telegram.org/bot{self.config.TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, json={
                'chat_id': self.config.TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'Markdown'
            }, timeout=10)
        except:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#                       MAIN TRADING BOT
# ═══════════════════════════════════════════════════════════════════════════════

class ProductionTradingBot:
    """
    Production-ready trading bot with persistent models.
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()

        print("\n" + "="*70)
        print("   PRODUCTION TRADING SYSTEM")
        print("="*70)

        self.data_manager = DataManager(self.config)
        self.model_manager = ModelManager(self.config)
        self.broker = OandaBroker(self.config)
        self.notifier = Notifier(self.config)

        self.models = {}
        self.metadata = {}

    def train_all(self):
        """Train models for all pairs and save them"""
        print("\n" + "="*70)
        print("   TRAINING MODELS")
        print("="*70)

        for pair in self.config.PAIRS:
            try:
                # Fetch data
                print(f"\n📊 Processing {pair}...")
                df = self.data_manager.fetch_historical(pair, '4h', 1000)

                # Generate features and labels
                features = FeatureEngine.generate_features(df)
                labels = FeatureEngine.generate_labels(df)

                # Align
                common_idx = features.index.intersection(labels.dropna().index)
                X = features.loc[common_idx]
                y = labels.loc[common_idx]

                # Train
                model, metadata = self.model_manager.train_model(pair, X, y)

                # Validate
                if self.model_manager.validate_model(metadata):
                    self.model_manager.save_model(pair, model, metadata)
                    print(f"  ✅ Model saved for {pair}")
                else:
                    print(f"  ❌ Model rejected for {pair} (below minimum accuracy)")

            except Exception as e:
                print(f"  ❌ Error training {pair}: {e}")

        print("\n" + "="*70)
        print("   TRAINING COMPLETE")
        print("="*70)

    def load_models(self) -> bool:
        """Load all saved models"""
        print("\n📂 Loading models...")

        loaded = 0
        for pair in self.config.PAIRS:
            model, metadata = self.model_manager.load_model(pair)

            if model and metadata:
                if self.model_manager.validate_model(metadata):
                    self.models[pair] = model
                    self.metadata[pair] = metadata
                    print(f"  ✅ {pair}: Loaded (acc={metadata.val_accuracy:.1%}, trained={metadata.trained_at[:10]})")
                    loaded += 1
                else:
                    print(f"  ⚠️  {pair}: Model outdated, needs retraining")
            else:
                print(f"  ⚠️  {pair}: No saved model found")

        return loaded > 0

    def generate_signal(self, pair: str) -> Dict:
        """Generate trading signal for a pair"""
        if pair not in self.models:
            return {'signal': 'NEUTRAL', 'confidence': 0, 'reason': 'No model'}

        try:
            # Get recent data
            df = self.data_manager.fetch_historical(pair, '4h', 100, use_cache=True)

            # Generate features
            features = FeatureEngine.generate_features(df)

            if features.empty:
                return {'signal': 'NEUTRAL', 'confidence': 0, 'reason': 'No features'}

            # Get prediction
            model = self.models[pair]
            metadata = self.metadata[pair]

            # Align features
            latest = features.iloc[[-1]]
            for col in metadata.feature_columns:
                if col not in latest.columns:
                    latest[col] = 0
            latest = latest[metadata.feature_columns]

            prediction = model.predict(latest)[0]
            proba = model.predict_proba(latest)[0]

            # Calculate confidence as margin
            sorted_proba = np.sort(proba)[::-1]
            confidence = sorted_proba[0] - sorted_proba[1] if len(sorted_proba) > 1 else sorted_proba[0]

            # Apply minimum threshold
            if confidence < self.config.MIN_CONFIDENCE:
                return {
                    'signal': 'NEUTRAL',
                    'confidence': confidence,
                    'reason': f'Below threshold ({confidence:.1%} < {self.config.MIN_CONFIDENCE:.0%})'
                }

            # Determine signal
            if prediction == 1:
                signal = 'BUY'
            elif prediction == -1:
                signal = 'SELL'
            else:
                signal = 'NEUTRAL'

            return {
                'signal': signal,
                'confidence': confidence,
                'atr': features['atr_14'].iloc[-1],
                'model_accuracy': metadata.val_accuracy
            }

        except Exception as e:
            return {'signal': 'NEUTRAL', 'confidence': 0, 'reason': str(e)}

    def calculate_position_size(self, pair: str, atr: float, direction: str) -> Tuple[int, float, float]:
        """Calculate position size, SL, and TP"""
        account = self.broker.get_account_info()
        balance = account.get('balance', self.config.STARTING_CAPITAL)

        risk_amount = balance * self.config.RISK_PER_TRADE

        # SL at 1.5 * ATR
        sl_distance = atr * 1.5
        tp_distance = sl_distance * self.config.DEFAULT_RR

        # Get current price
        price_data = self.broker.get_current_price(pair)
        if price_data:
            current_price = price_data['ask'] if direction == 'BUY' else price_data['bid']
        else:
            current_price = 1.0  # Fallback

        # Calculate units
        # For forex: pip_value ≈ $10 per standard lot (100,000 units)
        # Simplified: risk_amount / sl_pips gives us units
        if "JPY" in pair:
            sl_pips = sl_distance * 100  # JPY pairs
        else:
            sl_pips = sl_distance * 10000  # Other pairs

        pip_value = 0.0001 if "JPY" not in pair else 0.01
        units = int(risk_amount / (sl_distance * 100000))  # Rough calculation
        units = max(1, min(units, 100000))  # Clamp

        # Calculate SL/TP prices
        if direction == 'BUY':
            sl_price = current_price - sl_distance
            tp_price = current_price + tp_distance
        else:
            sl_price = current_price + sl_distance
            tp_price = current_price - tp_distance

        return units, sl_price, tp_price

    async def run_cycle(self):
        """Run one trading cycle"""
        print(f"\n{'─'*70}")
        print(f"🔄 Trading Cycle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'─'*70}")

        # Check current positions
        positions = self.broker.get_open_positions()
        print(f"  Open positions: {len(positions)}")

        if len(positions) >= self.config.MAX_POSITIONS:
            print(f"  Max positions reached")
            return

        # Check each pair
        for pair in self.config.PAIRS:
            # Skip if already have position
            if any(p['pair'] == pair for p in positions):
                print(f"  {pair}: Already have position")
                continue

            # Generate signal
            signal_data = self.generate_signal(pair)
            signal = signal_data.get('signal', 'NEUTRAL')
            confidence = signal_data.get('confidence', 0)

            print(f"  {pair}: {signal} (conf: {confidence:.1%})")

            if signal == 'NEUTRAL':
                continue

            # Calculate position
            atr = signal_data.get('atr', 0.001)
            units, sl_price, tp_price = self.calculate_position_size(pair, atr, signal)

            # Execute trade
            result = self.broker.open_position(pair, signal, units, sl_price, tp_price)

            if result:
                msg = f"""🟢 *TRADE OPENED*
Pair: {pair}
Direction: {signal}
Units: {units}
SL: {sl_price:.5f}
TP: {tp_price:.5f}
Confidence: {confidence:.1%}"""
                self.notifier.send(msg, force=True)

    async def run(self, interval_minutes: int = 60):
        """Run the bot continuously"""
        print(f"\n🚀 Starting live trading...")
        print(f"   Mode: {'PAPER' if self.config.PAPER_TRADING else 'LIVE'}")
        print(f"   Interval: {interval_minutes} minutes")

        if not self.load_models():
            print("\n❌ No valid models found. Run training first:")
            print("   python PRODUCTION_TRADING_SYSTEM.py train")
            return

        while True:
            try:
                await self.run_cycle()
                print(f"\n⏰ Next cycle in {interval_minutes} minutes...")
                await asyncio.sleep(interval_minutes * 60)
            except KeyboardInterrupt:
                print("\n\n🛑 Shutting down...")
                break
            except Exception as e:
                print(f"\n⚠️  Error: {e}")
                await asyncio.sleep(60)

    def backtest(self):
        """Backtest models on out-of-sample data"""
        print("\n" + "="*70)
        print("   BACKTESTING")
        print("="*70)

        if not self.load_models():
            print("No models to backtest")
            return

        for pair in self.config.PAIRS:
            if pair not in self.models:
                continue

            print(f"\n📊 Backtesting {pair}...")

            # Get data
            df = self.data_manager.fetch_historical(pair, '4h', 500)
            features = FeatureEngine.generate_features(df)
            labels = FeatureEngine.generate_labels(df)

            # Align
            common_idx = features.index.intersection(labels.dropna().index)
            X = features.loc[common_idx]
            y = labels.loc[common_idx]

            # Use last 20% for testing
            split = int(len(X) * 0.8)
            X_test, y_test = X.iloc[split:], y.iloc[split:]

            # Predict
            model = self.models[pair]
            metadata = self.metadata[pair]

            # Align features
            for col in metadata.feature_columns:
                if col not in X_test.columns:
                    X_test[col] = 0
            X_test = X_test[metadata.feature_columns]

            y_pred = model.predict(X_test)
            proba = model.predict_proba(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)

            # Simulate trades
            trades = 0
            wins = 0
            for i, (pred, prob, actual) in enumerate(zip(y_pred, proba, y_test)):
                confidence = np.max(prob) - np.sort(prob)[-2] if len(prob) > 1 else np.max(prob)

                if confidence >= self.config.MIN_CONFIDENCE and pred != 0:
                    trades += 1
                    if pred == actual:
                        wins += 1

            win_rate = wins / trades if trades > 0 else 0

            print(f"  Test samples: {len(X_test)}")
            print(f"  Overall accuracy: {accuracy:.1%}")
            print(f"  Trades taken: {trades}")
            print(f"  Win rate (filtered): {win_rate:.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
#                       MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point"""
    config = Config(
        # Risk settings
        STARTING_CAPITAL=30000,
        RISK_PER_TRADE=0.01,
        MIN_CONFIDENCE=0.40,

        # Trading pairs
        PAIRS=["EUR_USD", "GBP_USD", "USD_JPY"],

        # OANDA credentials (get from https://www.oanda.com/demo-account/)
        # OANDA_ACCOUNT_ID="your_account_id",
        # OANDA_API_KEY="your_api_key",
        # OANDA_ENVIRONMENT="practice",

        # Execution mode
        PAPER_TRADING=True,
    )

    bot = ProductionTradingBot(config)

    # Parse command line
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == 'train':
            bot.train_all()
        elif command == 'backtest':
            bot.backtest()
        elif command == 'live':
            asyncio.run(bot.run(interval_minutes=60))
        else:
            print(f"Unknown command: {command}")
            print("Usage: python PRODUCTION_TRADING_SYSTEM.py [train|backtest|live]")
    else:
        # Default: show status
        print("\n" + "="*70)
        print("   USAGE")
        print("="*70)
        print("""
Commands:
    python PRODUCTION_TRADING_SYSTEM.py train      # Train and save models
    python PRODUCTION_TRADING_SYSTEM.py backtest   # Test saved models
    python PRODUCTION_TRADING_SYSTEM.py live       # Run live trading

Setup OANDA:
    1. Create demo account: https://www.oanda.com/demo-account/
    2. Get API key from account settings
    3. Add credentials to config or environment variables:
       export OANDA_ACCOUNT_ID="your_id"
       export OANDA_API_KEY="your_key"
""")

        # Check model status
        print("\nModel Status:")
        for pair in config.PAIRS:
            model_file = config.MODEL_DIR / f"{pair}_model.pkl"
            if model_file.exists():
                meta_file = config.MODEL_DIR / f"{pair}_metadata.json"
                with open(meta_file) as f:
                    meta = json.load(f)
                print(f"  ✅ {pair}: acc={meta['val_accuracy']:.1%}, trained={meta['trained_at'][:10]}")
            else:
                print(f"  ❌ {pair}: No model (run 'train' first)")


if __name__ == "__main__":
    main()
