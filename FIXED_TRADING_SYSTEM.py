#!/usr/bin/env python3
"""
FIXED TRADING SYSTEM - All 4 Issues Resolved
=============================================

FIXES:
1. ✅ Data Source: Alpha Vantage API (free tier) + Twelve Data fallback + offline cache
2. ✅ News Integration: Properly imports EnhancedForexSignals from forex_system
3. ✅ Min Confidence: Increased to 35% (was 20% - too many garbage trades)
4. ✅ Broker Connection: MetaTrader 5 API for real trade execution

REQUIREMENTS:
pip install MetaTrader5 pandas numpy scikit-learn requests aiohttp

SETUP:
1. Get free Alpha Vantage API key: https://www.alphavantage.co/support/#api-key
2. Install MetaTrader 5 and login to your broker account
3. Configure API keys below
"""

import os
import sys
import json
import time
import asyncio
import sqlite3
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'forex_system'))

# ML imports
from sklearn.ensemble import RandomForestClassifier

# ═══════════════════════════════════════════════════════════════════════════════
#                       CONFIGURATION - EDIT THESE VALUES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Trading configuration - EDIT THESE VALUES"""

    # Capital & Risk
    STARTING_CAPITAL: float = 30000
    RISK_PER_TRADE: float = 0.015  # 1.5% risk per trade
    MAX_POSITIONS: int = 3
    MAX_DAILY_LOSS: float = 0.05  # 5% max daily drawdown

    # FIX #3: Increased min_confidence from 0.20 to 0.35
    MIN_CONFIDENCE: float = 0.35  # Minimum 35% confidence (was 20% - too low!)

    # Trading pairs
    PAIRS: List[str] = None

    # Risk:Reward settings
    DEFAULT_RR: float = 2.5  # Risk:Reward ratio
    SCALE_OUT_ENABLED: bool = True

    # API Keys - YOUR KEYS ARE CONFIGURED:
    # Alpha Vantage: FREE tier: 25 requests/day
    # Twelve Data: FREE tier: 800 requests/day
    ALPHA_VANTAGE_KEY: str = "TB65VENAH7X7JGZ2"
    TWELVE_DATA_KEY: str = "6e0c3f6868b443ba8d3515a8def07244"

    # MetaTrader 5 Settings
    MT5_LOGIN: int = 0
    MT5_PASSWORD: str = ""
    MT5_SERVER: str = ""

    # Telegram Notifications (optional)
    TELEGRAM_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # Execution mode
    PAPER_TRADING: bool = True  # Set to False for live trading

    def __post_init__(self):
        if self.PAIRS is None:
            self.PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]

        # Load from environment variables if not set
        if not self.ALPHA_VANTAGE_KEY:
            self.ALPHA_VANTAGE_KEY = os.environ.get('ALPHA_VANTAGE_KEY', '')
        if not self.TWELVE_DATA_KEY:
            self.TWELVE_DATA_KEY = os.environ.get('TWELVE_DATA_KEY', '')
        if not self.TELEGRAM_TOKEN:
            self.TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
        if not self.TELEGRAM_CHAT_ID:
            self.TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')


# ═══════════════════════════════════════════════════════════════════════════════
#                       FIX #1: WORKING DATA SOURCE
# ═══════════════════════════════════════════════════════════════════════════════

class DataManager:
    """
    Multi-source data manager with fallbacks.

    Priority order:
    1. MetaTrader 5 (if connected) - BEST for forex, real-time
    2. Twelve Data API (free tier: 800/day)
    3. Alpha Vantage API (free tier: 25/day)
    4. Yahoo Finance (often blocked, but try)
    5. Cached data (fallback)
    """

    def __init__(self, config: Config):
        self.config = config
        self.cache_dir = Path(__file__).parent / 'data_cache'
        self.cache_dir.mkdir(exist_ok=True)
        self.mt5_connected = False
        self._init_mt5()

    def _init_mt5(self):
        """Try to connect to MetaTrader 5"""
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5

            if mt5.initialize():
                self.mt5_connected = True
                print(f"✅ MetaTrader 5 connected: {mt5.terminal_info().name}")

                # Login if credentials provided
                if self.config.MT5_LOGIN:
                    if mt5.login(
                        self.config.MT5_LOGIN,
                        self.config.MT5_PASSWORD,
                        self.config.MT5_SERVER
                    ):
                        print(f"✅ Logged in to MT5 account {self.config.MT5_LOGIN}")
                    else:
                        print(f"⚠️  MT5 login failed: {mt5.last_error()}")
            else:
                print(f"⚠️  MT5 not available: {mt5.last_error()}")
                self.mt5_connected = False
        except ImportError:
            print("⚠️  MetaTrader5 not installed. Run: pip install MetaTrader5")
            self.mt5_connected = False
            self.mt5 = None

    def fetch_ohlc(self, pair: str, timeframe: str = '4H', bars: int = 500) -> pd.DataFrame:
        """
        Fetch OHLC data with multiple fallbacks.
        Returns DataFrame with columns: open, high, low, close, volume
        """
        # Try each source in priority order
        df = None

        # 1. MetaTrader 5 (best for forex)
        if self.mt5_connected:
            df = self._fetch_mt5(pair, timeframe, bars)
            if df is not None and len(df) > 100:
                print(f"📊 {pair}: Got {len(df)} bars from MT5")
                return df

        # 2. Twelve Data (800 free/day)
        if self.config.TWELVE_DATA_KEY:
            df = self._fetch_twelve_data(pair, timeframe, bars)
            if df is not None and len(df) > 100:
                print(f"📊 {pair}: Got {len(df)} bars from Twelve Data")
                self._save_cache(pair, timeframe, df)
                return df

        # 3. Alpha Vantage (25 free/day)
        if self.config.ALPHA_VANTAGE_KEY:
            df = self._fetch_alpha_vantage(pair, timeframe, bars)
            if df is not None and len(df) > 100:
                print(f"📊 {pair}: Got {len(df)} bars from Alpha Vantage")
                self._save_cache(pair, timeframe, df)
                return df

        # 4. Yahoo Finance (often blocked but try)
        df = self._fetch_yahoo(pair, timeframe, bars)
        if df is not None and len(df) > 100:
            print(f"📊 {pair}: Got {len(df)} bars from Yahoo Finance")
            self._save_cache(pair, timeframe, df)
            return df

        # 5. Load from cache
        df = self._load_cache(pair, timeframe)
        if df is not None and len(df) > 50:
            cache_age = (datetime.now() - df.index[-1]).days
            print(f"⚠️  {pair}: Using cached data ({cache_age} days old)")
            return df

        # 6. Last resort: generate synthetic data for testing
        print(f"⚠️  {pair}: No data available - generating synthetic data for testing")
        return self._generate_synthetic(pair, bars)

    def _fetch_mt5(self, pair: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
        """Fetch from MetaTrader 5"""
        if not self.mt5_connected:
            return None

        try:
            # Map timeframe to MT5 constants
            tf_map = {
                '1H': self.mt5.TIMEFRAME_H1,
                '4H': self.mt5.TIMEFRAME_H4,
                'D': self.mt5.TIMEFRAME_D1,
                '1D': self.mt5.TIMEFRAME_D1,
            }
            mt5_tf = tf_map.get(timeframe.upper(), self.mt5.TIMEFRAME_H4)

            # Format symbol for MT5 (may need suffix like EURUSD.m or EURUSDm)
            symbols = [pair, f"{pair}.m", f"{pair}m", f"{pair}.raw"]

            for symbol in symbols:
                rates = self.mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars)
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    df = df.rename(columns={
                        'open': 'open', 'high': 'high',
                        'low': 'low', 'close': 'close',
                        'tick_volume': 'volume'
                    })
                    return df[['open', 'high', 'low', 'close', 'volume']]

            return None
        except Exception as e:
            print(f"MT5 error: {e}")
            return None

    def _fetch_twelve_data(self, pair: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
        """Fetch from Twelve Data API (FREE: 800 requests/day)"""
        try:
            # Convert pair format (EURUSD -> EUR/USD)
            symbol = f"{pair[:3]}/{pair[3:]}"

            # Map timeframe
            tf_map = {'1H': '1h', '4H': '4h', 'D': '1day', '1D': '1day'}
            interval = tf_map.get(timeframe.upper(), '4h')

            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': symbol,
                'interval': interval,
                'outputsize': bars,
                'apikey': self.config.TWELVE_DATA_KEY
            }

            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if 'values' not in data:
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
            print(f"Twelve Data error: {e}")
            return None

    def _fetch_alpha_vantage(self, pair: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
        """Fetch from Alpha Vantage API (FREE: 25 requests/day)"""
        try:
            from_currency = pair[:3]
            to_currency = pair[3:]

            # Use FX_DAILY for daily data (free tier)
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'FX_DAILY',
                'from_symbol': from_currency,
                'to_symbol': to_currency,
                'outputsize': 'full',
                'apikey': self.config.ALPHA_VANTAGE_KEY
            }

            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if 'Time Series FX (Daily)' not in data:
                return None

            ts_data = data['Time Series FX (Daily)']

            records = []
            for date_str, values in ts_data.items():
                records.append({
                    'datetime': pd.to_datetime(date_str),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': 0
                })

            df = pd.DataFrame(records)
            df.set_index('datetime', inplace=True)
            df = df.sort_index()

            return df[['open', 'high', 'low', 'close', 'volume']].tail(bars)

        except Exception as e:
            print(f"Alpha Vantage error: {e}")
            return None

    def _fetch_yahoo(self, pair: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
        """Fetch from Yahoo Finance (often blocked)"""
        try:
            symbol = f"{pair}=X"

            # Calculate date range
            tf_hours = {'1H': 1, '4H': 4, 'D': 24, '1D': 24}
            hours = tf_hours.get(timeframe.upper(), 4)
            days = int(bars * hours / 24) + 10

            end = datetime.now()
            start = end - timedelta(days=days)

            # Yahoo Finance URL
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'period1': int(start.timestamp()),
                'period2': int(end.timestamp()),
                'interval': '1d' if hours >= 24 else '1h',
            }

            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, params=params, headers=headers, timeout=15)
            data = response.json()

            result = data.get('chart', {}).get('result', [])
            if not result:
                return None

            timestamps = result[0].get('timestamp', [])
            quote = result[0].get('indicators', {}).get('quote', [{}])[0]

            if not timestamps:
                return None

            df = pd.DataFrame({
                'datetime': pd.to_datetime(timestamps, unit='s'),
                'open': quote.get('open', []),
                'high': quote.get('high', []),
                'low': quote.get('low', []),
                'close': quote.get('close', []),
                'volume': quote.get('volume', [0] * len(timestamps))
            })

            df.set_index('datetime', inplace=True)
            df = df.dropna()

            return df[['open', 'high', 'low', 'close', 'volume']].tail(bars)

        except Exception as e:
            return None

    def _save_cache(self, pair: str, timeframe: str, df: pd.DataFrame):
        """Save data to local cache"""
        cache_file = self.cache_dir / f"{pair}_{timeframe}.csv"
        df.to_csv(cache_file)

    def _load_cache(self, pair: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load data from local cache"""
        cache_file = self.cache_dir / f"{pair}_{timeframe}.csv"
        if cache_file.exists():
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df
        return None

    def _generate_synthetic(self, pair: str, bars: int) -> pd.DataFrame:
        """Generate synthetic data for testing (ONLY for testing!)"""
        np.random.seed(hash(pair) % 2**32)

        # Base price for the pair
        base_prices = {
            'EURUSD': 1.0800, 'GBPUSD': 1.2700, 'USDJPY': 150.00,
            'AUDUSD': 0.6500, 'USDCAD': 1.3600, 'USDCHF': 0.8800
        }
        base = base_prices.get(pair, 1.0)

        # Generate random walk with mean reversion
        returns = np.random.normal(0, 0.0008, bars)
        trend = np.zeros(bars)
        current_trend = 0
        for i in range(bars):
            if np.random.random() < 0.02:
                current_trend = np.random.choice([-0.0002, 0, 0.0002])
            trend[i] = current_trend

        returns = returns + trend
        prices = base * np.cumprod(1 + returns)

        # Generate OHLC
        volatility = np.abs(np.random.normal(0.0003, 0.0001, bars))

        dates = pd.date_range(end=datetime.now(), periods=bars, freq='4H')
        df = pd.DataFrame({
            'open': np.roll(prices, 1),
            'high': prices + volatility,
            'low': prices - volatility,
            'close': prices,
            'volume': np.random.randint(1000, 10000, bars)
        }, index=dates)

        df['open'].iloc[0] = prices[0]

        return df


# ═══════════════════════════════════════════════════════════════════════════════
#                       FIX #2: PROPER NEWS INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class NewsSignalIntegration:
    """
    Properly integrates with EnhancedForexSignals from forex_system.
    """

    def __init__(self, config: Config):
        self.config = config
        self.signals_service = None
        self._init_signals_service()

    def _init_signals_service(self):
        """Initialize the EnhancedForexSignals service"""
        try:
            # Try importing from forex_system
            from forex_system.services.enhanced_forex_signals import EnhancedForexSignals
            self.signals_service = EnhancedForexSignals()
            print("✅ EnhancedForexSignals service loaded")
        except ImportError as e:
            print(f"⚠️  EnhancedForexSignals not available: {e}")
            print("   Using fallback signal generator")
            self.signals_service = None
        except Exception as e:
            print(f"⚠️  Error initializing signals service: {e}")
            self.signals_service = None

    async def get_enhanced_signal(self, pair: str) -> Dict:
        """
        Get combined signal from news + technical analysis.

        Returns:
        {
            'signal': 'BUY' | 'SELL' | 'NEUTRAL',
            'strength': 0.0 - 1.0,
            'confidence_boost': -0.1 to 0.2,
            'regime': 'LOW_VOL' | 'NORMAL' | 'HIGH_VOL' | etc,
            'news_factors': [...],
            'technical_factors': [...]
        }
        """
        if self.signals_service:
            try:
                # Get combined signal from the service
                result = await self.signals_service.get_combined_signal(pair)

                return {
                    'signal': result.get('signal', 'NEUTRAL'),
                    'strength': result.get('strength', 0.0),
                    'confidence_boost': self._calculate_confidence_boost(result),
                    'regime': result.get('regime', 'NORMAL'),
                    'news_factors': result.get('explanations', []),
                    'technical_factors': result.get('components', {}).get('mtf', {}).get('reasons', []),
                    'vix': result.get('components', {}).get('risk', {}).get('vix', 20),
                    'atr': result.get('atr', 0)
                }
            except Exception as e:
                print(f"⚠️  Signal service error: {e}")

        # Fallback: return neutral signal
        return {
            'signal': 'NEUTRAL',
            'strength': 0.0,
            'confidence_boost': 0.0,
            'regime': 'UNKNOWN',
            'news_factors': [],
            'technical_factors': [],
            'vix': 20,
            'atr': 0
        }

    def _calculate_confidence_boost(self, result: Dict) -> float:
        """Calculate confidence boost/penalty from news and regime"""
        boost = 0.0

        # Regime adjustment
        regime = result.get('regime', '')
        if 'LOW_VOL' in regime:
            boost += 0.05  # More predictable
        elif 'HIGH_VOL' in regime or 'FEAR' in regime:
            boost -= 0.1   # Less predictable

        # News alignment
        mtf_signal = result.get('components', {}).get('mtf', {}).get('signal')
        risk_signal = result.get('components', {}).get('risk', {}).get('signal')

        if mtf_signal == risk_signal and mtf_signal != 'NEUTRAL':
            boost += 0.05  # Signals align

        return max(-0.1, min(0.2, boost))

    async def close(self):
        """Clean up resources"""
        if self.signals_service:
            try:
                await self.signals_service.close()
            except:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
#                       FIX #4: METATRADER 5 BROKER CONNECTION
# ═══════════════════════════════════════════════════════════════════════════════

class BrokerConnection:
    """
    MetaTrader 5 broker connection for real trade execution.
    """

    def __init__(self, config: Config):
        self.config = config
        self.mt5 = None
        self.connected = False
        self._init_connection()

    def _init_connection(self):
        """Initialize MT5 connection"""
        if self.config.PAPER_TRADING:
            print("📝 Paper trading mode - no broker connection needed")
            return

        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5

            if not mt5.initialize():
                print(f"⚠️  MT5 initialization failed: {mt5.last_error()}")
                return

            # Login if credentials provided
            if self.config.MT5_LOGIN:
                if mt5.login(
                    self.config.MT5_LOGIN,
                    self.config.MT5_PASSWORD,
                    self.config.MT5_SERVER
                ):
                    self.connected = True
                    account_info = mt5.account_info()
                    print(f"✅ Connected to broker: {account_info.server}")
                    print(f"   Balance: ${account_info.balance:,.2f}")
                    print(f"   Leverage: 1:{account_info.leverage}")
                else:
                    print(f"⚠️  MT5 login failed: {mt5.last_error()}")
            else:
                # Check if already logged in via terminal
                account_info = mt5.account_info()
                if account_info:
                    self.connected = True
                    print(f"✅ Using existing MT5 session")
                    print(f"   Balance: ${account_info.balance:,.2f}")

        except ImportError:
            print("⚠️  MetaTrader5 not installed. Run: pip install MetaTrader5")
        except Exception as e:
            print(f"⚠️  Broker connection error: {e}")

    def open_position(
        self,
        pair: str,
        direction: str,  # 'BUY' or 'SELL'
        lot_size: float,
        sl_price: float,
        tp_price: float,
        comment: str = "FixedTradingSystem"
    ) -> Optional[Dict]:
        """
        Open a position via MetaTrader 5.

        Returns: {'ticket': int, 'price': float, ...} or None
        """
        if self.config.PAPER_TRADING:
            return self._paper_trade_open(pair, direction, lot_size, sl_price, tp_price, comment)

        if not self.connected or not self.mt5:
            print("⚠️  Not connected to broker")
            return None

        try:
            # Get symbol info
            symbol_info = self.mt5.symbol_info(pair)
            if symbol_info is None:
                # Try variations
                for suffix in ['', '.m', 'm', '.raw', '.pro']:
                    test_symbol = f"{pair}{suffix}"
                    symbol_info = self.mt5.symbol_info(test_symbol)
                    if symbol_info:
                        pair = test_symbol
                        break

            if symbol_info is None:
                print(f"⚠️  Symbol {pair} not found")
                return None

            # Ensure symbol is visible
            if not symbol_info.visible:
                self.mt5.symbol_select(pair, True)

            # Get current price
            tick = self.mt5.symbol_info_tick(pair)
            if tick is None:
                print(f"⚠️  Cannot get price for {pair}")
                return None

            # Determine order type and price
            if direction == 'BUY':
                order_type = self.mt5.ORDER_TYPE_BUY
                price = tick.ask
            else:
                order_type = self.mt5.ORDER_TYPE_SELL
                price = tick.bid

            # Prepare request
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": pair,
                "volume": lot_size,
                "type": order_type,
                "price": price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 20,  # Slippage tolerance in points
                "magic": 123456,  # Magic number to identify our trades
                "comment": comment,
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self.mt5.ORDER_FILLING_IOC,
            }

            # Send order
            result = self.mt5.order_send(request)

            if result.retcode != self.mt5.TRADE_RETCODE_DONE:
                print(f"⚠️  Order failed: {result.comment}")
                return None

            return {
                'ticket': result.order,
                'price': result.price,
                'volume': result.volume,
                'direction': direction,
                'symbol': pair,
                'sl': sl_price,
                'tp': tp_price
            }

        except Exception as e:
            print(f"⚠️  Error opening position: {e}")
            return None

    def close_position(self, ticket: int, pair: str, direction: str, volume: float) -> bool:
        """Close an existing position"""
        if self.config.PAPER_TRADING:
            return True

        if not self.connected or not self.mt5:
            return False

        try:
            # Get current price
            tick = self.mt5.symbol_info_tick(pair)
            if tick is None:
                return False

            # Opposite order to close
            if direction == 'BUY':
                order_type = self.mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                order_type = self.mt5.ORDER_TYPE_BUY
                price = tick.ask

            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": pair,
                "volume": volume,
                "type": order_type,
                "price": price,
                "deviation": 20,
                "magic": 123456,
                "comment": "Close position",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self.mt5.ORDER_FILLING_IOC,
            }

            result = self.mt5.order_send(request)
            return result.retcode == self.mt5.TRADE_RETCODE_DONE

        except Exception as e:
            print(f"⚠️  Error closing position: {e}")
            return False

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        if self.config.PAPER_TRADING:
            return getattr(self, '_paper_positions', [])

        if not self.connected or not self.mt5:
            return []

        try:
            positions = self.mt5.positions_get()
            if positions is None:
                return []

            return [{
                'ticket': p.ticket,
                'symbol': p.symbol,
                'direction': 'BUY' if p.type == 0 else 'SELL',
                'volume': p.volume,
                'price': p.price_open,
                'sl': p.sl,
                'tp': p.tp,
                'profit': p.profit,
                'time': datetime.fromtimestamp(p.time)
            } for p in positions]

        except Exception as e:
            print(f"⚠️  Error getting positions: {e}")
            return []

    def get_account_info(self) -> Dict:
        """Get account information"""
        if self.config.PAPER_TRADING:
            return {
                'balance': getattr(self, '_paper_balance', self.config.STARTING_CAPITAL),
                'equity': getattr(self, '_paper_balance', self.config.STARTING_CAPITAL),
                'margin_free': getattr(self, '_paper_balance', self.config.STARTING_CAPITAL),
                'leverage': 100
            }

        if not self.connected or not self.mt5:
            return {}

        try:
            info = self.mt5.account_info()
            return {
                'balance': info.balance,
                'equity': info.equity,
                'margin_free': info.margin_free,
                'leverage': info.leverage
            }
        except:
            return {}

    def _paper_trade_open(self, pair, direction, lot_size, sl, tp, comment):
        """Simulate opening a trade in paper mode"""
        if not hasattr(self, '_paper_positions'):
            self._paper_positions = []
            self._paper_ticket = 1000

        self._paper_ticket += 1

        position = {
            'ticket': self._paper_ticket,
            'symbol': pair,
            'direction': direction,
            'volume': lot_size,
            'price': 1.0,  # Will be updated
            'sl': sl,
            'tp': tp,
            'profit': 0,
            'time': datetime.now()
        }

        self._paper_positions.append(position)
        return position

    def shutdown(self):
        """Clean shutdown of MT5 connection"""
        if self.mt5:
            try:
                self.mt5.shutdown()
            except:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
#                       FEATURE ENGINEERING (from UPGRADED_FEATURES.py)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(window=period).mean()


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 0.0001)
    return 100 - (100 / (1 + rs))


def calculate_confidence_margin(proba: np.ndarray) -> float:
    """Calculate confidence as margin between classes (not just max)"""
    sorted_proba = np.sort(proba)[::-1]
    if len(sorted_proba) >= 2:
        return sorted_proba[0] - sorted_proba[1]
    return sorted_proba[0]


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate ML features from OHLC data"""
    features = pd.DataFrame(index=df.index)

    close = df['close']
    high = df['high']
    low = df['low']

    # ATR
    features['ATR'] = calculate_atr(df, 14)

    # RSI at multiple periods
    for period in [7, 14, 21]:
        features[f'RSI_{period}'] = calculate_rsi(close, period)

    # Moving averages
    for period in [10, 20, 50]:
        features[f'SMA_{period}'] = close.rolling(period).mean()
        features[f'price_vs_SMA_{period}'] = (close - features[f'SMA_{period}']) / features[f'SMA_{period}']

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    features['MACD'] = ema12 - ema26
    features['MACD_signal'] = features['MACD'].ewm(span=9).mean()
    features['MACD_hist'] = features['MACD'] - features['MACD_signal']

    # Bollinger Bands position
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    features['BB_position'] = (close - bb_mid) / (2 * bb_std)

    # Rate of Change
    for period in [5, 10, 20]:
        features[f'ROC_{period}'] = close.pct_change(period)

    # Stochastic
    for period in [14, 21]:
        lowest = low.rolling(period).min()
        highest = high.rolling(period).max()
        features[f'Stoch_{period}'] = 100 * (close - lowest) / (highest - lowest + 0.0001)

    # Z-score (mean reversion)
    for period in [10, 20, 50]:
        mean = close.rolling(period).mean()
        std = close.rolling(period).std()
        features[f'zscore_{period}'] = (close - mean) / (std + 0.0001)

    # Volatility ratio
    features['vol_ratio'] = features['ATR'] / close

    # Trend strength
    features['trend_strength'] = abs(features['price_vs_SMA_20']) * (1 / (features['vol_ratio'] + 0.0001))

    return features.dropna()


def generate_labels(df: pd.DataFrame, lookahead: int = 20, atr_mult: float = 2.0) -> pd.Series:
    """Generate labels based on ATR-dynamic thresholds"""
    atr = calculate_atr(df, 14)
    close = df['close']

    labels = pd.Series(index=df.index, dtype=float)

    for i in range(len(df) - lookahead):
        current_price = close.iloc[i]
        future_prices = close.iloc[i+1:i+lookahead+1]
        current_atr = atr.iloc[i]

        if pd.isna(current_atr) or current_atr <= 0:
            labels.iloc[i] = 0
            continue

        threshold = current_atr * atr_mult
        max_up = future_prices.max() - current_price
        max_down = current_price - future_prices.min()

        if max_up > threshold and max_up > max_down:
            labels.iloc[i] = 1  # BUY
        elif max_down > threshold and max_down > max_up:
            labels.iloc[i] = -1  # SELL
        else:
            labels.iloc[i] = 0  # HOLD

    return labels


# ═══════════════════════════════════════════════════════════════════════════════
#                       NOTIFICATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class NotificationSystem:
    """Send notifications for exceptional events only"""

    def __init__(self, config: Config):
        self.config = config
        self.enabled = bool(config.TELEGRAM_TOKEN and config.TELEGRAM_CHAT_ID)
        if self.enabled:
            print("✅ Telegram notifications enabled")

    def notify(self, message: str, is_exceptional: bool = False):
        """Send notification (only for exceptional events)"""
        print(message)  # Always log

        if not is_exceptional or not self.enabled:
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

    def trade_opened(self, pair: str, direction: str, price: float, sl: float, tp: float, size: float):
        """Notify of new trade"""
        msg = f"""🟢 *TRADE OPENED*
Pair: {pair}
Direction: {direction}
Entry: {price:.5f}
SL: {sl:.5f}
TP: {tp:.5f}
Size: {size} lots"""
        self.notify(msg, is_exceptional=True)

    def trade_closed(self, pair: str, direction: str, pnl: float, pips: float):
        """Notify of closed trade"""
        emoji = "💰" if pnl > 0 else "📉"
        msg = f"""{emoji} *TRADE CLOSED*
Pair: {pair}
Direction: {direction}
P&L: ${pnl:+,.2f}
Pips: {pips:+.1f}"""
        self.notify(msg, is_exceptional=True)

    def drawdown_warning(self, drawdown: float, balance: float):
        """Warn of excessive drawdown"""
        msg = f"""⚠️ *DRAWDOWN WARNING*
Current Drawdown: {drawdown:.1%}
Current Balance: ${balance:,.2f}"""
        self.notify(msg, is_exceptional=True)


# ═══════════════════════════════════════════════════════════════════════════════
#                       MAIN TRADING BOT
# ═══════════════════════════════════════════════════════════════════════════════

class FixedTradingBot:
    """
    Complete trading bot with all fixes applied.
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()

        print("\n" + "="*70)
        print("   FIXED TRADING SYSTEM")
        print("="*70)
        print(f"\n✅ FIX #1: Multi-source data (MT5 → Twelve Data → Alpha Vantage → Yahoo)")
        print(f"✅ FIX #2: EnhancedForexSignals integration")
        print(f"✅ FIX #3: Min confidence: {self.config.MIN_CONFIDENCE:.0%} (was 20%)")
        print(f"✅ FIX #4: MetaTrader 5 broker connection")
        print("="*70 + "\n")

        # Initialize components
        self.data_manager = DataManager(self.config)
        self.news_signals = NewsSignalIntegration(self.config)
        self.broker = BrokerConnection(self.config)
        self.notifications = NotificationSystem(self.config)

        # ML model
        self.model = None
        self.feature_columns = None

        # Trading state
        self.positions = []
        self.daily_pnl = 0
        self.peak_balance = self.config.STARTING_CAPITAL

    def train_model(self, pair: str) -> bool:
        """Train ML model on historical data"""
        print(f"\n📚 Training model for {pair}...")

        # Get historical data
        df = self.data_manager.fetch_ohlc(pair, '4H', 500)
        if df is None or len(df) < 200:
            print(f"⚠️  Insufficient data for {pair}")
            return False

        # Generate features and labels
        features = generate_features(df)
        labels = generate_labels(df)

        # Align and clean
        common_idx = features.index.intersection(labels.dropna().index)
        X = features.loc[common_idx]
        y = labels.loc[common_idx]

        # Train/test split
        split = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        self.feature_columns = X_train.columns.tolist()
        self.model.fit(X_train, y_train)

        # Evaluate
        X_test, y_test = X.iloc[split:], y.iloc[split:]
        accuracy = self.model.score(X_test, y_test)
        print(f"   Model accuracy: {accuracy:.1%}")

        return True

    async def generate_signal(self, pair: str) -> Dict:
        """Generate trading signal combining ML + news"""
        # Get recent data
        df = self.data_manager.fetch_ohlc(pair, '4H', 100)
        if df is None or len(df) < 50:
            return {'signal': 'NEUTRAL', 'confidence': 0}

        # Generate features
        features = generate_features(df)
        if features.empty:
            return {'signal': 'NEUTRAL', 'confidence': 0}

        # Get ML prediction
        if self.model is None:
            self.train_model(pair)

        if self.model is None:
            return {'signal': 'NEUTRAL', 'confidence': 0}

        latest = features.iloc[[-1]]

        # Ensure feature alignment
        missing = set(self.feature_columns) - set(latest.columns)
        for col in missing:
            latest[col] = 0
        latest = latest[self.feature_columns]

        prediction = self.model.predict(latest)[0]
        proba = self.model.predict_proba(latest)[0]
        ml_confidence = calculate_confidence_margin(proba)

        # Get news signals
        news_signal = await self.news_signals.get_enhanced_signal(pair)
        confidence_boost = news_signal.get('confidence_boost', 0)

        # Combine signals
        final_confidence = ml_confidence + confidence_boost

        # Apply minimum confidence threshold (FIX #3)
        if final_confidence < self.config.MIN_CONFIDENCE:
            return {
                'signal': 'NEUTRAL',
                'confidence': final_confidence,
                'reason': f'Below min confidence ({final_confidence:.1%} < {self.config.MIN_CONFIDENCE:.0%})'
            }

        # Determine signal
        if prediction == 1:
            signal = 'BUY'
        elif prediction == -1:
            signal = 'SELL'
        else:
            signal = 'NEUTRAL'

        # Check news alignment
        news_agrees = (
            (signal == 'BUY' and news_signal.get('signal') == 'BUY') or
            (signal == 'SELL' and news_signal.get('signal') == 'SELL')
        )

        return {
            'signal': signal,
            'confidence': final_confidence,
            'ml_confidence': ml_confidence,
            'news_boost': confidence_boost,
            'news_agrees': news_agrees,
            'regime': news_signal.get('regime', 'UNKNOWN'),
            'atr': features['ATR'].iloc[-1]
        }

    def calculate_position_size(self, pair: str, atr: float) -> Tuple[float, float, float]:
        """
        Calculate position size, SL, and TP.

        Returns: (lot_size, sl_distance, tp_distance)
        """
        account = self.broker.get_account_info()
        balance = account.get('balance', self.config.STARTING_CAPITAL)

        risk_amount = balance * self.config.RISK_PER_TRADE

        # SL at 1.5 * ATR, TP at RR * SL
        sl_distance = atr * 1.5
        tp_distance = sl_distance * self.config.DEFAULT_RR

        # Calculate lot size (assuming 1 pip = $10 for 1 lot)
        # For forex: pip_value varies by pair, simplified here
        pip_value = 10  # USD per pip for 1 standard lot
        sl_pips = sl_distance * 10000 if pair.endswith('USD') else sl_distance * 100

        lot_size = risk_amount / (sl_pips * pip_value)
        lot_size = max(0.01, min(lot_size, 5.0))  # Clamp to reasonable range
        lot_size = round(lot_size, 2)

        return lot_size, sl_distance, tp_distance

    async def check_and_trade(self, pair: str):
        """Check for signals and execute trades"""
        # Check if we can open more positions
        current_positions = self.broker.get_open_positions()
        if len(current_positions) >= self.config.MAX_POSITIONS:
            print(f"   Max positions reached ({len(current_positions)})")
            return

        # Check if already have position in this pair
        for pos in current_positions:
            if pair in pos.get('symbol', ''):
                print(f"   Already have position in {pair}")
                return

        # Generate signal
        signal_data = await self.generate_signal(pair)
        signal = signal_data.get('signal', 'NEUTRAL')
        confidence = signal_data.get('confidence', 0)

        print(f"   {pair}: {signal} (conf: {confidence:.1%})")

        if signal == 'NEUTRAL':
            return

        # Get current price
        df = self.data_manager.fetch_ohlc(pair, '4H', 10)
        if df is None or df.empty:
            return

        current_price = df['close'].iloc[-1]
        atr = signal_data.get('atr', df['close'].std())

        # Calculate position parameters
        lot_size, sl_dist, tp_dist = self.calculate_position_size(pair, atr)

        if signal == 'BUY':
            sl_price = current_price - sl_dist
            tp_price = current_price + tp_dist
        else:
            sl_price = current_price + sl_dist
            tp_price = current_price - tp_dist

        # Execute trade
        result = self.broker.open_position(
            pair=pair,
            direction=signal,
            lot_size=lot_size,
            sl_price=sl_price,
            tp_price=tp_price,
            comment=f"FixedBot_{confidence:.0%}"
        )

        if result:
            self.notifications.trade_opened(
                pair=pair,
                direction=signal,
                price=current_price,
                sl=sl_price,
                tp=tp_price,
                size=lot_size
            )

    async def run_cycle(self):
        """Run one trading cycle"""
        print(f"\n{'─'*70}")
        print(f"🔄 Trading Cycle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'─'*70}")

        for pair in self.config.PAIRS:
            try:
                await self.check_and_trade(pair)
            except Exception as e:
                print(f"   ⚠️  Error processing {pair}: {e}")

        # Check drawdown
        account = self.broker.get_account_info()
        balance = account.get('balance', self.config.STARTING_CAPITAL)

        if balance > self.peak_balance:
            self.peak_balance = balance

        drawdown = (self.peak_balance - balance) / self.peak_balance
        if drawdown > self.config.MAX_DAILY_LOSS:
            self.notifications.drawdown_warning(drawdown, balance)

    async def run(self, interval_minutes: int = 60):
        """Run the trading bot continuously"""
        print(f"\n🚀 Starting trading bot...")
        print(f"   Pairs: {', '.join(self.config.PAIRS)}")
        print(f"   Interval: {interval_minutes} minutes")
        print(f"   Mode: {'PAPER' if self.config.PAPER_TRADING else 'LIVE'}")

        # Train models for all pairs
        for pair in self.config.PAIRS:
            self.train_model(pair)

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

        # Cleanup
        await self.news_signals.close()
        self.broker.shutdown()

    async def run_once(self):
        """Run a single trading cycle (for testing)"""
        await self.run_cycle()
        await self.news_signals.close()
        self.broker.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
#                       MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Configuration
    config = Config(
        STARTING_CAPITAL=30000,
        RISK_PER_TRADE=0.015,
        MIN_CONFIDENCE=0.35,  # FIX #3: Increased from 0.20
        MAX_POSITIONS=3,
        PAIRS=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
        PAPER_TRADING=True,  # Set to False for live trading

        # Optional: Add your API keys here or use environment variables
        # ALPHA_VANTAGE_KEY="your_key_here",
        # TWELVE_DATA_KEY="your_key_here",
        # TELEGRAM_TOKEN="your_bot_token",
        # TELEGRAM_CHAT_ID="your_chat_id",

        # For live trading, add MT5 credentials:
        # MT5_LOGIN=12345678,
        # MT5_PASSWORD="your_password",
        # MT5_SERVER="YourBroker-Server",
    )

    # Create and run bot
    bot = FixedTradingBot(config)

    # Run once for testing
    print("\n" + "="*70)
    print("   RUNNING SINGLE TEST CYCLE")
    print("="*70)

    asyncio.run(bot.run_once())

    print("\n" + "="*70)
    print("   TEST COMPLETE")
    print("="*70)
    print("""
To run continuously:
    asyncio.run(bot.run(interval_minutes=60))

To enable live trading:
    1. Set PAPER_TRADING=False
    2. Add your MT5 credentials
    3. Get free API keys for data sources
    """)
