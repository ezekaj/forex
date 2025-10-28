"""
Data acquisition service for historical forex data.

Generates realistic synthetic OHLCV data for development and backtesting.
Can be extended to use real data sources (Alpha Vantage, Polygon.io, etc.)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, Float, DateTime, Index
from sqlalchemy.exc import IntegrityError

from ..models import Base


class HistoricalData(Base):
    """Historical OHLCV data model."""

    __tablename__ = "historical_data"

    id = Column(Integer, primary_key=True, index=True)
    pair = Column(String(10), nullable=False, index=True)
    timeframe = Column(String(5), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_pair_timeframe_timestamp', 'pair', 'timeframe', 'timestamp', unique=True),
    )


class DataService:
    """Service for generating and caching historical forex data."""

    # Base prices for forex pairs (realistic mid-rates)
    BASE_PRICES = {
        'EURUSD': 1.0850,
        'GBPUSD': 1.2650,
        'USDJPY': 149.50,
        'AUDUSD': 0.6450,
        'USDCAD': 1.3850,
        'USDCHF': 0.8950,
        'NZDUSD': 0.5950,
        'EURGBP': 0.8580,
        'EURJPY': 162.25,
        'GBPJPY': 189.15,
    }

    # Volatility parameters (daily volatility %)
    VOLATILITY = {
        'EURUSD': 0.006,   # 0.6% daily
        'GBPUSD': 0.008,   # 0.8% daily
        'USDJPY': 0.007,
        'AUDUSD': 0.009,
        'USDCAD': 0.006,
        'USDCHF': 0.007,
        'NZDUSD': 0.010,
        'EURGBP': 0.005,
        'EURJPY': 0.008,
        'GBPJPY': 0.010,
    }

    # Timeframe to minutes mapping
    TIMEFRAME_MINUTES = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '1h': 60,
        '4h': 240,
        '1d': 1440,
    }

    def __init__(self, session: Optional[Session] = None, use_cache: bool = True):
        """
        Initialize DataService.

        Args:
            session: SQLAlchemy session for database operations
            use_cache: Whether to use database caching
        """
        self.session = session
        self.use_cache = use_cache

    def get_historical_data(
        self,
        pair: str,
        timeframe: str = '1h',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a currency pair.

        Args:
            pair: Currency pair (e.g., 'EURUSD')
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            start_date: Start date for data (default: 30 days ago)
            end_date: End date for data (default: now)
            force_refresh: If True, regenerate data

        Returns:
            DataFrame with OHLCV data

        Raises:
            ValueError: If pair or timeframe is invalid
        """
        # Validate inputs
        if pair not in self.BASE_PRICES:
            raise ValueError(f"Invalid pair: {pair}. Supported: {list(self.BASE_PRICES.keys())}")

        if timeframe not in self.TIMEFRAME_MINUTES:
            raise ValueError(f"Invalid timeframe: {timeframe}. Supported: {list(self.TIMEFRAME_MINUTES.keys())}")

        # Set default date range
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        # Try to get from cache first
        if self.session and self.use_cache and not force_refresh:
            cached_data = self._get_from_cache(pair, timeframe, start_date, end_date)
            if cached_data is not None and len(cached_data) > 0:
                return cached_data

        # Generate synthetic data
        df = self._generate_synthetic_data(pair, timeframe, start_date, end_date)

        # Cache in database if session provided
        if self.session and self.use_cache and df is not None and len(df) > 0:
            self._save_to_cache(pair, timeframe, df)

        return df

    def _generate_synthetic_data(
        self,
        pair: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Generate realistic synthetic OHLCV data.

        Args:
            pair: Currency pair
            timeframe: Timeframe
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        # Calculate number of bars needed
        minutes_per_bar = self.TIMEFRAME_MINUTES[timeframe]
        total_minutes = int((end_date - start_date).total_seconds() / 60)
        num_bars = total_minutes // minutes_per_bar

        if num_bars <= 0:
            return pd.DataFrame()

        # Generate timestamps
        timestamps = [
            start_date + timedelta(minutes=i * minutes_per_bar)
            for i in range(num_bars)
        ]

        # Get base price and volatility
        base_price = self.BASE_PRICES[pair]
        daily_vol = self.VOLATILITY[pair]

        # Scale volatility to timeframe
        timeframe_vol = daily_vol * np.sqrt(minutes_per_bar / 1440)

        # Generate price movement using geometric Brownian motion
        np.random.seed(hash(pair + timeframe + str(start_date.date())) % (2**32))
        returns = np.random.normal(0, timeframe_vol, num_bars)
        cumulative_returns = np.cumsum(returns)
        close_prices = base_price * (1 + cumulative_returns)

        # Generate OHLCV data
        data = []
        for i, (timestamp, close) in enumerate(zip(timestamps, close_prices)):
            # Calculate open (previous close or base)
            open_price = close_prices[i-1] if i > 0 else base_price

            # Generate high/low with realistic ranges
            bar_range = abs(np.random.normal(0, timeframe_vol * 0.5)) * open_price
            high = max(open_price, close) + bar_range
            low = min(open_price, close) - bar_range

            # Generate volume (random but realistic)
            base_volume = 10000 if 'JPY' in pair else 100000
            volume = base_volume * (0.5 + np.random.random())

            data.append({
                'timestamp': timestamp,
                'open': round(open_price, 5 if 'JPY' not in pair else 3),
                'high': round(high, 5 if 'JPY' not in pair else 3),
                'low': round(low, 5 if 'JPY' not in pair else 3),
                'close': round(close, 5 if 'JPY' not in pair else 3),
                'volume': round(volume, 2)
            })

        df = pd.DataFrame(data)
        return df

    def _get_from_cache(
        self,
        pair: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Get data from database cache.

        Args:
            pair: Currency pair
            timeframe: Timeframe
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame or None if not found
        """
        try:
            query = self.session.query(HistoricalData).filter(
                HistoricalData.pair == pair,
                HistoricalData.timeframe == timeframe,
                HistoricalData.timestamp >= start_date,
                HistoricalData.timestamp <= end_date
            ).order_by(HistoricalData.timestamp)

            results = query.all()

            if not results:
                return None

            # Convert to DataFrame
            data = [{
                'timestamp': r.timestamp,
                'open': r.open,
                'high': r.high,
                'low': r.low,
                'close': r.close,
                'volume': r.volume
            } for r in results]

            df = pd.DataFrame(data)
            return df

        except Exception as e:
            print(f"Error reading from cache: {str(e)}")
            return None

    def _save_to_cache(self, pair: str, timeframe: str, df: pd.DataFrame) -> None:
        """
        Save data to database cache.

        Args:
            pair: Currency pair
            timeframe: Timeframe
            df: DataFrame with OHLCV data
        """
        try:
            for _, row in df.iterrows():
                data_point = HistoricalData(
                    pair=pair,
                    timeframe=timeframe,
                    timestamp=row['timestamp'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                )

                try:
                    self.session.add(data_point)
                    self.session.commit()
                except IntegrityError:
                    # Data already exists, skip
                    self.session.rollback()
                    continue

        except Exception as e:
            print(f"Error saving to cache: {str(e)}")
            self.session.rollback()

    def get_multiple_pairs(
        self,
        pairs: List[str],
        timeframe: str = '1h',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple pairs.

        Args:
            pairs: List of currency pairs
            timeframe: Timeframe
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary mapping pair to DataFrame
        """
        results = {}

        for pair in pairs:
            try:
                df = self.get_historical_data(pair, timeframe, start_date, end_date)
                if not df.empty:
                    results[pair] = df
            except Exception as e:
                print(f"Error fetching {pair}: {str(e)}")
                continue

        return results

    def clear_cache(self, pair: Optional[str] = None, timeframe: Optional[str] = None) -> int:
        """
        Clear cached data.

        Args:
            pair: Optional pair to clear (clears all if None)
            timeframe: Optional timeframe to clear (clears all if None)

        Returns:
            Number of records deleted
        """
        if not self.session:
            return 0

        try:
            query = self.session.query(HistoricalData)

            if pair:
                query = query.filter(HistoricalData.pair == pair)
            if timeframe:
                query = query.filter(HistoricalData.timeframe == timeframe)

            count = query.count()
            query.delete()
            self.session.commit()

            return count

        except Exception as e:
            print(f"Error clearing cache: {str(e)}")
            self.session.rollback()
            return 0
