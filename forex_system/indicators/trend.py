"""
Trend indicators for technical analysis.

Includes: SMA, EMA, MACD, ADX
"""
import pandas as pd
import ta

from .base import BaseIndicator


class SMA(BaseIndicator):
    """Simple Moving Average (SMA)."""

    def __init__(self, period: int = 20):
        """
        Initialize SMA.

        Args:
            period: Number of periods for moving average
        """
        super().__init__(f"SMA({period})")
        self.validate_period(period, min_value=1)
        self.period = period

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Simple Moving Average.

        Args:
            df: DataFrame with 'close' column

        Returns:
            Series with SMA values
        """
        self.validate_data(df, min_periods=self.period, required_columns=['close'])
        return ta.trend.sma_indicator(df['close'], window=self.period)


class EMA(BaseIndicator):
    """Exponential Moving Average (EMA)."""

    def __init__(self, period: int = 20):
        """
        Initialize EMA.

        Args:
            period: Number of periods for exponential moving average
        """
        super().__init__(f"EMA({period})")
        self.validate_period(period, min_value=1)
        self.period = period

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Exponential Moving Average.

        Args:
            df: DataFrame with 'close' column

        Returns:
            Series with EMA values
        """
        self.validate_data(df, min_periods=self.period, required_columns=['close'])
        return ta.trend.ema_indicator(df['close'], window=self.period)


class MACD(BaseIndicator):
    """Moving Average Convergence Divergence (MACD)."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """
        Initialize MACD.

        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
        """
        super().__init__(f"MACD({fast},{slow},{signal})")
        self.validate_period(fast, min_value=1)
        self.validate_period(slow, min_value=1)
        self.validate_period(signal, min_value=1)

        if fast >= slow:
            raise ValueError("Fast period must be less than slow period")

        self.fast = fast
        self.slow = slow
        self.signal = signal

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD indicator.

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with columns ['macd', 'macd_signal', 'macd_diff']
        """
        min_periods = self.slow + self.signal
        self.validate_data(df, min_periods=min_periods, required_columns=['close'])

        macd = ta.trend.macd(df['close'], window_slow=self.slow, window_fast=self.fast)
        signal = ta.trend.macd_signal(
            df['close'],
            window_slow=self.slow,
            window_fast=self.fast,
            window_sign=self.signal
        )
        diff = ta.trend.macd_diff(
            df['close'],
            window_slow=self.slow,
            window_fast=self.fast,
            window_sign=self.signal
        )

        return pd.DataFrame({
            'macd': macd,
            'macd_signal': signal,
            'macd_diff': diff
        })


class ADX(BaseIndicator):
    """Average Directional Index (ADX)."""

    def __init__(self, period: int = 14):
        """
        Initialize ADX.

        Args:
            period: Number of periods for ADX calculation
        """
        super().__init__(f"ADX({period})")
        self.validate_period(period, min_value=1)
        self.period = period

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ADX indicator.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns

        Returns:
            DataFrame with columns ['adx', 'adx_pos', 'adx_neg']
        """
        min_periods = self.period * 2
        self.validate_data(df, min_periods=min_periods, required_columns=['high', 'low', 'close'])

        adx = ta.trend.adx(df['high'], df['low'], df['close'], window=self.period)
        adx_pos = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=self.period)
        adx_neg = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=self.period)

        return pd.DataFrame({
            'adx': adx,
            'adx_pos': adx_pos,
            'adx_neg': adx_neg
        })
