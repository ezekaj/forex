"""
Momentum indicators for technical analysis.

Includes: RSI, Stochastic, CCI, ROC
"""
import pandas as pd
import ta

from .base import BaseIndicator


class RSI(BaseIndicator):
    """Relative Strength Index (RSI)."""

    def __init__(self, period: int = 14):
        """
        Initialize RSI.

        Args:
            period: Number of periods for RSI calculation
        """
        super().__init__(f"RSI({period})")
        self.validate_period(period, min_value=1)
        self.period = period

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate RSI indicator.

        Args:
            df: DataFrame with 'close' column

        Returns:
            Series with RSI values (0-100)
        """
        self.validate_data(df, min_periods=self.period + 1, required_columns=['close'])
        return ta.momentum.rsi(df['close'], window=self.period)


class Stochastic(BaseIndicator):
    """Stochastic Oscillator."""

    def __init__(self, k_period: int = 14, d_period: int = 3):
        """
        Initialize Stochastic Oscillator.

        Args:
            k_period: Period for %K line
            d_period: Period for %D line (signal)
        """
        super().__init__(f"Stochastic({k_period},{d_period})")
        self.validate_period(k_period, min_value=1)
        self.validate_period(d_period, min_value=1)
        self.k_period = k_period
        self.d_period = d_period

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns

        Returns:
            DataFrame with columns ['stoch_k', 'stoch_d']
        """
        min_periods = self.k_period + self.d_period
        self.validate_data(df, min_periods=min_periods, required_columns=['high', 'low', 'close'])

        stoch_k = ta.momentum.stoch(
            df['high'],
            df['low'],
            df['close'],
            window=self.k_period,
            smooth_window=self.d_period
        )
        stoch_d = ta.momentum.stoch_signal(
            df['high'],
            df['low'],
            df['close'],
            window=self.k_period,
            smooth_window=self.d_period
        )

        return pd.DataFrame({
            'stoch_k': stoch_k,
            'stoch_d': stoch_d
        })


class CCI(BaseIndicator):
    """Commodity Channel Index (CCI)."""

    def __init__(self, period: int = 20):
        """
        Initialize CCI.

        Args:
            period: Number of periods for CCI calculation
        """
        super().__init__(f"CCI({period})")
        self.validate_period(period, min_value=1)
        self.period = period

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate CCI indicator.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns

        Returns:
            Series with CCI values
        """
        self.validate_data(df, min_periods=self.period, required_columns=['high', 'low', 'close'])
        return ta.trend.cci(df['high'], df['low'], df['close'], window=self.period)


class ROC(BaseIndicator):
    """Rate of Change (ROC)."""

    def __init__(self, period: int = 12):
        """
        Initialize ROC.

        Args:
            period: Number of periods for ROC calculation
        """
        super().__init__(f"ROC({period})")
        self.validate_period(period, min_value=1)
        self.period = period

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Rate of Change.

        Args:
            df: DataFrame with 'close' column

        Returns:
            Series with ROC values (percentage)
        """
        self.validate_data(df, min_periods=self.period + 1, required_columns=['close'])
        return ta.momentum.roc(df['close'], window=self.period)
