"""
Volatility indicators for technical analysis.

Includes: ATR, Bollinger Bands, Standard Deviation
"""
import pandas as pd
import ta

from .base import BaseIndicator


class ATR(BaseIndicator):
    """Average True Range (ATR)."""

    def __init__(self, period: int = 14):
        """
        Initialize ATR.

        Args:
            period: Number of periods for ATR calculation
        """
        super().__init__(f"ATR({period})")
        self.validate_period(period, min_value=1)
        self.period = period

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Average True Range.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns

        Returns:
            Series with ATR values
        """
        self.validate_data(df, min_periods=self.period, required_columns=['high', 'low', 'close'])
        return ta.volatility.average_true_range(
            df['high'],
            df['low'],
            df['close'],
            window=self.period
        )


class BollingerBands(BaseIndicator):
    """Bollinger Bands."""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands.

        Args:
            period: Number of periods for moving average
            std_dev: Number of standard deviations for bands
        """
        super().__init__(f"BB({period},{std_dev})")
        self.validate_period(period, min_value=2)

        if std_dev <= 0:
            raise ValueError("Standard deviation must be positive")

        self.period = period
        self.std_dev = std_dev

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with columns ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct']
        """
        self.validate_data(df, min_periods=self.period, required_columns=['close'])

        bb_upper = ta.volatility.bollinger_hband(
            df['close'],
            window=self.period,
            window_dev=self.std_dev
        )
        bb_middle = ta.volatility.bollinger_mavg(
            df['close'],
            window=self.period
        )
        bb_lower = ta.volatility.bollinger_lband(
            df['close'],
            window=self.period,
            window_dev=self.std_dev
        )
        bb_width = ta.volatility.bollinger_wband(
            df['close'],
            window=self.period,
            window_dev=self.std_dev
        )
        bb_pct = ta.volatility.bollinger_pband(
            df['close'],
            window=self.period,
            window_dev=self.std_dev
        )

        return pd.DataFrame({
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'bb_width': bb_width,
            'bb_pct': bb_pct
        })


class StandardDeviation(BaseIndicator):
    """Standard Deviation (Volatility measure)."""

    def __init__(self, period: int = 20):
        """
        Initialize Standard Deviation.

        Args:
            period: Number of periods for calculation
        """
        super().__init__(f"StdDev({period})")
        self.validate_period(period, min_value=2)
        self.period = period

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate rolling standard deviation.

        Args:
            df: DataFrame with 'close' column

        Returns:
            Series with standard deviation values
        """
        self.validate_data(df, min_periods=self.period, required_columns=['close'])
        return df['close'].rolling(window=self.period).std()


class KeltnerChannel(BaseIndicator):
    """Keltner Channel."""

    def __init__(self, period: int = 20, atr_period: int = 10, multiplier: float = 2.0):
        """
        Initialize Keltner Channel.

        Args:
            period: Period for EMA calculation
            atr_period: Period for ATR calculation
            multiplier: ATR multiplier for channel width
        """
        super().__init__(f"KC({period},{atr_period},{multiplier})")
        self.validate_period(period, min_value=1)
        self.validate_period(atr_period, min_value=1)

        if multiplier <= 0:
            raise ValueError("Multiplier must be positive")

        self.period = period
        self.atr_period = atr_period
        self.multiplier = multiplier

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Keltner Channel.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns

        Returns:
            DataFrame with columns ['kc_upper', 'kc_middle', 'kc_lower']
        """
        min_periods = max(self.period, self.atr_period)
        self.validate_data(df, min_periods=min_periods, required_columns=['high', 'low', 'close'])

        kc_upper = ta.volatility.keltner_channel_hband(
            df['high'],
            df['low'],
            df['close'],
            window=self.period,
            window_atr=self.atr_period
        )
        kc_middle = ta.volatility.keltner_channel_mband(
            df['high'],
            df['low'],
            df['close'],
            window=self.period,
            window_atr=self.atr_period
        )
        kc_lower = ta.volatility.keltner_channel_lband(
            df['high'],
            df['low'],
            df['close'],
            window=self.period,
            window_atr=self.atr_period
        )

        return pd.DataFrame({
            'kc_upper': kc_upper,
            'kc_middle': kc_middle,
            'kc_lower': kc_lower
        })
