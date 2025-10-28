"""
Volume indicators for technical analysis.

Includes: OBV, Volume SMA, MFI, VWAP
"""
import pandas as pd
import ta

from .base import BaseIndicator


class OBV(BaseIndicator):
    """On-Balance Volume (OBV)."""

    def __init__(self):
        """Initialize OBV."""
        super().__init__("OBV")

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume.

        Args:
            df: DataFrame with 'close' and 'volume' columns

        Returns:
            Series with OBV values
        """
        self.validate_data(df, min_periods=2, required_columns=['close', 'volume'])
        return ta.volume.on_balance_volume(df['close'], df['volume'])


class VolumeSMA(BaseIndicator):
    """Volume Simple Moving Average."""

    def __init__(self, period: int = 20):
        """
        Initialize Volume SMA.

        Args:
            period: Number of periods for moving average
        """
        super().__init__(f"VolumeSMA({period})")
        self.validate_period(period, min_value=1)
        self.period = period

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume SMA.

        Args:
            df: DataFrame with 'volume' column

        Returns:
            Series with volume moving average values
        """
        self.validate_data(df, min_periods=self.period, required_columns=['volume'])
        return ta.trend.sma_indicator(df['volume'], window=self.period)


class MFI(BaseIndicator):
    """Money Flow Index (MFI)."""

    def __init__(self, period: int = 14):
        """
        Initialize MFI.

        Args:
            period: Number of periods for MFI calculation
        """
        super().__init__(f"MFI({period})")
        self.validate_period(period, min_value=1)
        self.period = period

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Money Flow Index.

        Args:
            df: DataFrame with 'high', 'low', 'close', 'volume' columns

        Returns:
            Series with MFI values (0-100)
        """
        self.validate_data(
            df,
            min_periods=self.period,
            required_columns=['high', 'low', 'close', 'volume']
        )
        return ta.volume.money_flow_index(
            df['high'],
            df['low'],
            df['close'],
            df['volume'],
            window=self.period
        )


class VWAP(BaseIndicator):
    """Volume Weighted Average Price (VWAP)."""

    def __init__(self):
        """Initialize VWAP."""
        super().__init__("VWAP")

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price.

        Args:
            df: DataFrame with 'high', 'low', 'close', 'volume' columns

        Returns:
            Series with VWAP values
        """
        self.validate_data(
            df,
            min_periods=1,
            required_columns=['high', 'low', 'close', 'volume']
        )
        return ta.volume.volume_weighted_average_price(
            df['high'],
            df['low'],
            df['close'],
            df['volume']
        )


class ADI(BaseIndicator):
    """Accumulation/Distribution Index (ADI)."""

    def __init__(self):
        """Initialize ADI."""
        super().__init__("ADI")

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Accumulation/Distribution Index.

        Args:
            df: DataFrame with 'high', 'low', 'close', 'volume' columns

        Returns:
            Series with ADI values
        """
        self.validate_data(
            df,
            min_periods=1,
            required_columns=['high', 'low', 'close', 'volume']
        )
        return ta.volume.acc_dist_index(
            df['high'],
            df['low'],
            df['close'],
            df['volume']
        )


class CMF(BaseIndicator):
    """Chaikin Money Flow (CMF)."""

    def __init__(self, period: int = 20):
        """
        Initialize CMF.

        Args:
            period: Number of periods for CMF calculation
        """
        super().__init__(f"CMF({period})")
        self.validate_period(period, min_value=1)
        self.period = period

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Chaikin Money Flow.

        Args:
            df: DataFrame with 'high', 'low', 'close', 'volume' columns

        Returns:
            Series with CMF values
        """
        self.validate_data(
            df,
            min_periods=self.period,
            required_columns=['high', 'low', 'close', 'volume']
        )
        return ta.volume.chaikin_money_flow(
            df['high'],
            df['low'],
            df['close'],
            df['volume'],
            window=self.period
        )
