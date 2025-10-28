"""
Base class for all technical indicators.

Provides common interface and validation for indicator calculations.
"""
from abc import ABC, abstractmethod
from typing import Optional, Union
import pandas as pd
import numpy as np


class BaseIndicator(ABC):
    """Abstract base class for technical indicators."""

    def __init__(self, name: str):
        """
        Initialize indicator.

        Args:
            name: Human-readable indicator name
        """
        self.name = name

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate indicator from OHLCV dataframe.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            Series or DataFrame with indicator values

        Raises:
            ValueError: If data is invalid or insufficient
        """
        pass

    def validate_data(
        self,
        df: pd.DataFrame,
        min_periods: Optional[int] = None,
        required_columns: Optional[list] = None
    ) -> None:
        """
        Validate input dataframe has required data.

        Args:
            df: Input dataframe
            min_periods: Minimum number of rows required
            required_columns: List of required column names

        Raises:
            ValueError: If validation fails
        """
        if df is None or df.empty:
            raise ValueError(f"{self.name}: Input dataframe is empty")

        # Check required columns
        if required_columns is None:
            required_columns = ['open', 'high', 'low', 'close', 'volume']

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"{self.name}: Missing required columns: {missing_cols}"
            )

        # Check minimum data points
        if min_periods and len(df) < min_periods:
            raise ValueError(
                f"{self.name}: Insufficient data. Need {min_periods} rows, "
                f"got {len(df)}"
            )

        # Check for all NaN values in required columns
        for col in required_columns:
            if df[col].isna().all():
                raise ValueError(f"{self.name}: Column '{col}' contains only NaN values")

    def validate_period(self, period: int, min_value: int = 1) -> None:
        """
        Validate period parameter.

        Args:
            period: Period value to validate
            min_value: Minimum allowed period

        Raises:
            ValueError: If period is invalid
        """
        if not isinstance(period, int):
            raise ValueError(f"{self.name}: Period must be an integer")

        if period < min_value:
            raise ValueError(
                f"{self.name}: Period must be >= {min_value}, got {period}"
            )

    def handle_errors(self, func, *args, **kwargs) -> Union[pd.Series, pd.DataFrame, None]:
        """
        Execute function with error handling.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Function result or None on error
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error calculating {self.name}: {str(e)}")
            return None
