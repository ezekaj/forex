"""
Base strategy interface for trading strategies.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np


class Signal(Enum):
    """Trading signals."""
    BUY = 1
    HOLD = 0
    SELL = -1


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, name: str):
        """
        Initialize strategy.

        Args:
            name: Strategy name
        """
        self.name = name
        self.is_trained = False
        self.metadata = {}

    @abstractmethod
    def train(self, features_df: pd.DataFrame, labels: pd.Series) -> Dict[str, Any]:
        """
        Train the strategy on historical data.

        Args:
            features_df: DataFrame with features
            labels: Series with labels (BUY/HOLD/SELL)

        Returns:
            Dictionary with training metrics
        """
        pass

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generate trading signals for given features.

        Args:
            features: DataFrame with features (single or multiple rows)

        Returns:
            Array of predictions (Signal.BUY/HOLD/SELL values)
        """
        pass

    @abstractmethod
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities for each class.

        Args:
            features: DataFrame with features

        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass

    def generate_labels(
        self,
        df: pd.DataFrame,
        lookahead_bars: int = 5,
        buy_threshold: float = 0.003,
        sell_threshold: float = -0.003,
        binary: bool = False
    ) -> pd.Series:
        """
        Generate BUY/HOLD/SELL labels from price data.

        Labels are based on future returns:
        - BUY: If future return > buy_threshold
        - SELL: If future return < sell_threshold
        - HOLD: Otherwise (only if binary=False)

        Args:
            df: DataFrame with 'close' column
            lookahead_bars: Number of bars to look ahead
            buy_threshold: Threshold for BUY signal (e.g., 0.003 = 0.3%)
            sell_threshold: Threshold for SELL signal (e.g., -0.003 = -0.3%)
            binary: If True, only generate BUY/SELL (skip HOLD, use NaN instead)

        Returns:
            Series with labels (1=BUY, 0=HOLD, -1=SELL) or (1=BUY, -1=SELL, NaN)
        """
        if 'close' not in df.columns:
            raise ValueError("DataFrame must have 'close' column")

        # Calculate future returns
        future_close = df['close'].shift(-lookahead_bars)
        future_returns = (future_close - df['close']) / df['close']

        if binary:
            # Binary classification: Only BUY or SELL, no HOLD
            # Everything that doesn't meet threshold becomes NaN and gets filtered out
            labels = pd.Series(np.nan, index=df.index)
            labels[future_returns > buy_threshold] = Signal.BUY.value
            labels[future_returns < sell_threshold] = Signal.SELL.value
        else:
            # Ternary classification: BUY/HOLD/SELL
            labels = pd.Series(Signal.HOLD.value, index=df.index)
            labels[future_returns > buy_threshold] = Signal.BUY.value
            labels[future_returns < sell_threshold] = Signal.SELL.value

        # Remove last N bars (no future data available)
        labels.iloc[-lookahead_bars:] = np.nan

        return labels

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get strategy metadata.

        Returns:
            Dictionary with strategy information
        """
        return {
            'name': self.name,
            'is_trained': self.is_trained,
            **self.metadata
        }
