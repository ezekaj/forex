"""
Trading strategies module.

Provides ML-based trading strategies for forex markets.
"""

from .base import BaseStrategy, Signal
from .random_forest import RandomForestStrategy
from .xgboost_strategy import XGBoostStrategy

__all__ = ["BaseStrategy", "Signal", "RandomForestStrategy", "XGBoostStrategy"]
