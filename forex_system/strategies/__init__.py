"""
Trading strategies module.

Provides ML-based trading strategies for forex markets.
"""

from .base import BaseStrategy, Signal
from .random_forest import RandomForestStrategy

__all__ = ["BaseStrategy", "Signal", "RandomForestStrategy"]
