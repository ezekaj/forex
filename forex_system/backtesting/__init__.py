"""
Backtesting module for strategy validation.

Provides realistic simulation of trading with costs and metrics.
"""

from .engine import BacktestEngine
from .metrics import BacktestMetrics

__all__ = ["BacktestEngine", "BacktestMetrics"]
