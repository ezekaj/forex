"""
Technical indicators module for forex trading system.

Provides trend, momentum, volatility, and volume indicators.
"""

# Base class
from .base import BaseIndicator

# Trend indicators
from .trend import SMA, EMA, MACD, ADX

# Momentum indicators
from .momentum import RSI, Stochastic, CCI, ROC

# Volatility indicators
from .volatility import ATR, BollingerBands, StandardDeviation, KeltnerChannel

# Volume indicators
from .volume import OBV, VolumeSMA, MFI, VWAP, ADI, CMF


__all__ = [
    # Base
    'BaseIndicator',

    # Trend
    'SMA',
    'EMA',
    'MACD',
    'ADX',

    # Momentum
    'RSI',
    'Stochastic',
    'CCI',
    'ROC',

    # Volatility
    'ATR',
    'BollingerBands',
    'StandardDeviation',
    'KeltnerChannel',

    # Volume
    'OBV',
    'VolumeSMA',
    'MFI',
    'VWAP',
    'ADI',
    'CMF',
]
