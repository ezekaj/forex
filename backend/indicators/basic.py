"""
Technical indicators ported from AlphaThink TypeScript implementation.
Provides core indicators: RSI, Bollinger Bands, ATR, ADX, Pivot Points, MACD.
"""

import numpy as np
from typing import List, Dict, Tuple
from ..models.asset import Candle


class TechnicalIndicators:
    """Collection of technical indicators"""

    @staticmethod
    def calculate_atr(candles: List[Candle], period: int = 14) -> float:
        """
        Calculate Average True Range (ATR) - volatility indicator.
        ATR measures market volatility by calculating the average of true ranges.
        """
        if len(candles) < period + 1:
            return 0.0

        tr_sum = 0.0

        # Calculate True Range for the initial period
        for i in range(1, period + 1):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i - 1].close

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_sum += tr

        atr = tr_sum / period

        # Smoothing (Wilder's smoothing)
        for i in range(period + 1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i - 1].close

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            atr = ((atr * (period - 1)) + tr) / period

        return atr

    @staticmethod
    def calculate_pivot_points(last_candle: Candle) -> Dict[str, float]:
        """
        Calculate Pivot Points - support and resistance levels.
        Standard pivot points based on previous period's high, low, close.
        """
        pp = (last_candle.high + last_candle.low + last_candle.close) / 3
        r1 = 2 * pp - last_candle.low
        r2 = pp + (last_candle.high - last_candle.low)
        r3 = r1 + (last_candle.high - last_candle.low)
        s1 = 2 * pp - last_candle.high
        s2 = pp - (last_candle.high - last_candle.low)
        s3 = s1 - (last_candle.high - last_candle.low)

        return {
            "pivot": pp,
            "r1": r1,
            "r2": r2,
            "r3": r3,
            "s1": s1,
            "s2": s2,
            "s3": s3
        }

    @staticmethod
    def calculate_adx(candles: List[Candle], period: int = 14) -> float:
        """
        Calculate ADX (Average Directional Index) - trend strength indicator.
        Simplified version for high-frequency trading speed.
        Returns value from 0 to 100 (25+ indicates strong trend).
        """
        if len(candles) < period * 2:
            return 20.0  # Default to weak trend if insufficient data

        up_moves = 0.0
        down_moves = 0.0
        total_moves = 0.0

        # Measure consistency of directional movement over last period
        for i in range(len(candles) - period, len(candles)):
            change = candles[i].close - candles[i].open
            total_moves += abs(change)
            if change > 0:
                up_moves += change
            else:
                down_moves += abs(change)

        if total_moves == 0:
            return 20.0

        # Directionality: how much movement is in one direction vs scattered
        directionality = abs(up_moves - down_moves) / total_moves
        return directionality * 100  # 0 to 100

    @staticmethod
    def calculate_rsi(candles: List[Candle], period: int = 14) -> float:
        """
        Calculate RSI (Relative Strength Index) - momentum oscillator.
        Measures speed and magnitude of price changes.
        Returns value from 0 to 100 (>70 overbought, <30 oversold).
        """
        if len(candles) < period + 1:
            return 50.0  # Neutral

        prices = [c.close for c in candles]
        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        recent_changes = changes[-period:]

        gains = sum(c for c in recent_changes if c > 0)
        losses = sum(abs(c) for c in recent_changes if c < 0)

        avg_gain = gains / period
        avg_loss = losses / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_bollinger_bands(
        candles: List[Candle],
        period: int = 20,
        multiplier: float = 2.0
    ) -> Dict[str, float]:
        """
        Calculate Bollinger Bands - volatility bands.
        Consists of middle band (SMA), upper band (SMA + 2*StdDev), lower band (SMA - 2*StdDev).
        """
        prices = [c.close for c in candles]

        if len(prices) < period:
            return {"upper": 0, "middle": 0, "lower": 0}

        # Get recent prices
        slice_prices = prices[-period:]

        # Calculate mean (middle band)
        mean = sum(slice_prices) / period

        # Calculate standard deviation
        squared_diffs = [(p - mean) ** 2 for p in slice_prices]
        variance = sum(squared_diffs) / period
        std_dev = variance ** 0.5

        return {
            "middle": mean,
            "upper": mean + (std_dev * multiplier),
            "lower": mean - (std_dev * multiplier)
        }

    @staticmethod
    def calculate_macd(
        candles: List[Candle],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict[str, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        Trend-following momentum indicator.
        Returns MACD line, signal line, and histogram.
        """
        prices = [c.close for c in candles]

        if len(prices) < slow_period:
            return {"macd_line": 0, "signal_line": 0, "histogram": 0}

        # Simplified version for speed (full version requires EMA calculation)
        # For production, use proper EMA calculation
        last = prices[-1]
        prev = prices[-5] if len(prices) >= 5 else prices[0]
        momentum = last - prev

        return {
            "macd_line": momentum,
            "signal_line": 0,
            "histogram": momentum
        }

    @staticmethod
    def calculate_sma(candles: List[Candle], period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(candles) < period:
            return 0.0

        prices = [c.close for c in candles[-period:]]
        return sum(prices) / period

    @staticmethod
    def calculate_ema(candles: List[Candle], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(candles) < period:
            return 0.0

        prices = [c.close for c in candles]
        multiplier = 2 / (period + 1)

        # Start with SMA for initial EMA
        ema = sum(prices[:period]) / period

        # Calculate EMA for remaining prices
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    @classmethod
    def get_all_indicators(cls, candles: List[Candle]) -> Dict:
        """
        Calculate all indicators at once.
        Returns complete set of technical indicators.
        """
        if len(candles) == 0:
            return {}

        last_candle = candles[-1]

        return {
            "rsi": cls.calculate_rsi(candles),
            "bollinger": cls.calculate_bollinger_bands(candles),
            "macd": cls.calculate_macd(candles),
            "atr": cls.calculate_atr(candles),
            "adx": cls.calculate_adx(candles),
            "pivots": cls.calculate_pivot_points(last_candle),
            "sma_20": cls.calculate_sma(candles, 20),
            "sma_50": cls.calculate_sma(candles, 50),
            "ema_12": cls.calculate_ema(candles, 12),
            "ema_26": cls.calculate_ema(candles, 26)
        }


# Convenience functions for direct access
def calculate_atr(candles: List[Candle], period: int = 14) -> float:
    return TechnicalIndicators.calculate_atr(candles, period)


def calculate_rsi(candles: List[Candle], period: int = 14) -> float:
    return TechnicalIndicators.calculate_rsi(candles, period)


def calculate_bollinger_bands(candles: List[Candle], period: int = 20, multiplier: float = 2.0):
    return TechnicalIndicators.calculate_bollinger_bands(candles, period, multiplier)


def calculate_adx(candles: List[Candle], period: int = 14) -> float:
    return TechnicalIndicators.calculate_adx(candles, period)


def calculate_pivot_points(last_candle: Candle):
    return TechnicalIndicators.calculate_pivot_points(last_candle)


def get_all_indicators(candles: List[Candle]) -> Dict:
    return TechnicalIndicators.get_all_indicators(candles)
