"""
Technical Indicators
====================
Calculate all technical indicators for trading analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators from OHLCV data."""

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index."""
        high = df['high']
        low = df['low']
        close = df['close']

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr = TechnicalIndicators.atr(df, 1)
        atr = tr.rolling(window=period).mean()

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx

    @staticmethod
    def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator."""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()

        k = 100 * (df['close'] - low_min) / (high_max - low_min)
        d = k.rolling(window=d_period).mean()
        return k, d

    @staticmethod
    def support_resistance(df: pd.DataFrame, window: int = 20, num_levels: int = 3) -> Dict:
        """
        Find support and resistance levels.

        Returns:
            Dict with 'support' and 'resistance' lists
        """
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()

        # Find local maxima (resistance) and minima (support)
        resistance_levels = []
        support_levels = []

        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(df['high'].iloc[i])
            if df['low'].iloc[i] == lows.iloc[i]:
                support_levels.append(df['low'].iloc[i])

        # Cluster similar levels and return top ones
        resistance_levels = list(set([round(r, 5) for r in resistance_levels]))
        support_levels = list(set([round(s, 5) for s in support_levels]))

        resistance_levels.sort(reverse=True)
        support_levels.sort()

        current_price = df['close'].iloc[-1]

        # Filter to levels near current price
        resistance = [r for r in resistance_levels if r > current_price][:num_levels]
        support = [s for s in support_levels if s < current_price][:num_levels]

        return {
            'support': support,
            'resistance': resistance,
            'current_price': current_price
        }

    @staticmethod
    def trend_direction(df: pd.DataFrame, short_period: int = 20, long_period: int = 50) -> str:
        """
        Determine trend direction.

        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        close = df['close']
        short_ma = close.rolling(window=short_period).mean()
        long_ma = close.rolling(window=long_period).mean()

        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        current_price = close.iloc[-1]

        if current_price > current_short > current_long:
            return 'bullish'
        elif current_price < current_short < current_long:
            return 'bearish'
        else:
            return 'neutral'

    @staticmethod
    def volatility_regime(df: pd.DataFrame, atr_period: int = 14) -> str:
        """
        Determine volatility regime.

        Returns:
            'high', 'normal', or 'low'
        """
        atr = TechnicalIndicators.atr(df, atr_period)
        atr_mean = atr.rolling(window=50).mean()
        atr_std = atr.rolling(window=50).std()

        current_atr = atr.iloc[-1]
        mean_atr = atr_mean.iloc[-1]
        std_atr = atr_std.iloc[-1]

        if pd.isna(mean_atr) or pd.isna(std_atr):
            return 'normal'

        if current_atr > mean_atr + std_atr:
            return 'high'
        elif current_atr < mean_atr - std_atr:
            return 'low'
        else:
            return 'normal'


def calculate_all_indicators(df: pd.DataFrame) -> Dict:
    """
    Calculate all indicators for a DataFrame.

    Args:
        df: OHLCV DataFrame

    Returns:
        Dict containing all indicator values and analysis
    """
    ti = TechnicalIndicators()

    # Normalize column names
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    indicators = {}

    # Price info
    indicators['current_price'] = df['close'].iloc[-1]
    indicators['price_change_24h'] = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100 if len(df) > 6 else 0

    # Moving averages
    indicators['ema_9'] = ti.ema(df['close'], 9).iloc[-1]
    indicators['ema_21'] = ti.ema(df['close'], 21).iloc[-1]
    indicators['ema_50'] = ti.ema(df['close'], 50).iloc[-1]
    indicators['ema_200'] = ti.ema(df['close'], 200).iloc[-1] if len(df) >= 200 else None

    # RSI
    rsi = ti.rsi(df['close'], 14)
    indicators['rsi'] = rsi.iloc[-1]
    indicators['rsi_series'] = rsi

    # MACD
    macd, signal, hist = ti.macd(df['close'])
    indicators['macd'] = macd.iloc[-1]
    indicators['macd_signal'] = signal.iloc[-1]
    indicators['macd_histogram'] = hist.iloc[-1]
    indicators['macd_series'] = macd
    indicators['macd_signal_series'] = signal

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = ti.bollinger_bands(df['close'])
    indicators['bb_upper'] = bb_upper.iloc[-1]
    indicators['bb_middle'] = bb_middle.iloc[-1]
    indicators['bb_lower'] = bb_lower.iloc[-1]
    indicators['bb_position'] = (df['close'].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) if bb_upper.iloc[-1] != bb_lower.iloc[-1] else 0.5

    # ATR
    atr = ti.atr(df, 14)
    indicators['atr'] = atr.iloc[-1]
    indicators['atr_percent'] = (atr.iloc[-1] / df['close'].iloc[-1]) * 100

    # ADX
    adx = ti.adx(df, 14)
    indicators['adx'] = adx.iloc[-1]

    # Stochastic
    stoch_k, stoch_d = ti.stochastic(df)
    indicators['stoch_k'] = stoch_k.iloc[-1]
    indicators['stoch_d'] = stoch_d.iloc[-1]

    # Support/Resistance
    sr = ti.support_resistance(df)
    indicators['support_levels'] = sr['support']
    indicators['resistance_levels'] = sr['resistance']

    # Trend
    indicators['trend'] = ti.trend_direction(df)

    # Volatility regime
    indicators['volatility'] = ti.volatility_regime(df)

    # Signal conditions
    indicators['rsi_oversold'] = indicators['rsi'] < 30 if not pd.isna(indicators['rsi']) else False
    indicators['rsi_overbought'] = indicators['rsi'] > 70 if not pd.isna(indicators['rsi']) else False
    indicators['macd_bullish'] = indicators['macd_histogram'] > 0 if not pd.isna(indicators['macd_histogram']) else False
    indicators['price_above_ema50'] = indicators['current_price'] > indicators['ema_50'] if indicators['ema_50'] else False

    return indicators


def format_indicators_for_llm(indicators: Dict, pair: str) -> str:
    """
    Format indicators as text for LLM consumption.

    Args:
        indicators: Dict from calculate_all_indicators
        pair: Currency pair name

    Returns:
        Formatted string for LLM prompt
    """
    text = f"""
=== {pair} Technical Analysis ===

PRICE:
  Current: {indicators['current_price']:.5f}
  24h Change: {indicators['price_change_24h']:.2f}%

MOVING AVERAGES:
  EMA(9): {indicators['ema_9']:.5f}
  EMA(21): {indicators['ema_21']:.5f}
  EMA(50): {indicators['ema_50']:.5f}
  Price vs EMA50: {'Above' if indicators['price_above_ema50'] else 'Below'}

MOMENTUM:
  RSI(14): {indicators['rsi']:.1f} {'(OVERSOLD)' if indicators['rsi_oversold'] else '(OVERBOUGHT)' if indicators['rsi_overbought'] else ''}
  Stochastic K/D: {indicators['stoch_k']:.1f} / {indicators['stoch_d']:.1f}
  MACD: {indicators['macd']:.6f}
  MACD Signal: {indicators['macd_signal']:.6f}
  MACD Histogram: {indicators['macd_histogram']:.6f} ({'Bullish' if indicators['macd_bullish'] else 'Bearish'})

VOLATILITY:
  ATR(14): {indicators['atr']:.5f} ({indicators['atr_percent']:.2f}%)
  ADX(14): {indicators['adx']:.1f}
  Regime: {indicators['volatility'].upper()}

BOLLINGER BANDS:
  Upper: {indicators['bb_upper']:.5f}
  Middle: {indicators['bb_middle']:.5f}
  Lower: {indicators['bb_lower']:.5f}
  Position: {indicators['bb_position']*100:.1f}% (0=lower, 100=upper)

STRUCTURE:
  Trend: {indicators['trend'].upper()}
  Support: {', '.join([f"{s:.5f}" for s in indicators['support_levels'][:3]]) if indicators['support_levels'] else 'None identified'}
  Resistance: {', '.join([f"{r:.5f}" for r in indicators['resistance_levels'][:3]]) if indicators['resistance_levels'] else 'None identified'}
"""
    return text.strip()


if __name__ == "__main__":
    # Test with sample data
    from chart_generator import generate_sample_data

    df = generate_sample_data("EURUSD", bars=200)
    indicators = calculate_all_indicators(df)

    print(format_indicators_for_llm(indicators, "EURUSD"))
