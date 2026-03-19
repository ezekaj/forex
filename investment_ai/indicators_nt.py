"""
Indicators extracted from NautilusTrader's codebase.
Standalone implementations — no framework dependency.

Sources:
  - KeltnerPosition: nautilus_trader/indicators/volatility.pyx lines 738-852
  - EfficiencyRatio: nautilus_trader/indicators/efficiency_ratio.pyx
  - Pressure: nautilus_trader/indicators/volume.pyx lines 333-443
  - FuzzyCandlesticks: nautilus_trader/indicators/fuzzy_candlesticks.pyx
"""

import numpy as np
import pandas as pd


def keltner_position(df: pd.DataFrame, period: int = 20, atr_period: int = 14, k: float = 2.0) -> pd.Series:
    """
    Where price sits relative to Keltner Channel, measured in ATR multiples.

    +2.0 = price is 2×ATR above the channel midline (very extended uptrend)
    -1.5 = price is 1.5×ATR below midline (oversold)
     0.0 = price is at the channel midline

    Better than raw returns for cross-sectional ranking because it's
    volatility-normalized. A +2.0 score on AAPL (low vol) is comparable
    to a +2.0 on TSLA (high vol).
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Keltner midline = EMA
    mid = close.ewm(span=period, adjust=False).mean()

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_period, min_periods=atr_period, adjust=False).mean()

    # Position = (price - midline) / ATR
    position = (close - mid) / (atr + 1e-10)
    return position


def efficiency_ratio(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Kaufman Efficiency Ratio — signal-to-noise filter.

    ER = abs(net_price_change) / sum(abs(daily_changes))

    ER = 1.0: perfectly trending (all daily moves in same direction)
    ER = 0.0: pure noise (daily moves cancel out)

    Use: only include stocks with ER > 0.3 in momentum portfolio.
    Filters out sideways, choppy stocks that whipsaw momentum strategies.
    """
    close = df["close"]
    net_change = (close - close.shift(period)).abs()
    daily_changes = close.diff().abs().rolling(period).sum()
    er = net_change / (daily_changes + 1e-10)
    return er


def pressure(df: pd.DataFrame, atr_period: int = 14, vol_period: int = 20) -> pd.DataFrame:
    """
    Volume-confirming buy/sell pressure indicator.

    buy_pressure  = ((close - low) / ATR) × (volume / avg_volume)
    sell_pressure = ((high - close) / ATR) × (volume / avg_volume)
    pressure = buy_pressure - sell_pressure

    Positive = buying dominant (volume on up-moves)
    Negative = selling dominant (volume on down-moves)

    Returns DataFrame with columns: pressure, pressure_cumulative
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_period, min_periods=atr_period, adjust=False).mean()

    # Volume ratio
    avg_vol = volume.rolling(vol_period).mean()
    vol_ratio = volume / (avg_vol + 1e-10)

    # Pressure
    buy_p = ((close - low) / (atr + 1e-10)) * vol_ratio
    sell_p = ((high - close) / (atr + 1e-10)) * vol_ratio
    net_pressure = buy_p - sell_p

    result = pd.DataFrame(index=df.index)
    result["pressure"] = net_pressure
    result["pressure_cumulative"] = net_pressure.cumsum()
    return result


def fuzzy_candlesticks(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Convert OHLCV into categorical vectors for LLM chart analysis.
    Compresses price data by ~80% in tokens while preserving signal.

    Output columns:
      direction: UP, DOWN, DOJI
      size: SMALL, MEDIUM, LARGE (relative to period avg range)
      body_pct: body as % of total range (0-100)
      upper_wick_pct: upper wick as % of range
      lower_wick_pct: lower wick as % of range
      volume_cat: LOW, NORMAL, HIGH (relative to period avg)
    """
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # Range statistics
    bar_range = h - l
    avg_range = bar_range.rolling(period).mean()
    body = (c - o).abs()

    result = pd.DataFrame(index=df.index)

    # Direction
    result["direction"] = np.where(c > o, "UP", np.where(c < o, "DOWN", "DOJI"))

    # Size relative to average
    range_ratio = bar_range / (avg_range + 1e-10)
    result["size"] = np.where(range_ratio > 1.5, "LARGE", np.where(range_ratio < 0.5, "SMALL", "MEDIUM"))

    # Body percentage of range
    result["body_pct"] = (body / (bar_range + 1e-10) * 100).round(0).astype(int)

    # Wicks
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    result["upper_wick_pct"] = (upper_wick / (bar_range + 1e-10) * 100).round(0).astype(int)
    result["lower_wick_pct"] = (lower_wick / (bar_range + 1e-10) * 100).round(0).astype(int)

    # Volume category
    avg_vol = v.rolling(period).mean()
    vol_ratio = v / (avg_vol + 1e-10)
    result["volume_cat"] = np.where(vol_ratio > 1.5, "HIGH", np.where(vol_ratio < 0.5, "LOW", "NORMAL"))

    return result


def spread_analyzer(bid: pd.Series, ask: pd.Series, period: int = 20) -> pd.DataFrame:
    """
    Analyze spread width for execution quality filtering.

    Skip stocks when current spread > 2× 20-day average.
    Wide spreads = poor fill quality on rebalance.

    Note: requires bid/ask data. For daily bars, approximate with (high-low)/close.
    """
    spread = ask - bid
    avg_spread = spread.rolling(period).mean()
    spread_ratio = spread / (avg_spread + 1e-10)

    result = pd.DataFrame(index=bid.index)
    result["spread"] = spread
    result["avg_spread"] = avg_spread
    result["spread_ratio"] = spread_ratio
    result["skip"] = spread_ratio > 2.0  # flag to skip this stock
    return result


def approximate_spread(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Approximate spread from daily OHLCV (no bid/ask data).
    Uses (high - low) / close as proxy.
    Returns spread_ratio: current vs 20-day average.
    """
    spread_proxy = (df["high"] - df["low"]) / (df["close"] + 1e-10)
    avg_spread = spread_proxy.rolling(period).mean()
    return spread_proxy / (avg_spread + 1e-10)
