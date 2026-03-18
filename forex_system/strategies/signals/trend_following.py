"""
Trend Following Signal Generator — SMA Crossover + ADX Filter.

Uses the 20/50 SMA crossover for direction, ADX(14) for trend strength filter.
Only generates signals when the trend is strong enough (ADX > 25).

Signal:
    LONG (+1):  SMA(20) > SMA(50) AND ADX > 25
    SHORT (-1): SMA(20) < SMA(50) AND ADX > 25
    FLAT (0):   ADX < 25 (market not trending)

Strength: based on ADX level and SMA separation.
"""

import numpy as np
import pandas as pd


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute ADX, +DI, -DI."""
    high, low, close = df["high"], df["low"], df["close"]

    # True range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Smoothed averages (Wilder's smoothing)
    alpha = 1 / period
    atr = pd.Series(tr, index=df.index).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di = pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, min_periods=period, adjust=False).mean() / (atr + 1e-10) * 100
    minus_di = pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, min_periods=period, adjust=False).mean() / (atr + 1e-10) * 100

    # ADX
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10) * 100
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    return pd.DataFrame({"adx": adx, "plus_di": plus_di, "minus_di": minus_di}, index=df.index)


def compute_trend_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trend following signals to OHLCV DataFrame.

    Columns added:
        sma20           — 20-day SMA
        sma50           — 50-day SMA
        adx             — ADX(14)
        plus_di         — +DI
        minus_di        — -DI
        sma20_slope     — 5-day slope of SMA20 (normalized)
        trend_signal    — -1, 0, or +1
        trend_strength  — 0.0 to 1.0

    Returns new DataFrame.
    """
    out = df.copy()
    out["sma20"] = _sma(out["close"], 20)
    out["sma50"] = _sma(out["close"], 50)

    adx_df = _adx(out)
    out["adx"] = adx_df["adx"]
    out["plus_di"] = adx_df["plus_di"]
    out["minus_di"] = adx_df["minus_di"]

    # SMA20 slope: rate of change over 5 days, normalized by price
    out["sma20_slope"] = (out["sma20"] - out["sma20"].shift(5)) / (out["close"] + 1e-10)

    # Default: no signal
    out["trend_signal"] = 0
    out["trend_strength"] = 0.0

    trending = out["adx"] > 25

    # Long: SMA20 > SMA50 in trending market
    long_mask = trending & (out["sma20"] > out["sma50"])
    out.loc[long_mask, "trend_signal"] = 1

    # Short: SMA20 < SMA50 in trending market
    short_mask = trending & (out["sma20"] < out["sma50"])
    out.loc[short_mask, "trend_signal"] = -1

    # Strength: ADX 25→0.0, ADX 50+→1.0
    out.loc[trending, "trend_strength"] = ((out.loc[trending, "adx"] - 25) / 25).clip(0, 1)

    # Exit: ADX drops below 20 (trend exhaustion)
    out["trend_exit"] = (out["adx"] < 20) & (out["adx"].shift(1) >= 20)

    return out
