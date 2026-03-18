"""
Market Regime Detector — classifies market state for strategy selection.

Three regimes:
    TRENDING (2):  ADX > 25 — trend following strategies dominate
    RANGING (0):   ADX < 20 — mean reversion strategies dominate
    VOLATILE (1):  ATR/price > 2x its 100-day MA — reduce position sizes

The regime determines which strategy signals to trust:
    - TRENDING → weight trend following and momentum higher
    - RANGING → weight RSI mean reversion higher
    - VOLATILE → reduce all position sizes by 50%
"""

import numpy as np
import pandas as pd


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def detect_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add regime classification to DataFrame.

    Requires 'adx' column (from trend_following signals).
    If 'adx' is missing, computes it.

    Columns added:
        atr             — ATR(14)
        atr_pct         — ATR as % of price
        atr_pct_ma100   — 100-day MA of atr_pct
        atr_ratio       — atr_pct / atr_pct_ma100
        regime          — 'trending', 'ranging', or 'volatile'
        regime_code     — 2=trending, 0=ranging, 1=volatile
        volatile        — boolean, True when ATR ratio > 2.0

    Returns new DataFrame.
    """
    out = df.copy()

    # Compute ADX if not present
    if "adx" not in out.columns:
        from forex_system.strategies.signals.trend_following import compute_trend_signals
        temp = compute_trend_signals(out)
        out["adx"] = temp["adx"]

    # ATR
    out["atr"] = _atr(out)
    out["atr_pct"] = out["atr"] / (out["close"] + 1e-10) * 100
    out["atr_pct_ma100"] = out["atr_pct"].rolling(window=100, min_periods=50).mean()
    out["atr_ratio"] = out["atr_pct"] / (out["atr_pct_ma100"] + 1e-10)

    # Volatile: ATR is 2x its 100-day average
    out["volatile"] = out["atr_ratio"] > 2.0

    # Regime classification (volatile overrides trend/range)
    out["regime"] = "ranging"
    out["regime_code"] = 0

    trending_mask = out["adx"] > 25
    out.loc[trending_mask, "regime"] = "trending"
    out.loc[trending_mask, "regime_code"] = 2

    volatile_mask = out["volatile"]
    out.loc[volatile_mask, "regime"] = "volatile"
    out.loc[volatile_mask, "regime_code"] = 1

    return out
