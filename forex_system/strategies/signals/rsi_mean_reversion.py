"""
RSI(2) Mean Reversion Signal Generator.

Connors-style: buy extreme oversold in uptrends, sell extreme overbought in downtrends.
SMA(200) acts as trend filter — only trade with the long-term trend.

Signal:
    LONG (+1):  RSI(2) < 10 AND close > SMA(200)
    SHORT (-1): RSI(2) > 90 AND close < SMA(200)
    FLAT (0):   Otherwise

Strength: how extreme the RSI reading is (0 = barely triggered, 1 = most extreme).
"""

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_rsi_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSI(2) mean reversion signals to OHLCV DataFrame.

    Columns added:
        rsi2            — RSI with period 2
        sma200          — 200-day simple moving average
        rsi_signal      — -1, 0, or +1
        rsi_strength    — 0.0 to 1.0 (how extreme the RSI reading)
        rsi_exit        — True when RSI(2) crosses back toward 50

    Returns new DataFrame (original is not mutated).
    """
    out = df.copy()
    out["rsi2"] = _rsi(out["close"], period=2)
    out["sma200"] = out["close"].rolling(window=200, min_periods=200).mean()

    # Default: no signal
    out["rsi_signal"] = 0
    out["rsi_strength"] = 0.0

    # Long: RSI(2) < 10, price above SMA(200)
    long_mask = (out["rsi2"] < 10) & (out["close"] > out["sma200"])
    out.loc[long_mask, "rsi_signal"] = 1
    # Strength: RSI=0 → strength=1.0, RSI=10 → strength=0.0
    out.loc[long_mask, "rsi_strength"] = (10 - out.loc[long_mask, "rsi2"]).clip(0, 10) / 10

    # Short: RSI(2) > 90, price below SMA(200)
    short_mask = (out["rsi2"] > 90) & (out["close"] < out["sma200"])
    out.loc[short_mask, "rsi_signal"] = -1
    # Strength: RSI=100 → strength=1.0, RSI=90 → strength=0.0
    out.loc[short_mask, "rsi_strength"] = (out.loc[short_mask, "rsi2"] - 90).clip(0, 10) / 10

    # Exit signal: RSI crosses 50 (mean reversion complete)
    out["rsi_exit"] = (
        ((out["rsi2"] > 50) & (out["rsi2"].shift(1) <= 50)) |
        ((out["rsi2"] < 50) & (out["rsi2"].shift(1) >= 50))
    )

    return out
