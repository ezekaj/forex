"""
Dual Momentum Signal Generator — Antonacci-style.

Uses two timeframes of momentum: 12-month (252 trading days) and 1-month (21 trading days).
Both must agree for a signal. When they disagree, stay flat.

Signal:
    LONG (+1):  12-month return > 0 AND 1-month return > 0
    SHORT (-1): 12-month return < 0 AND 1-month return < 0
    FLAT (0):   Timeframes disagree (one positive, one negative)

Strength: based on magnitude of the weaker timeframe's momentum.
"""

import numpy as np
import pandas as pd


def compute_momentum_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add dual momentum signals to OHLCV DataFrame.

    Columns added:
        mom_12m         — 252-day return (%)
        mom_1m          — 21-day return (%)
        mom_3m          — 63-day return (%) (for additional context)
        mom_signal      — -1, 0, or +1
        mom_strength    — 0.0 to 1.0

    Returns new DataFrame.
    """
    out = df.copy()

    # Momentum: percentage return over lookback period
    out["mom_12m"] = out["close"].pct_change(252) * 100
    out["mom_3m"] = out["close"].pct_change(63) * 100
    out["mom_1m"] = out["close"].pct_change(21) * 100

    # Default: no signal
    out["mom_signal"] = 0
    out["mom_strength"] = 0.0

    # Long: both timeframes positive
    long_mask = (out["mom_12m"] > 0) & (out["mom_1m"] > 0)
    out.loc[long_mask, "mom_signal"] = 1

    # Short: both timeframes negative
    short_mask = (out["mom_12m"] < 0) & (out["mom_1m"] < 0)
    out.loc[short_mask, "mom_signal"] = -1

    # Strength: the weaker timeframe determines conviction
    # Normalize: 0% return → 0 strength, ±20% 1-month → 1.0 strength
    for mask, sign in [(long_mask, 1), (short_mask, -1)]:
        if mask.sum() == 0:
            continue
        weaker = pd.concat([
            out.loc[mask, "mom_12m"].abs() / 50,   # 50% annual → strength 1.0
            out.loc[mask, "mom_1m"].abs() / 10,     # 10% monthly → strength 1.0
        ], axis=1).min(axis=1)
        out.loc[mask, "mom_strength"] = weaker.clip(0, 1)

    # Exit: either timeframe flips
    prev_signal = out["mom_signal"].shift(1)
    out["mom_exit"] = (out["mom_signal"] == 0) & (prev_signal != 0)

    return out
