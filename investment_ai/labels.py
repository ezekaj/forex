"""
Label Generation — Triple Barrier (de Prado) + ATR-Dynamic.

Primary target: Triple barrier (+1/-1/0)
  - Upper barrier (profit target): entry + pt_sl_ratio × ATR × multiplier
  - Lower barrier (stop loss): entry - ATR × multiplier
  - Vertical barrier (time): max_holding bars
  - Label = which barrier is hit first

This is superior to binary UP/DOWN because the label ENCODES exit discipline.
The model learns "will the profit target be hit before the stop-loss?" — a much
cleaner question than "will price go up or down?"

Uses existing implementations from:
  - forex_system/training/data/label_generator.py (triple barrier)
  - UPGRADED_FEATURES.py (ATR-dynamic binary)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from forex_system.training.data.label_generator import LabelGenerator


def generate_triple_barrier_labels(
    df: pd.DataFrame,
    pt_sl_ratio: float = 2.0,
    max_holding: int = 10,
    atr_multiplier: float = 2.0,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Generate triple barrier labels using de Prado's method.

    With default params:
      - Stop-loss: 2 × ATR(14) below entry
      - Profit target: 4 × ATR(14) above entry (2:1 R:R)
      - Time barrier: 10 bars

    Args:
        df: OHLCV DataFrame with datetime index.
        pt_sl_ratio: Profit target / stop loss ratio (2.0 = 2:1 R:R).
        max_holding: Maximum bars to hold before time exit.
        atr_multiplier: ATR multiplier for stop-loss width.
        atr_period: ATR calculation period.

    Returns:
        DataFrame with columns: label (+1/-1/0), ret, holding_period, barrier_hit
    """
    gen = LabelGenerator()
    return gen.generate_triple_barrier_labels(
        df,
        pt_sl_ratio=pt_sl_ratio,
        max_holding=max_holding,
        atr_multiplier=atr_multiplier,
        atr_period=atr_period,
    )


def generate_atr_dynamic_labels(
    df: pd.DataFrame,
    lookahead_bars: int = 10,
    atr_multiplier: float = 2.0,
    atr_period: int = 14,
    binary: bool = True,
) -> pd.Series:
    """
    Generate ATR-dynamic threshold labels (from UPGRADED_FEATURES).

    Threshold adapts to current volatility: high-vol markets need bigger moves
    to count as a signal, low-vol markets trigger on smaller moves.

    Args:
        df: OHLCV DataFrame.
        lookahead_bars: Number of bars to look ahead.
        atr_multiplier: Threshold = ATR × multiplier.
        atr_period: ATR calculation period.
        binary: If True, only +1/-1 (no 0/HOLD, NaN instead).

    Returns:
        Series with labels: +1 (BUY), -1 (SELL), 0 (HOLD) or NaN.
    """
    # Always use corrected version — UPGRADED_FEATURES has a bug where
    # ATR (price units) is compared directly to returns (fraction), causing 0 labels.
    return _manual_atr_labels(df, lookahead_bars, atr_multiplier, atr_period, binary)


def _manual_atr_labels(df, lookahead, multiplier, period, binary):
    """Fallback ATR-dynamic label generation."""
    close = df["close"]
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - close.shift(1)).abs(),
        (df["low"] - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    future_close = close.shift(-lookahead)
    future_ret = (future_close - close) / close

    buy_thresh = atr * multiplier / close
    sell_thresh = -atr * multiplier / close

    if binary:
        labels = pd.Series(np.nan, index=df.index)
        labels[future_ret > buy_thresh] = 1
        labels[future_ret < sell_thresh] = -1
    else:
        labels = pd.Series(0, index=df.index, dtype=float)
        labels[future_ret > buy_thresh] = 1
        labels[future_ret < sell_thresh] = -1

    labels.iloc[-lookahead:] = np.nan
    return labels


def get_purge_gap(max_holding: int) -> int:
    """
    Minimum purge gap for walk-forward to prevent label leakage.
    Must be >= max_holding since triple barrier looks that far forward.
    """
    return max(max_holding, 10)
