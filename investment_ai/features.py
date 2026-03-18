"""
Unified Feature Pipeline — 120+ features from 4 sources.

Source 1: FeatureEngineer (forex_system) — 56 base technical features
Source 2: UPGRADED_FEATURES — WorldQuant alphas, Hurst, momentum, mean reversion, interactions
Source 3: Strategy signals — RSI(2) mean reversion, trend following, dual momentum
Source 4: Derived features — returns, volume ratio, ATR ratio, consensus

Features are normalized per walk-forward fold (rank normalization) to ensure
stationarity across different price regimes.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure root is on path for UPGRADED_FEATURES import
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from forex_system.training.config import TrainingConfig, ASSET_REGISTRY, get_asset
from forex_system.training.data.price_loader import UniversalPriceLoader

log = logging.getLogger(__name__)


def generate_features(df: pd.DataFrame, asset_class: str = "stock") -> pd.DataFrame:
    """
    Generate 120+ features from OHLCV data by merging all sources.

    Args:
        df: DataFrame with columns [open, high, low, close, volume], datetime index.
        asset_class: 'stock', 'crypto', 'commodity', or 'index' — affects some features.

    Returns:
        DataFrame with original OHLCV + all feature columns.
        NaN rows from indicator warmup (first ~252 rows) are NOT dropped — caller decides.
    """
    if len(df) < 260:
        raise ValueError(f"Need ≥260 bars for feature generation, got {len(df)}")

    out = df.copy()

    # ── Source 1: FeatureEngineer (56 features) ──
    out = _add_base_features(out)

    # ── Source 2: UPGRADED_FEATURES (WorldQuant + Hurst + momentum + mean reversion) ──
    out = _add_upgraded_features(out)

    # ── Source 3: Strategy signals ──
    out = _add_strategy_signals(out)

    # ── Source 4: Derived features ──
    out = _add_derived_features(out, asset_class)

    n_features = len([c for c in out.columns if c not in df.columns])
    log.info(f"Generated {n_features} features ({len(out)} bars)")
    return out


def _add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Source 1: 56 features from FeatureEngineer."""
    try:
        from forex_system.services.feature_engineering import FeatureEngineer
        fe = FeatureEngineer()
        # FeatureEngineer.generate_features drops NaN rows — we don't want that.
        # Instead, add features individually to preserve all rows.
        out = df.copy()
        out = fe._add_trend_indicators(out)
        out = fe._add_momentum_indicators(out)
        out = fe._add_volatility_indicators(out)
        out = fe._add_volume_indicators(out)
        out = fe._add_price_action_features(out)
        out = fe._add_crossover_features(out)
        out = fe._add_divergence_features(out)
        out = fe._add_volatility_features(out)
        # Time features from index
        if hasattr(out.index, 'dayofweek'):
            out["day_of_week"] = out.index.dayofweek
            out["day_of_month"] = out.index.day
        return out
    except Exception as e:
        log.warning(f"FeatureEngineer failed: {e}. Adding manual base features.")
        return _add_manual_base_features(df)


def _add_manual_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback: compute key indicators manually if FeatureEngineer fails."""
    out = df.copy()
    close = out["close"]

    # SMAs
    for p in [20, 50, 200]:
        out[f"sma_{p}"] = close.rolling(p).mean()

    # EMAs
    for p in [12, 26]:
        out[f"ema_{p}"] = close.ewm(span=p, adjust=False).mean()

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    out["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # ATR
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - close.shift(1)).abs(),
        (out["low"] - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    # Bollinger
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out["bb_upper"] = sma20 + 2 * std20
    out["bb_lower"] = sma20 - 2 * std20
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / (sma20 + 1e-10)
    out["bb_pct"] = (close - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"] + 1e-10)

    # Price vs SMA
    out["price_vs_sma_20"] = (close - out["sma_20"]) / (out["sma_20"] + 1e-10)
    out["price_vs_sma_50"] = (close - out["sma_50"]) / (out["sma_50"] + 1e-10)

    # Returns
    out["return_1"] = close.pct_change(1)
    out["return_5"] = close.pct_change(5)
    out["return_10"] = close.pct_change(10)

    # Volume
    if "volume" in out.columns:
        out["vol_sma"] = out["volume"].rolling(20).mean()
        out["volume_ratio"] = out["volume"] / (out["vol_sma"] + 1e-10)

    return out


def _add_upgraded_features(df: pd.DataFrame) -> pd.DataFrame:
    """Source 2: WorldQuant alphas, Hurst, advanced momentum, mean reversion, interactions."""
    try:
        from UPGRADED_FEATURES import (
            calculate_worldquant_alphas,
            calculate_advanced_momentum,
            calculate_mean_reversion_features,
            calculate_feature_interactions,
            calculate_hurst_exponent,
        )

        out = df.copy()

        # WorldQuant alphas (10 factors)
        out = calculate_worldquant_alphas(out)

        # Advanced momentum (ROC, stochastic, CCI, Williams %R, multi-period RSI, MACD accel)
        out = calculate_advanced_momentum(out)

        # Mean reversion (z-scores, BB position, VWAP distance)
        out = calculate_mean_reversion_features(out)

        # Feature interactions (RSI×ATR, trend×MACD, volume×price)
        out = calculate_feature_interactions(out)

        # Rolling Hurst exponent (regime detection)
        out["hurst_exponent"] = out["close"].rolling(100).apply(
            lambda x: calculate_hurst_exponent(x, max_lag=50) if len(x) >= 50 else 0.5,
            raw=False,
        )
        out["hurst_regime"] = out["hurst_exponent"].apply(
            lambda h: 1 if h > 0.55 else (-1 if h < 0.45 else 0)
        )

        return out
    except Exception as e:
        log.warning(f"UPGRADED_FEATURES failed: {e}. Skipping advanced features.")
        return df


def _add_strategy_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Source 3: RSI(2), trend following, dual momentum signals."""
    out = df.copy()

    try:
        from forex_system.strategies.signals.rsi_mean_reversion import compute_rsi_signals
        rsi_df = compute_rsi_signals(df)
        for col in ["rsi2", "rsi_signal", "rsi_strength"]:
            if col in rsi_df.columns and col not in out.columns:
                out[col] = rsi_df[col]
    except Exception as e:
        log.warning(f"RSI signals failed: {e}")

    try:
        from forex_system.strategies.signals.trend_following import compute_trend_signals
        trend_df = compute_trend_signals(df)
        for col in ["sma20_slope", "trend_signal", "trend_strength"]:
            if col in trend_df.columns and col not in out.columns:
                out[col] = trend_df[col]
        # ADX from trend signals (may already exist from FeatureEngineer)
        if "adx" not in out.columns and "adx" in trend_df.columns:
            out["adx"] = trend_df["adx"]
    except Exception as e:
        log.warning(f"Trend signals failed: {e}")

    try:
        from forex_system.strategies.signals.dual_momentum import compute_momentum_signals
        mom_df = compute_momentum_signals(df)
        for col in ["mom_12m", "mom_1m", "mom_signal", "mom_strength"]:
            if col in mom_df.columns and col not in out.columns:
                out[col] = mom_df[col]
    except Exception as e:
        log.warning(f"Momentum signals failed: {e}")

    # Consensus features
    sig_cols = [c for c in ["rsi_signal", "trend_signal", "mom_signal"] if c in out.columns]
    if sig_cols:
        out["signal_sum"] = out[sig_cols].sum(axis=1)
        out["signal_consensus"] = (out[sig_cols].abs().sum(axis=1) > 0).astype(int) * np.sign(out["signal_sum"])

    return out


def _add_derived_features(df: pd.DataFrame, asset_class: str) -> pd.DataFrame:
    """Source 4: Returns, volume ratio, ATR ratio, etc."""
    out = df.copy()
    close = out["close"]

    # Multi-period returns (if not already added)
    for period, name in [(1, "ret_1d"), (5, "ret_5d"), (10, "ret_10d"), (20, "ret_20d")]:
        if name not in out.columns:
            out[name] = close.pct_change(period) * 100

    # RSI(2) if not from strategy signals
    if "rsi2" not in out.columns:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=0.5, min_periods=2, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=0.5, min_periods=2, adjust=False).mean()
        out["rsi2"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # Volume ratio
    if "volume" in out.columns and "volume_ratio" not in out.columns:
        vol_avg = out["volume"].rolling(20).mean()
        out["volume_ratio"] = out["volume"] / (vol_avg + 1e-10)

    # ATR ratio (current vs 100-day MA — volatility expansion/contraction)
    if "atr" in out.columns and "atr_ratio" not in out.columns:
        atr_ma100 = out["atr"].rolling(100, min_periods=50).mean()
        out["atr_ratio"] = out["atr"] / (atr_ma100 + 1e-10)

    # Overnight gap (stocks only)
    if asset_class == "stock":
        out["overnight_gap"] = (out["open"] - close.shift(1)) / (close.shift(1) + 1e-10) * 100

    # Time features
    if hasattr(out.index, "dayofweek"):
        if "day_of_week" not in out.columns:
            out["day_of_week"] = out.index.dayofweek
        if "day_of_month" not in out.columns:
            out["day_of_month"] = out.index.day

    return out


def normalize_features(
    features: pd.DataFrame,
    method: str = "rank",
    exclude_cols: list[str] = None,
) -> pd.DataFrame:
    """
    Normalize features for stationarity across walk-forward folds.

    Args:
        features: Feature DataFrame (numeric columns only).
        method: 'rank' (percentile rank, 0-1) or 'zscore' (rolling z-score).
        exclude_cols: Columns to skip normalization (e.g., binary signals).

    Returns:
        Normalized DataFrame.
    """
    exclude = set(exclude_cols or [])
    # Also exclude binary/categorical columns
    for col in features.columns:
        if features[col].nunique() <= 5:
            exclude.add(col)

    out = features.copy()
    norm_cols = [c for c in out.columns if c not in exclude]

    if method == "rank":
        out[norm_cols] = out[norm_cols].rank(pct=True)
    elif method == "zscore":
        for col in norm_cols:
            mean = out[col].rolling(252, min_periods=50).mean()
            std = out[col].rolling(252, min_periods=50).std()
            out[col] = (out[col] - mean) / (std + 1e-10)

    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return only the ML-usable feature columns (exclude OHLCV and non-numeric)."""
    ohlcv = {"open", "high", "low", "close", "volume"}
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in ohlcv]
