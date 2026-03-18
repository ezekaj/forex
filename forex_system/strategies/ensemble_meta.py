"""
Ensemble Meta-Learner — XGBoost combines strategy signals into final decisions.

Architecture:
    Layer 1: Three rule-based signal generators (no ML)
        - RSI(2) mean reversion
        - SMA crossover trend following
        - Dual momentum

    Layer 2: Regime detector (rule-based)
        - Trending / Ranging / Volatile

    Layer 3: XGBoost meta-learner (ML)
        - Takes signals + regime + raw technicals as features
        - Predicts probability of UP vs DOWN over horizon
        - Walk-forward trained (never sees future data)

    Layer 4: Position sizing
        - Kelly criterion from XGBoost probability
        - Halved in volatile regime
        - No trade below 55% confidence

The meta-learner does NOT predict the market directly.
It predicts WHICH STRATEGY SIGNAL TO TRUST in current conditions.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from forex_system.strategies.signals.rsi_mean_reversion import compute_rsi_signals
from forex_system.strategies.signals.trend_following import compute_trend_signals
from forex_system.strategies.signals.dual_momentum import compute_momentum_signals
from forex_system.strategies.regime_detector import detect_regime

log = logging.getLogger(__name__)

# Features used by the meta-learner
FEATURE_COLS = [
    # Strategy signals
    "rsi_signal", "rsi_strength",
    "trend_signal", "trend_strength",
    "mom_signal", "mom_strength",
    # Regime
    "regime_code", "volatile",
    # Raw technicals
    "rsi2", "adx", "atr_pct", "atr_ratio",
    "sma20_slope",
    # Price action
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    # Volatility context
    "bb_width", "bb_pct",
    # Volume
    "volume_ratio",
    # Strategy agreement
    "signal_consensus", "signal_sum",
]


@dataclass(frozen=True)
class EnsembleConfig:
    """Configuration for the ensemble system."""
    # XGBoost parameters
    n_estimators: int = 200
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.7
    min_child_weight: int = 5
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42

    # Trading parameters
    horizon_days: int = 3          # prediction horizon
    min_confidence: float = 0.55   # minimum XGBoost probability to trade
    cost_per_trade: float = 0.001  # 0.1% per side (0.2% round trip)
    max_position_frac: float = 0.25  # max 25% of capital per trade
    volatile_reduction: float = 0.5  # reduce position 50% in volatile regime


class EnsembleMeta:
    """XGBoost meta-learner that combines strategy signals."""

    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self._model: XGBClassifier | None = None

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all signal generators and regime detector.
        Returns DataFrame with all feature columns ready for XGBoost.

        The returned DataFrame has the same index as the input but with
        additional columns for all signals, regime, and derived features.
        NaN rows (from indicator warmup) are NOT dropped here — caller decides.
        """
        # Layer 1: Generate all strategy signals
        out = compute_rsi_signals(df)
        out = _merge_new_columns(out, compute_trend_signals(df))
        out = _merge_new_columns(out, compute_momentum_signals(df))

        # Layer 2: Regime detection
        out = detect_regime(out)

        # Additional raw features
        out["ret_1d"] = out["close"].pct_change(1) * 100
        out["ret_5d"] = out["close"].pct_change(5) * 100
        out["ret_10d"] = out["close"].pct_change(10) * 100
        out["ret_20d"] = out["close"].pct_change(20) * 100

        # Bollinger Bands (20-day, 2 std)
        sma20 = out["close"].rolling(20).mean()
        std20 = out["close"].rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        out["bb_width"] = (bb_upper - bb_lower) / (sma20 + 1e-10)
        out["bb_pct"] = (out["close"] - bb_lower) / (bb_upper - bb_lower + 1e-10)

        # Volume ratio
        out["volume_ratio"] = out["volume"] / (out["volume"].rolling(20).mean() + 1e-10)

        # Strategy agreement features
        signals = out[["rsi_signal", "trend_signal", "mom_signal"]]
        out["signal_sum"] = signals.sum(axis=1)
        out["signal_consensus"] = (signals.abs().sum(axis=1) > 0).astype(int) * np.sign(out["signal_sum"])

        # Convert volatile bool to int for XGBoost
        out["volatile"] = out["volatile"].astype(int)

        return out

    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate binary labels: 1 = price goes UP, 0 = price goes DOWN.
        Based on forward return over `horizon_days`.

        Uses CLOSE-to-CLOSE return shifted by 1 day (we enter at next open,
        but approximate with next close since daily granularity).
        """
        horizon = self.config.horizon_days
        forward_ret = df["close"].pct_change(horizon).shift(-horizon)
        labels = (forward_ret > 0).astype(int)
        # Last `horizon` rows have no label (no future data)
        labels.iloc[-horizon:] = np.nan
        return labels

    def get_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract only the feature columns used by XGBoost."""
        available = [c for c in FEATURE_COLS if c in df.columns]
        return df[available]

    def train(self, features: pd.DataFrame, labels: pd.Series) -> XGBClassifier:
        """
        Train XGBoost on one walk-forward fold.
        Returns the trained model (does not store it — caller manages models per fold).
        """
        # Drop NaN rows
        valid = features.index.intersection(labels.dropna().index)
        X = features.loc[valid]
        y = labels.loc[valid]

        # Drop any remaining NaN features
        mask = X.notna().all(axis=1)
        X = X[mask]
        y = y[mask]

        if len(X) < 50:
            log.warning(f"Only {len(X)} training samples — too few for reliable model")
            return None

        cfg = self.config
        model = XGBClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            min_child_weight=cfg.min_child_weight,
            gamma=cfg.gamma,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            random_state=cfg.random_state,
            n_jobs=-1,
            tree_method="hist",
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
        )

        # Use last 20% of training data for early stopping
        split = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        return model

    def predict(
        self, model: XGBClassifier, features: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict direction and probability.

        Returns:
            directions: array of +1 (long) or -1 (short)
            probabilities: array of confidence values (0.5 = no edge, 1.0 = certain UP)
        """
        if model is None:
            return np.zeros(len(features)), np.full(len(features), 0.5)

        # Drop NaN rows, predict, map back
        valid_mask = features.notna().all(axis=1)
        proba = np.full(len(features), 0.5)
        directions = np.zeros(len(features), dtype=int)

        if valid_mask.sum() > 0:
            raw_proba = model.predict_proba(features[valid_mask])[:, 1]
            proba[valid_mask.values] = raw_proba

        # Direction: above 0.5 = long, below 0.5 = short
        directions[proba > 0.5] = 1
        directions[proba <= 0.5] = -1

        return directions, proba

    def compute_position_size(
        self, probability: float, regime_code: int, is_volatile: bool
    ) -> float:
        """
        Kelly-criterion position sizing, capped and regime-adjusted.

        Kelly: f* = (p * b - q) / b
        Where p = win prob, q = 1-p, b = payoff ratio (assumed 1.0 for simplicity)
        Half-Kelly for safety: f = f* / 2
        """
        cfg = self.config

        # No trade below minimum confidence
        if probability < cfg.min_confidence and probability > (1 - cfg.min_confidence):
            return 0.0

        # Effective win probability (for long: probability, for short: 1-probability)
        p = max(probability, 1 - probability)
        q = 1 - p

        # Payoff ratio (assume 1:1 for now — refined later with historical data)
        b = 1.0

        # Kelly fraction
        kelly = (p * b - q) / b
        half_kelly = kelly / 2

        # Cap at max position
        position = min(half_kelly, cfg.max_position_frac)

        # Reduce in volatile regime
        if is_volatile:
            position *= cfg.volatile_reduction

        return max(position, 0.0)


def _merge_new_columns(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Merge only columns from `other` that don't already exist in `base`."""
    new_cols = [c for c in other.columns if c not in base.columns]
    for col in new_cols:
        base[col] = other[col]
    return base
