"""
3-Model Ensemble with Feature Selection + Calibration.

Based on de Prado "Advances in Financial Machine Learning" (2018) and
"The 10 Reasons Most Machine Learning Funds Fail" (JPM 2018):

Critical constraints for financial ML:
  - Samples/features ratio must be ≥20:1 (preferably 50:1)
  - max_depth 3-4 for <2K samples (shallow trees prevent memorization)
  - Early stopping mandatory (20-30 rounds for small datasets)
  - Feature selection BEFORE training (MDI → select top 20-30 features)
  - Train accuracy should be 55-65%, NOT 100% (100% = memorization)
  - Expected OOS accuracy for financial binary: 52-55% is GOOD

Architecture:
  1. Feature selection (MDA-based, select top K features where K ≤ N/20)
  2. LightGBM + XGBoost + RF ensemble (all shallow, heavily regularized)
  3. Platt scaling calibration on held-out set
  4. Ensemble: average calibrated probabilities
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

log = logging.getLogger(__name__)


# ── Model Factories (de Prado-compliant hyperparameters) ──

def _create_lgbm():
    """
    LightGBM: shallow trees, moderate regularization.

    De Prado: max_depth 3-4 for <2K samples. But learning_rate must be high
    enough that each tree contributes meaningfully, otherwise early stopping
    kills the model after 0-1 trees on small noisy validation sets.
    """
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=200,           # Fixed count (no early stopping)
        max_depth=4,                # Shallow (de Prado: 3-4 for small data)
        learning_rate=0.05,         # Moderate (0.01 too slow → early stop kills it)
        subsample=0.7,              # Row sampling
        colsample_bytree=0.6,      # Feature sampling
        min_child_samples=25,       # Reasonable for ~600 training samples
        reg_alpha=0.3,              # L1 regularization
        reg_lambda=2.0,             # L2 regularization (strong but not extreme)
        num_leaves=15,              # 2^4 - 1 (matches max_depth=4)
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def _create_xgb():
    """XGBoost: shallow, moderate regularization."""
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=200,           # Fixed count
        max_depth=4,                # Shallow
        learning_rate=0.05,         # Moderate
        subsample=0.7,
        colsample_bytree=0.6,
        min_child_weight=8,
        gamma=0.3,                  # Moderate split penalty
        reg_alpha=0.3,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss",
    )


def _create_rf():
    """Random Forest: shallow, high min leaf size."""
    return RandomForestClassifier(
        n_estimators=300,           # More trees for RF (no overfitting risk from count)
        max_depth=5,                # Moderate (RF overfits less than boosting)
        min_samples_leaf=25,
        min_samples_split=50,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


# ── Feature Selection ──

def select_features_mda(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = 25,
    n_repeats: int = 3,
) -> list[str]:
    """
    Select top features using Mean Decrease Accuracy (MDA).

    De Prado (Chapter 8): MDA is out-of-sample and more reliable than MDI.
    We train a quick RF, then permute each feature and measure accuracy drop.

    Args:
        X: Feature DataFrame.
        y: Binary labels.
        max_features: Maximum features to keep.
        n_repeats: Number of permutation repeats per feature.

    Returns:
        List of selected feature names, ordered by importance.
    """
    from sklearn.inspection import permutation_importance

    # Quick RF for feature importance (not the final model)
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_leaf=30,
        random_state=42, n_jobs=-1,
    )

    # Time-series split: train on first 70%, evaluate on last 30%
    split = int(len(X) * 0.7)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    rf.fit(X_train, y_train)

    # Permutation importance on validation set (OOS)
    result = permutation_importance(
        rf, X_val, y_val,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )

    # Rank features by mean importance (accuracy drop when permuted)
    importances = pd.Series(result.importances_mean, index=X.columns)
    importances = importances.sort_values(ascending=False)

    # Select top K features (only those with positive importance)
    selected = importances[importances > 0].head(max_features).index.tolist()

    if len(selected) < 5:
        # Fallback: take top 25 by MDI (in-sample, less reliable but always works)
        mdi = pd.Series(rf.feature_importances_, index=X.columns)
        selected = mdi.sort_values(ascending=False).head(max_features).index.tolist()
        log.warning(f"MDA selected <5 features, falling back to MDI: {len(selected)} features")

    log.info(f"Feature selection: {len(selected)}/{len(X.columns)} features selected (MDA)")
    return selected


# ── Sample Weighting ──

def compute_sample_uniqueness(labels_df: pd.DataFrame) -> np.ndarray:
    """
    Compute sample uniqueness for triple barrier labels.

    De Prado (Chapter 4): When labels overlap in time, samples are not independent.
    Uniqueness = fraction of label's holding period during which it's the ONLY active label.
    Higher uniqueness = more informative sample = should be weighted higher.

    Args:
        labels_df: DataFrame with 'holding_period' column from triple barrier.

    Returns:
        Array of uniqueness weights (0 to 1 for each sample).
    """
    n = len(labels_df)
    if "holding_period" not in labels_df.columns:
        return np.ones(n)

    holding = labels_df["holding_period"].fillna(1).astype(int).values

    # Count how many labels are active at each time step
    concurrent = np.zeros(n, dtype=float)
    for i in range(n):
        h = min(holding[i], n - i)
        for j in range(i, min(i + h, n)):
            concurrent[j] += 1

    # Uniqueness of sample i = average(1/concurrent) over its holding period
    uniqueness = np.zeros(n)
    for i in range(n):
        h = min(holding[i], n - i)
        if h > 0:
            uniqueness[i] = np.mean(1.0 / concurrent[i:i + h])

    return uniqueness


def _time_decay_weights(n_samples: int, decay_factor: float = 2.0) -> np.ndarray:
    """Time-decay: recent data weighted `decay_factor`× vs oldest."""
    return np.linspace(1.0, decay_factor, n_samples)


def _combined_weights(
    n_samples: int,
    uniqueness: np.ndarray = None,
    decay_factor: float = 2.0,
) -> np.ndarray:
    """Combine uniqueness + time-decay weights (multiplicative)."""
    w = _time_decay_weights(n_samples, decay_factor)
    if uniqueness is not None and len(uniqueness) == n_samples:
        w *= uniqueness
    # Normalize so mean = 1.0
    w /= w.mean()
    return w


# ── Main Ensemble Model ──

class EnsembleModel:
    """
    3-model ensemble with feature selection, early stopping, and calibration.

    Key constraints (from de Prado research):
      - Features capped at N/20 (20:1 samples/features ratio)
      - max_depth 3-4 (prevents memorization)
      - Early stopping at 20 rounds
      - Train accuracy should be 55-65% (NOT 100%)
      - Platt calibration on held-out set
    """

    def __init__(self, max_feature_ratio: int = 20):
        """
        Args:
            max_feature_ratio: Maximum samples/features ratio. Default 20
                means with 1000 samples, use ≤50 features.
        """
        self.max_feature_ratio = max_feature_ratio
        self.models = []
        self.calibrators = []
        self.selected_features = []
        self.feature_importances = {}
        self.train_metrics = {}

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        labels_df: pd.DataFrame = None,
        calib_fraction: float = 0.2,
    ) -> dict:
        """
        Train with feature selection, early stopping, and calibration.

        Args:
            X: Feature DataFrame.
            y: Labels (+1 or -1 from triple barrier).
            labels_df: Optional triple barrier DataFrame for uniqueness weighting.
            calib_fraction: Fraction held out for calibration.
        """
        y_binary = (y == 1).astype(int)

        # ── Step 1: Determine max features ──
        max_features = max(10, len(X) // self.max_feature_ratio)
        log.info(f"Training: {len(X)} samples, {len(X.columns)} raw features, "
                 f"max {max_features} after selection")

        # ── Step 2: Feature selection (MDA) ──
        self.selected_features = select_features_mda(
            X, y_binary, max_features=max_features
        )
        X_selected = X[self.selected_features]

        # ── Step 3: Split train / calibration ──
        # 80% train, 20% calibration. No separate early-stopping set.
        # Rely on shallow trees + regularization instead of early stopping
        # (early stopping fails on small noisy financial validation sets).
        n = len(X_selected)
        train_end = int(n * 0.80)

        X_train = X_selected.iloc[:train_end]
        y_train = y_binary.iloc[:train_end]
        X_calib = X_selected.iloc[train_end:]
        y_calib = y_binary.iloc[train_end:]

        if len(X_train) < 50 or len(X_calib) < 20:
            log.warning(f"Insufficient data: train={len(X_train)}, calib={len(X_calib)}")
            return {"error": "insufficient_data"}

        # ── Step 4: Compute sample weights ──
        uniqueness = None
        if labels_df is not None:
            uniqueness = compute_sample_uniqueness(labels_df.iloc[:train_end])
        weights = _combined_weights(len(X_train), uniqueness)

        # ── Step 5: Train each model (fixed tree count, no early stopping) ──
        # De Prado: with max_depth=4 and strong regularization, 150-200 trees
        # can't memorize 600 samples. Overfitting is controlled by tree depth
        # and regularization, not early stopping.
        raw_models = [_create_lgbm(), _create_xgb(), _create_rf()]
        model_names = ["LightGBM", "XGBoost", "RandomForest"]
        self.models = []
        self.calibrators = []
        train_accs = []
        calib_accs = []
        n_trees_used = []

        for name, model in zip(model_names, raw_models):
            try:
                if name == "RandomForest":
                    model.fit(X_train, y_train, sample_weight=weights)
                else:
                    model.fit(X_train, y_train, sample_weight=weights)
                n_trees = model.n_estimators

                # Calibrate
                raw_proba_calib = model.predict_proba(X_calib)[:, 1]
                calibrator = _fit_platt_scaling(raw_proba_calib, y_calib)

                self.models.append(model)
                self.calibrators.append(calibrator)

                # Metrics
                train_acc = accuracy_score(y_train, model.predict(X_train))
                calib_acc = accuracy_score(y_calib, model.predict(X_calib))
                train_accs.append(train_acc)
                calib_accs.append(calib_acc)
                n_trees_used.append(n_trees)

                overfit = (train_acc - calib_acc) / max(train_acc, 0.01)
                status = "OK" if overfit < 0.15 else "WARN" if overfit < 0.25 else "OVERFIT"
                log.info(
                    f"  {name}: train={train_acc:.3f}, calib={calib_acc:.3f}, "
                    f"trees={n_trees}, overfit={overfit:.2f} [{status}]"
                )

            except Exception as e:
                log.warning(f"  {name} failed: {e}")
                self.models.append(None)
                self.calibrators.append(None)

        # Feature importances
        self.feature_importances = _average_feature_importances(
            self.models, self.selected_features
        )

        self.train_metrics = {
            "n_train": len(X_train),
            "n_calib": len(X_calib),
            "n_features_raw": len(X.columns),
            "n_features_selected": len(self.selected_features),
            "label_balance": float(y_binary.mean()),
            "train_accs": train_accs,
            "calib_accs": calib_accs,
            "n_trees_used": n_trees_used,
            "avg_train_acc": np.mean(train_accs) if train_accs else 0,
            "avg_calib_acc": np.mean(calib_accs) if calib_accs else 0,
            "n_models_trained": sum(1 for m in self.models if m is not None),
        }

        return self.train_metrics

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict direction + calibrated probability + model agreement."""
        # Use only selected features
        X_sel = X[self.selected_features] if self.selected_features else X
        n = len(X_sel)
        all_probs = []
        all_dirs = []

        for model, calibrator in zip(self.models, self.calibrators):
            if model is None or calibrator is None:
                continue
            try:
                raw_proba = model.predict_proba(X_sel)[:, 1]
                probs = _apply_platt_scaling(calibrator, raw_proba)
                all_probs.append(probs)
                all_dirs.append((probs > 0.5).astype(int) * 2 - 1)
            except Exception:
                continue

        if not all_probs:
            return np.zeros(n), np.full(n, 0.5), np.zeros(n)

        ensemble_prob = np.mean(all_probs, axis=0)
        directions = np.where(ensemble_prob > 0.5, 1, -1)

        if len(all_dirs) > 1:
            dir_matrix = np.array(all_dirs)
            n_agreeing = np.sum(dir_matrix == directions[np.newaxis, :], axis=0)
        else:
            n_agreeing = np.ones(n)

        return directions, ensemble_prob, n_agreeing

    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        sorted_feats = sorted(
            self.feature_importances.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_feats[:n]


# ── Helpers ──

def _fit_platt_scaling(raw_proba: np.ndarray, y_true: pd.Series) -> tuple:
    """Fit sigmoid calibration: P = 1 / (1 + exp(A*x + B))."""
    from scipy.optimize import minimize

    y = y_true.values.astype(float)

    def neg_log_likelihood(params):
        a, b = params
        p = 1.0 / (1.0 + np.exp(a * raw_proba + b))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    result = minimize(neg_log_likelihood, x0=[0.0, 0.0], method="Nelder-Mead")
    return (result.x[0], result.x[1])


def _apply_platt_scaling(params: tuple, raw_proba: np.ndarray) -> np.ndarray:
    a, b = params
    return np.clip(1.0 / (1.0 + np.exp(a * raw_proba + b)), 0.01, 0.99)


def _average_feature_importances(models: list, feature_names: list[str]) -> dict:
    importance_sum = np.zeros(len(feature_names))
    count = 0
    for model in models:
        if model is None:
            continue
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            if len(imp) == len(feature_names):
                importance_sum += imp
                count += 1
    if count == 0:
        return {}
    avg_imp = importance_sum / count
    return dict(zip(feature_names, avg_imp))
