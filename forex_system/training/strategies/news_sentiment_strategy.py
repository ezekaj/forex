"""Phase 1 strategy: trade based on news sentiment features only."""

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from forex_system.training.config import TrainingConfig


class NewsSentimentStrategy:
    """
    XGBoost classifier trained exclusively on news-derived features.
    Predicts BUY (1) / SELL (-1) based on sentiment, velocity, momentum.
    """

    def __init__(
        self,
        n_estimators: int = None,
        max_depth: int = None,
        learning_rate: float = None,
        subsample: float = None,
        colsample_bytree: float = None,
        negative_weight: float = 1.5,
        random_state: int = 42,
        config: TrainingConfig = None,
    ):
        self.config = config or TrainingConfig()
        self.negative_weight = negative_weight

        self.model = XGBClassifier(
            n_estimators=n_estimators or self.config.XGBOOST_N_ESTIMATORS,
            max_depth=max_depth or self.config.XGBOOST_MAX_DEPTH,
            learning_rate=learning_rate or self.config.XGBOOST_LEARNING_RATE,
            subsample=subsample or self.config.XGBOOST_SUBSAMPLE,
            colsample_bytree=colsample_bytree or self.config.XGBOOST_COLSAMPLE_BYTREE,
            random_state=random_state,
            use_label_encoder=False,
            tree_method="hist",
            device="cpu",
        )
        self.feature_names: list[str] = []
        self.is_trained = False

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        val_features: Optional[pd.DataFrame] = None,
        val_labels: Optional[pd.Series] = None,
    ) -> dict[str, Any]:
        """
        Train on news features.
        Labels: 1=BUY, -1=SELL (binary).
        Applies asymmetric sample weighting for negative sentiment.
        """
        self.feature_names = list(features.columns)

        # Remap labels: -1 → 0, 1 → 1 (XGBoost needs 0-indexed)
        y_train = ((labels + 1) / 2).astype(int)

        # Sample weights: boost negative-sentiment samples
        sample_weight = np.ones(len(features))
        if "sent_score" in features.columns:
            negative_mask = features["sent_score"] < 0
            sample_weight[negative_mask] = self.negative_weight
        elif "negative_weight" in features.columns:
            sample_weight = features["negative_weight"].values

        fit_params = {"sample_weight": sample_weight}

        if val_features is not None and val_labels is not None:
            y_val = ((val_labels + 1) / 2).astype(int)
            self.model.fit(
                features, y_train,
                eval_set=[(val_features, y_val)],
                verbose=False,
                **fit_params,
            )
        else:
            self.model.fit(features, y_train, verbose=False, **fit_params)

        self.is_trained = True

        # Compute training metrics
        train_pred = self.model.predict(features)
        metrics = {
            "train_accuracy": float(accuracy_score(y_train, train_pred)),
            "train_samples": len(features),
            "features_used": len(self.feature_names),
        }

        if val_features is not None and val_labels is not None:
            val_pred = self.model.predict(val_features)
            metrics["val_accuracy"] = float(accuracy_score(y_val, val_pred))
            metrics["val_samples"] = len(val_features)
            metrics["val_report"] = classification_report(
                y_val, val_pred, target_names=["SELL", "BUY"], output_dict=True
            )

        return metrics

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict BUY (1) / SELL (-1)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        pred = self.model.predict(features)
        # Map back: 0 → -1, 1 → 1
        return (pred * 2 - 1).astype(int)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Get probability of BUY class."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict_proba(features)[:, 1]

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importances sorted by value."""
        if not self.is_trained:
            return {}
        importances = self.model.feature_importances_
        result = dict(zip(self.feature_names, importances))
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def get_hyperparams(self) -> dict:
        return {
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "learning_rate": self.model.learning_rate,
            "subsample": self.model.subsample,
            "colsample_bytree": self.model.colsample_bytree,
            "negative_weight": self.negative_weight,
        }

    def save(self, path: str):
        """Save model + metadata."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "hyperparams": self.get_hyperparams(),
        }, path)

    def load(self, path: str):
        """Load model + metadata."""
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.is_trained = True
