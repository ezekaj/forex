"""
XGBoost trading strategy.

Uses XGBoost gradient boosting for BUY/HOLD/SELL predictions.
Generally outperforms Random Forest on imbalanced datasets.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from pathlib import Path
from datetime import datetime

from .base import BaseStrategy, Signal


class XGBoostStrategy(BaseStrategy):
    """XGBoost classifier for forex trading."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
        gamma: float = 0.1,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize XGBoost strategy.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Step size shrinkage (lower = more robust)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of features
            min_child_weight: Minimum sum of instance weight needed in a child
            gamma: Minimum loss reduction required for split
            scale_pos_weight: Balancing of positive/negative weights (auto if None)
            random_state: Random seed
            n_jobs: Number of parallel threads (-1 = use all cores)
        """
        super().__init__("XGBoost")

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            tree_method='hist',  # Faster for large datasets
            objective='multi:softprob',  # Multi-class probability
            eval_metric='mlogloss'
        )

        self.feature_names = None
        self.train_metrics = {}

    def train(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train XGBoost model.

        Args:
            features_df: DataFrame with features
            labels: Series with labels (1=BUY, 0=HOLD, -1=SELL)
            validation_split: Fraction of data for validation

        Returns:
            Dictionary with training metrics
        """
        # Remove NaN labels
        valid_mask = ~labels.isna()
        X = features_df[valid_mask]
        y = labels[valid_mask]

        # Select only numeric columns (exclude timestamp, any non-numeric)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols]

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Time series split (last portion for validation)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # Map labels to 0, 1, 2 for XGBoost (requires non-negative integers)
        label_map = {-1: 0, 0: 1, 1: 2}  # SELL=0, HOLD=1, BUY=2
        y_train_mapped = y_train.map(label_map)
        y_val_mapped = y_val.map(label_map)

        print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples...")
        print(f"Class distribution (train):")
        print(f"  BUY: {(y_train == Signal.BUY.value).sum()} ({(y_train == Signal.BUY.value).sum() / len(y_train) * 100:.1f}%)")
        print(f"  HOLD: {(y_train == Signal.HOLD.value).sum()} ({(y_train == Signal.HOLD.value).sum() / len(y_train) * 100:.1f}%)")
        print(f"  SELL: {(y_train == Signal.SELL.value).sum()} ({(y_train == Signal.SELL.value).sum() / len(y_train) * 100:.1f}%)")

        # Train model with early stopping (XGBoost 3.0+ API)
        self.model.fit(
            X_train,
            y_train_mapped,
            eval_set=[(X_val, y_val_mapped)],
            verbose=False
        )
        self.is_trained = True

        # Get best iteration
        best_iteration = self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.model.n_estimators
        print(f"Training stopped at iteration {best_iteration}")

        # Evaluate on training set
        y_train_pred_mapped = self.model.predict(X_train)
        y_train_pred = pd.Series(y_train_pred_mapped).map({0: -1, 1: 0, 2: 1})
        train_acc = accuracy_score(y_train, y_train_pred)

        # Evaluate on validation set
        y_val_pred_mapped = self.model.predict(X_val)
        y_val_pred = pd.Series(y_val_pred_mapped).map({0: -1, 1: 0, 2: 1})
        val_acc = accuracy_score(y_val, y_val_pred)

        # Detailed validation metrics
        print("\nValidation Set Performance:")
        print(classification_report(
            y_val,
            y_val_pred,
            target_names=['SELL', 'HOLD', 'BUY'],
            zero_division=0
        ))

        # Confusion matrix
        cm = confusion_matrix(y_val, y_val_pred, labels=[-1, 0, 1])
        print("\nConfusion Matrix:")
        print("                Predicted")
        print("              SELL  HOLD  BUY")
        print(f"Actual SELL   {cm[0, 0]:4d}  {cm[0, 1]:4d}  {cm[0, 2]:4d}")
        print(f"       HOLD   {cm[1, 0]:4d}  {cm[1, 1]:4d}  {cm[1, 2]:4d}")
        print(f"       BUY    {cm[2, 0]:4d}  {cm[2, 1]:4d}  {cm[2, 2]:4d}")

        # Store metrics
        self.train_metrics = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'n_features': len(self.feature_names),
            'best_iteration': int(best_iteration),
            'class_distribution': {
                'BUY': int((y_train == Signal.BUY.value).sum()),
                'HOLD': int((y_train == Signal.HOLD.value).sum()),
                'SELL': int((y_train == Signal.SELL.value).sum())
            }
        }

        self.metadata.update({
            'trained_at': datetime.utcnow().isoformat(),
            'n_features': len(self.feature_names),
            'model_params': self.model.get_params(),
            'best_iteration': int(best_iteration)
        })

        return self.train_metrics

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generate trading signals.

        Args:
            features: DataFrame with features

        Returns:
            Array of predictions (1=BUY, 0=HOLD, -1=SELL)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Ensure features are in correct order
        if self.feature_names:
            features = features[self.feature_names]

        # Get predictions (0, 1, 2) and map back to (-1, 0, 1)
        predictions_mapped = self.model.predict(features)
        predictions = np.array([{0: -1, 1: 0, 2: 1}[p] for p in predictions_mapped])
        return predictions

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            features: DataFrame with features

        Returns:
            Array of shape (n_samples, 3) with probabilities for [SELL, HOLD, BUY]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Ensure features are in correct order
        if self.feature_names:
            features = features[self.feature_names]

        probabilities = self.model.predict_proba(features)
        # Reorder from [SELL(0), HOLD(1), BUY(2)] to [SELL, HOLD, BUY]
        return probabilities

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores from XGBoost.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            return None

        importances = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importances))

        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        return feature_importance

    def save(self, path: str) -> None:
        """
        Save trained model to disk.

        Args:
            path: File path to save model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'metadata': self.get_metadata(),
            'train_metrics': self.train_metrics
        }

        joblib.dump(model_data, path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """
        Load trained model from disk.

        Args:
            path: File path to load model from
        """
        model_data = joblib.load(path)

        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.metadata = model_data['metadata']
        self.train_metrics = model_data.get('train_metrics', {})
        self.is_trained = True

        print(f"Model loaded from {path}")
        print(f"Trained at: {self.metadata.get('trained_at', 'Unknown')}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Best iteration: {self.train_metrics.get('best_iteration', 'N/A')}")

    def print_top_features(self, n: int = 20) -> None:
        """
        Print top N most important features.

        Args:
            n: Number of top features to print
        """
        feature_importance = self.get_feature_importance()

        if not feature_importance:
            print("Model not trained yet")
            return

        print(f"\nTop {n} Most Important Features:")
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:n], 1):
            print(f"{i:2d}. {feature:30s} {importance:.4f}")
