"""
Random Forest trading strategy.

Uses sklearn RandomForestClassifier for BUY/HOLD/SELL predictions.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from pathlib import Path
from datetime import datetime

from .base import BaseStrategy, Signal


class RandomForestStrategy(BaseStrategy):
    """Random Forest classifier for forex trading."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 20,
        min_samples_split: int = 10,
        min_samples_leaf: int = 4,
        max_features: str = 'sqrt',
        class_weight: str = 'balanced',
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize Random Forest strategy.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf
            max_features: Features to consider for split
            class_weight: Class weighting strategy
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 = use all cores)
        """
        super().__init__("RandomForest")

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs
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
        Train Random Forest model.

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

        # Detect binary vs ternary classification
        unique_classes = sorted(y_train.unique())
        is_binary = len(unique_classes) == 2
        
        print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples...")
        print(f"Classification mode: {'BINARY' if is_binary else 'TERNARY'}")
        print(f"Class distribution (train):")
        if is_binary:
            print(f"  BUY: {(y_train == Signal.BUY.value).sum()} ({(y_train == Signal.BUY.value).sum() / len(y_train) * 100:.1f}%)")
            print(f"  SELL: {(y_train == Signal.SELL.value).sum()} ({(y_train == Signal.SELL.value).sum() / len(y_train) * 100:.1f}%)")
        else:
            print(f"  BUY: {(y_train == Signal.BUY.value).sum()} ({(y_train == Signal.BUY.value).sum() / len(y_train) * 100:.1f}%)")
            print(f"  HOLD: {(y_train == Signal.HOLD.value).sum()} ({(y_train == Signal.HOLD.value).sum() / len(y_train) * 100:.1f}%)")
            print(f"  SELL: {(y_train == Signal.SELL.value).sum()} ({(y_train == Signal.SELL.value).sum() / len(y_train) * 100:.1f}%)")

        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate on training set
        y_train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)

        # Evaluate on validation set
        y_val_pred = self.model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)

        # Detailed validation metrics
        print("\nValidation Set Performance:")
        if is_binary:
            print(classification_report(
                y_val,
                y_val_pred,
                target_names=['SELL', 'BUY'],
                labels=[-1, 1],
                zero_division=0
            ))
            # Binary confusion matrix
            cm = confusion_matrix(y_val, y_val_pred, labels=[-1, 1])
            print("\nConfusion Matrix:")
            print("                Predicted")
            print("              SELL  BUY")
            print(f"Actual SELL   {cm[0, 0]:4d}  {cm[0, 1]:4d}")
            print(f"       BUY    {cm[1, 0]:4d}  {cm[1, 1]:4d}")
        else:
            print(classification_report(
                y_val,
                y_val_pred,
                target_names=['SELL', 'HOLD', 'BUY'],
                zero_division=0
            ))
            # Ternary confusion matrix
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
            'class_distribution': {
                'BUY': int((y_train == Signal.BUY.value).sum()),
                'HOLD': int((y_train == Signal.HOLD.value).sum()),
                'SELL': int((y_train == Signal.SELL.value).sum())
            }
        }

        self.metadata.update({
            'trained_at': datetime.utcnow().isoformat(),
            'n_features': len(self.feature_names),
            'model_params': self.model.get_params()
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

        predictions = self.model.predict(features)
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
        return probabilities

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.

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
