"""
Hybrid ML + LLM trading strategy.

Combines traditional ML (Random Forest/XGBoost) with LLM review:
1. ML model generates candidate signals (fast, systematic)
2. LLM reviews signals using news context (smart, catches Black Swans)
3. LLM approves/rejects/modifies based on fundamentals

Inspired by AI-Trader's agent-based approach, adapted for forex.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base import BaseStrategy, Signal
from ..services.news_service import NewsService
from ..services.llm_service import LLMService, SignalReview


class HybridLLMStrategy(BaseStrategy):
    """
    Hybrid strategy: ML generates signals, LLM reviews them.

    Architecture:
    - Wraps any existing BaseStrategy (RF, XGBoost, etc.)
    - Adds news sentiment as additional features
    - Uses LLM to review ML predictions before execution
    - Tracks costs and performance metrics
    """

    def __init__(
        self,
        base_strategy: BaseStrategy,
        news_service: Optional[NewsService] = None,
        llm_service: Optional[LLMService] = None,
        enable_llm_review: bool = True,
        min_ml_confidence: float = 0.55,
        min_llm_confidence: float = 0.60,
        pair: str = 'EURUSD'
    ):
        """
        Initialize Hybrid LLM Strategy.

        Args:
            base_strategy: Underlying ML strategy (RF, XGBoost, etc.)
            news_service: NewsService instance (creates if None)
            llm_service: LLMService instance (creates if None)
            enable_llm_review: Whether to use LLM review (disable for baseline comparison)
            min_ml_confidence: Minimum ML confidence to generate signal
            min_llm_confidence: Minimum LLM confidence to approve signal
            pair: Currency pair (for news fetching)
        """
        super().__init__(f"Hybrid_{base_strategy.name}_LLM")

        self.base_strategy = base_strategy
        self.news_service = news_service or NewsService()
        self.llm_service = llm_service or LLMService(provider='anthropic')
        self.enable_llm_review = enable_llm_review
        self.min_ml_confidence = min_ml_confidence
        self.min_llm_confidence = min_llm_confidence
        self.pair = pair

        # Performance tracking
        self.review_stats = {
            'total_signals': 0,
            'approved': 0,
            'rejected': 0,
            'modified': 0,
            'total_cost_usd': 0.0,
            'reviews': []  # Store all reviews for analysis
        }

    def train(self, features_df: pd.DataFrame, labels: pd.Series) -> Dict[str, Any]:
        """
        Train the hybrid strategy.

        Adds news sentiment features, then trains base ML model.

        Args:
            features_df: DataFrame with technical features
            labels: Series with labels

        Returns:
            Training metrics
        """
        print(f"\n[{self.name}] Training hybrid strategy...")

        # Add news features to training data
        print("[Hybrid] Adding news sentiment features...")
        features_with_news = self._add_news_features(features_df)

        # Train base ML model on augmented features
        print(f"[Hybrid] Training base strategy ({self.base_strategy.name})...")
        metrics = self.base_strategy.train(features_with_news, labels)

        self.is_trained = True
        self.metadata = {
            'base_strategy': self.base_strategy.name,
            'training_metrics': metrics,
            'news_features_added': True,
            'llm_review_enabled': self.enable_llm_review
        }

        return metrics

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generate trading signals with LLM review.

        Flow:
        1. Add news sentiment features
        2. Get ML prediction + confidence
        3. If enable_llm_review: Review with LLM
        4. Return approved/modified signals

        Args:
            features: DataFrame with features

        Returns:
            Array of predictions (Signal values)
        """
        if not self.is_trained:
            raise ValueError("Strategy not trained. Call train() first.")

        # Add news features
        features_with_news = self._add_news_features(features)

        # Get ML predictions and probabilities
        ml_predictions = self.base_strategy.predict(features_with_news)
        ml_probabilities = self.base_strategy.predict_proba(features_with_news)

        # If LLM review disabled, return ML predictions
        if not self.enable_llm_review:
            return ml_predictions

        # LLM review for each prediction
        final_predictions = []
        for i in range(len(features)):
            ml_pred = ml_predictions[i]
            ml_proba = ml_probabilities[i]

            # Calculate ML confidence
            ml_confidence = np.max(ml_proba)

            # Filter low-confidence ML signals
            if ml_confidence < self.min_ml_confidence:
                final_predictions.append(Signal.HOLD.value)
                continue

            # Get market context for LLM
            try:
                row = features.iloc[i]
                timestamp = row.get('timestamp', datetime.utcnow())
                if isinstance(timestamp, str):
                    timestamp = pd.to_datetime(timestamp)

                market_context = self.news_service.get_market_context(self.pair, timestamp)
                current_price = row.get('close', row.get('price', 1.0))

                # Build signal dict for LLM
                signal_dict = {
                    'direction': self._signal_to_str(ml_pred),
                    'confidence': float(ml_confidence),
                    'features_summary': self._extract_key_features(row)
                }

                # LLM review
                review = self.llm_service.review_signal(
                    signal=signal_dict,
                    market_context=market_context,
                    price=current_price,
                    pair=self.pair
                )

                # Track review stats
                self._update_review_stats(review, ml_pred)

                # Apply LLM decision
                final_pred = self._apply_llm_decision(ml_pred, review)
                final_predictions.append(final_pred)

            except Exception as e:
                print(f"[Hybrid] LLM review error: {str(e)}")
                # On error, use ML prediction (fail-safe)
                final_predictions.append(ml_pred)

        return np.array(final_predictions)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities (from base ML model).

        Args:
            features: DataFrame with features

        Returns:
            Probability array
        """
        features_with_news = self._add_news_features(features)
        return self.base_strategy.predict_proba(features_with_news)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from base ML model.

        Returns:
            Feature importance dict
        """
        return self.base_strategy.get_feature_importance()

    def _add_news_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add news sentiment features to existing technical features.

        Adds 3 new features:
        - base_currency_sentiment: Sentiment for base currency (-1 to 1)
        - quote_currency_sentiment: Sentiment for quote currency (-1 to 1)
        - news_volume_24h: Number of news articles in last 24 hours

        Args:
            features_df: DataFrame with technical features

        Returns:
            DataFrame with news features added
        """
        df = features_df.copy()

        # Initialize news features
        df['base_currency_sentiment'] = 0.0
        df['quote_currency_sentiment'] = 0.0
        df['news_volume_24h'] = 0

        # Extract currencies
        base_currency = self.pair[:3]
        quote_currency = self.pair[3:6]

        # For each row, fetch news sentiment
        # Note: In production, cache this to avoid API rate limits
        for idx in df.index:
            try:
                row = df.loc[idx]
                timestamp = row.get('timestamp', datetime.utcnow())
                if isinstance(timestamp, str):
                    timestamp = pd.to_datetime(timestamp)

                # Get central bank sentiment (7-day lookback)
                base_sentiment_data = self.news_service.get_central_bank_sentiment(
                    base_currency, timestamp, lookback_days=7
                )
                quote_sentiment_data = self.news_service.get_central_bank_sentiment(
                    quote_currency, timestamp, lookback_days=7
                )

                df.at[idx, 'base_currency_sentiment'] = base_sentiment_data['sentiment']
                df.at[idx, 'quote_currency_sentiment'] = quote_sentiment_data['sentiment']

                # Get news volume (24h lookback)
                news_volume = self.news_service.get_news_volume(self.pair, timestamp, lookback_hours=24)
                df.at[idx, 'news_volume_24h'] = news_volume

            except Exception as e:
                print(f"[Hybrid] Error fetching news for row {idx}: {str(e)}")
                # Keep default values (0.0, 0.0, 0)
                continue

        return df

    def _signal_to_str(self, signal: int) -> str:
        """Convert signal value to string."""
        if signal == Signal.BUY.value:
            return 'BUY'
        elif signal == Signal.SELL.value:
            return 'SELL'
        else:
            return 'HOLD'

    def _extract_key_features(self, row: pd.Series) -> Dict[str, float]:
        """
        Extract key features from row for LLM context.

        Args:
            row: Feature row

        Returns:
            Dict of key feature values
        """
        # Select most important technical indicators for LLM context
        key_features = {}

        # Trend indicators
        if 'SMA_20' in row:
            key_features['SMA_20'] = row['SMA_20']
        if 'SMA_50' in row:
            key_features['SMA_50'] = row['SMA_50']
        if 'MACD' in row:
            key_features['MACD'] = row['MACD']

        # Momentum indicators
        if 'RSI_14' in row:
            key_features['RSI_14'] = row['RSI_14']
        if 'ADX_14' in row:
            key_features['ADX_14'] = row['ADX_14']

        # Volatility
        if 'ATR_14' in row:
            key_features['ATR_14'] = row['ATR_14']

        # News features (if added)
        if 'base_currency_sentiment' in row:
            key_features['base_currency_sentiment'] = row['base_currency_sentiment']
        if 'quote_currency_sentiment' in row:
            key_features['quote_currency_sentiment'] = row['quote_currency_sentiment']
        if 'news_volume_24h' in row:
            key_features['news_volume_24h'] = row['news_volume_24h']

        return key_features

    def _apply_llm_decision(self, ml_prediction: int, review: SignalReview) -> int:
        """
        Apply LLM's decision to ML prediction.

        Args:
            ml_prediction: ML model's signal
            review: LLM's review

        Returns:
            Final signal value
        """
        # Check LLM confidence threshold
        if review.confidence < self.min_llm_confidence:
            # Low LLM confidence -> HOLD (reject signal)
            return Signal.HOLD.value

        # Apply LLM decision
        if review.decision == 'APPROVE':
            return ml_prediction
        elif review.decision == 'REJECT':
            return Signal.HOLD.value
        elif review.decision == 'MODIFY':
            # For now, MODIFY means reduce position (convert to HOLD if very uncertain)
            if review.suggested_position_size and review.suggested_position_size < 0.5:
                return Signal.HOLD.value
            else:
                return ml_prediction
        else:
            # Unknown decision -> HOLD (safe default)
            return Signal.HOLD.value

    def _update_review_stats(self, review: SignalReview, ml_prediction: int) -> None:
        """
        Update review statistics.

        Args:
            review: LLM review
            ml_prediction: ML prediction
        """
        self.review_stats['total_signals'] += 1
        self.review_stats['total_cost_usd'] += review.cost_usd

        if review.decision == 'APPROVE':
            self.review_stats['approved'] += 1
        elif review.decision == 'REJECT':
            self.review_stats['rejected'] += 1
        elif review.decision == 'MODIFY':
            self.review_stats['modified'] += 1

        # Store review for analysis
        self.review_stats['reviews'].append({
            'ml_prediction': self._signal_to_str(ml_prediction),
            'llm_decision': review.decision,
            'llm_confidence': review.confidence,
            'reasoning': review.reasoning,
            'cost_usd': review.cost_usd
        })

    def get_review_stats(self) -> Dict[str, Any]:
        """
        Get LLM review statistics.

        Returns:
            Dict with review stats
        """
        stats = self.review_stats.copy()

        # Calculate approval rate
        if stats['total_signals'] > 0:
            stats['approval_rate'] = stats['approved'] / stats['total_signals']
            stats['rejection_rate'] = stats['rejected'] / stats['total_signals']
            stats['modification_rate'] = stats['modified'] / stats['total_signals']
            stats['avg_cost_per_signal'] = stats['total_cost_usd'] / stats['total_signals']
        else:
            stats['approval_rate'] = 0.0
            stats['rejection_rate'] = 0.0
            stats['modification_rate'] = 0.0
            stats['avg_cost_per_signal'] = 0.0

        return stats

    def save_review_log(self, filepath: str) -> None:
        """
        Save review log to file for analysis.

        Args:
            filepath: Path to save log
        """
        import json

        with open(filepath, 'w') as f:
            json.dump(self.review_stats, f, indent=2)

        print(f"[Hybrid] Review log saved to {filepath}")
