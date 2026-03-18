"""News feature extraction: sentiment scores, TF-IDF, velocity, momentum."""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from forex_system.training.config import TrainingConfig
from forex_system.training.models.flang_electra import FlangElectraSentiment


class NewsFeatureExtractor:
    """
    Extracts numeric features from news articles for ML training.

    Features:
    - FLANG-ELECTRA sentiment (positive/negative/neutral/score)
    - TF-IDF on article titles (top N features)
    - News velocity (article count per time window)
    - Sentiment momentum (rolling stats over multiple windows)
    - Negative news asymmetry weighting (1.5x for negative)
    """

    def __init__(
        self,
        config: TrainingConfig = None,
        tfidf_max_features: int = 200,
        sentiment_windows_hours: list[int] = None,
    ):
        self.config = config or TrainingConfig()
        self.tfidf_max_features = tfidf_max_features
        self.sentiment_windows = sentiment_windows_hours or [6, 12, 24, 48]
        self._scorer = None
        self._tfidf = None

    @property
    def scorer(self) -> FlangElectraSentiment:
        if self._scorer is None:
            self._scorer = FlangElectraSentiment()
        return self._scorer

    def extract_sentiment(self, df: pd.DataFrame, text_col: str = "title") -> pd.DataFrame:
        """Add FLANG-ELECTRA sentiment columns to DataFrame."""
        return self.scorer.score_dataframe(df, text_column=text_col)

    def extract_tfidf(
        self,
        texts: list[str],
        fit: bool = True,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Extract TF-IDF features from article titles.
        Returns (feature_matrix, feature_names).
        """
        clean_texts = [t if isinstance(t, str) else "" for t in texts]

        if fit or self._tfidf is None:
            self._tfidf = TfidfVectorizer(
                max_features=self.tfidf_max_features,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.95,
            )
            matrix = self._tfidf.fit_transform(clean_texts)
        else:
            matrix = self._tfidf.transform(clean_texts)

        feature_names = [f"tfidf_{name}" for name in self._tfidf.get_feature_names_out()]
        return matrix.toarray(), feature_names

    def compute_news_velocity(
        self,
        articles_df: pd.DataFrame,
        date_col: str = "entry_date",
        window_hours: int = 24,
    ) -> pd.Series:
        """
        Compute news velocity: number of articles in trailing window.
        High velocity signals market-moving events.
        """
        if articles_df.empty or date_col not in articles_df.columns:
            return pd.Series(0, index=articles_df.index)

        dates = pd.to_datetime(articles_df[date_col])
        velocity = pd.Series(0.0, index=articles_df.index)

        for i, dt in enumerate(dates):
            if pd.isna(dt):
                continue
            window_start = dt - pd.Timedelta(hours=window_hours)
            count = ((dates >= window_start) & (dates <= dt)).sum()
            velocity.iloc[i] = count

        return velocity

    def compute_sentiment_momentum(
        self,
        sentiment_scores: pd.Series,
        dates: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute rolling sentiment statistics over multiple windows.
        Returns DataFrame with mean, std, acceleration per window.
        """
        result = pd.DataFrame(index=sentiment_scores.index)

        for window in self.sentiment_windows:
            col_mean = f"sent_mean_{window}h"
            col_std = f"sent_std_{window}h"
            col_accel = f"sent_accel_{window}h"

            # Rolling window based on count (approximation)
            # For daily articles, window=24h ~ 1 day of articles
            approx_bars = max(1, window // 6)  # ~6 articles per period

            rolling_mean = sentiment_scores.rolling(approx_bars, min_periods=1).mean()
            rolling_std = sentiment_scores.rolling(approx_bars, min_periods=1).std().fillna(0)
            acceleration = rolling_mean.diff()

            result[col_mean] = rolling_mean
            result[col_std] = rolling_std
            result[col_accel] = acceleration.fillna(0)

        return result

    def build_news_feature_matrix(
        self,
        aligned_df: pd.DataFrame,
        include_tfidf: bool = True,
        include_momentum: bool = True,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Build complete news feature matrix from aligned news-price data.

        Input: DataFrame from NewsPriceAligner.build_training_dataset()
        Output: (features_df, feature_names)

        Features:
        - sent_positive, sent_negative, sent_neutral, sent_score (FLANG-ELECTRA)
        - news_velocity_24h (article count in trailing 24h)
        - negative_weight (1.5x for negative sentiment)
        - sent_mean_Xh, sent_std_Xh, sent_accel_Xh (momentum per window)
        - tfidf_* (top TF-IDF features from titles)
        """
        if aligned_df.empty:
            return pd.DataFrame(), []

        df = aligned_df.copy()
        feature_names = []

        # 1. FLANG-ELECTRA sentiment
        df = self.extract_sentiment(df, text_col="title")
        sent_cols = ["sent_positive", "sent_negative", "sent_neutral", "sent_score"]
        feature_names.extend(sent_cols)

        # 2. Negative asymmetry weight
        neg_weight = self.config.NEGATIVE_NEWS_WEIGHT
        df["negative_weight"] = np.where(
            df["sent_score"] < 0,
            neg_weight,
            1.0,
        )
        feature_names.append("negative_weight")

        # 3. News velocity
        df["news_velocity_24h"] = self.compute_news_velocity(df, window_hours=24)
        feature_names.append("news_velocity_24h")

        # 4. Sentiment momentum
        if include_momentum and "sent_score" in df.columns:
            dates = df.get("entry_date", df.get("article_date"))
            if dates is not None:
                momentum = self.compute_sentiment_momentum(df["sent_score"], dates)
                for col in momentum.columns:
                    df[col] = momentum[col].values
                    feature_names.append(col)

        # 5. TF-IDF
        if include_tfidf:
            titles = df["title"].fillna("").tolist()
            tfidf_matrix, tfidf_names = self.extract_tfidf(titles, fit=True)
            tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf_names, index=df.index)
            df = pd.concat([df, tfidf_df], axis=1)
            feature_names.extend(tfidf_names)

        features_df = df[feature_names].copy()
        features_df = features_df.fillna(0)

        return features_df, feature_names
