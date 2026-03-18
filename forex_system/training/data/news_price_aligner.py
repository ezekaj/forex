"""Align news articles to price bars and compute forward returns."""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from forex_system.training.config import TrainingConfig, ASSET_REGISTRY, get_asset
from forex_system.training.data.price_loader import UniversalPriceLoader
from forex_system.training.data.news_loader import NewsLoader


class NewsPriceAligner:
    """
    Joins news articles to concurrent price bars for ANY asset class.
    For each article, finds the nearest trading price bar and computes
    forward returns at 2d, 3d, 5d windows.
    """

    def __init__(
        self,
        price_loader: UniversalPriceLoader,
        news_loader: NewsLoader,
        config: TrainingConfig = None,
    ):
        self.price_loader = price_loader
        self.news_loader = news_loader
        self.config = config or TrainingConfig()

    def align(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Align news articles to daily price bars for any asset.
        Uses the asset registry to find relevant news keywords.
        """
        # Load daily prices (routes to correct source automatically)
        prices = self.price_loader.load_ohlcv(symbol, "1d")
        if prices.empty:
            return pd.DataFrame()

        # Get asset-specific keywords from registry
        asset = get_asset(symbol)
        news = self.news_loader.load_for_asset(
            keywords=asset.keywords,
            start_date=start_date,
            end_date=end_date,
        )
        if news.empty:
            return pd.DataFrame()

        close = prices["close"]
        results = []

        for _, article in news.iterrows():
            article_date = article["date"]
            if pd.isna(article_date):
                continue

            # Find nearest price bar on or after article date
            valid_dates = close.index[close.index >= article_date]
            if valid_dates.empty:
                continue
            entry_date = valid_dates[0]
            entry_price = close[entry_date]

            row = {
                "article_id": article.get("id"),
                "title": article.get("title", ""),
                "source": article.get("source", ""),
                "article_date": article_date,
                "entry_date": entry_date,
                "entry_price": entry_price,
                "symbol": symbol,
            }

            # Compute forward returns
            future_bars = close.index[close.index > entry_date]
            best_abs_return = 0
            best_window = None

            for window in self.config.NEWS_FORWARD_WINDOWS:
                col = f"return_{window}d"
                if len(future_bars) >= window:
                    exit_price = close[future_bars[window - 1]]
                    fwd_return = exit_price / entry_price - 1.0
                    row[col] = fwd_return
                    if abs(fwd_return) > best_abs_return:
                        best_abs_return = abs(fwd_return)
                        best_window = window
                else:
                    row[col] = np.nan

            if best_window is not None:
                row["best_window"] = best_window
                row["label"] = 1 if row[f"return_{best_window}d"] > 0 else -1
            else:
                row["best_window"] = np.nan
                row["label"] = np.nan

            results.append(row)

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        return result_df

    def build_training_dataset(
        self,
        symbols: list[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Build aligned dataset across multiple symbols."""
        if symbols is None:
            symbols = list(self.config.MAJOR_PAIRS)

        all_aligned = []
        for symbol in symbols:
            aligned = self.align(symbol, start_date, end_date)
            if not aligned.empty:
                all_aligned.append(aligned)

        if not all_aligned:
            return pd.DataFrame()
        return pd.concat(all_aligned, ignore_index=True)

    def get_alignment_stats(self, aligned_df: pd.DataFrame) -> dict:
        """Return alignment statistics."""
        if aligned_df.empty:
            return {"total": 0}

        label_counts = aligned_df["label"].value_counts().to_dict()
        return {
            "total_aligned": len(aligned_df),
            "symbols": aligned_df["symbol"].nunique(),
            "label_distribution": label_counts,
            "label_balance": label_counts.get(1, 0) / max(len(aligned_df), 1),
            "date_range": (
                str(aligned_df["article_date"].min()),
                str(aligned_df["article_date"].max()),
            ),
            "avg_return_2d": aligned_df.get("return_2d", pd.Series()).mean(),
            "avg_return_5d": aligned_df.get("return_5d", pd.Series()).mean(),
        }
