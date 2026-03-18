"""News Analyst Agent — scores financial news with DistilRoBERTa sentiment model."""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from forex_system.agents.base_agent import BaseAgent, AgentResult
from forex_system.models.sentiment import SentimentScorer
from forex_system.training.config import ASSET_REGISTRY, get_asset
from forex_system.training.data.news_loader import NewsLoader

log = logging.getLogger(__name__)


class NewsAnalyst(BaseAgent):
    """
    Scores news articles with DistilRoBERTa (98.23% accuracy).
    For each asset, finds relevant articles and returns aggregated sentiment.

    No LLM needed — uses a specialized BERT model for classification.
    """

    def __init__(self, news_db_path: str = None):
        super().__init__(name="news_analyst")
        from forex_system.training.config import TrainingConfig
        config = TrainingConfig()
        self.news_loader = NewsLoader(news_db_path or config.NEWS_DB_PATH)
        self.scorer = SentimentScorer()

    async def analyze_asset(
        self,
        symbol: str,
        date: str,
        lookback_days: int = 2,
    ) -> AgentResult:
        """
        Get news sentiment for a specific asset on a specific date.

        Returns AgentResult with:
        - article_count: number of articles found
        - avg_sentiment: average sentiment score (-1 to +1)
        - strong_positive: count of articles with sentiment > 0.7
        - strong_negative: count of articles with sentiment < -0.3
        - dominant_label: most common label (positive/negative/neutral)
        - articles: list of top articles with scores
        - event_types: detected event types
        """
        import time
        start = time.perf_counter()

        asset = get_asset(symbol)
        end_date = date
        dt = datetime.strptime(date, "%Y-%m-%d")
        start_date = (dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        # Load articles matching this asset's keywords
        articles = self.news_loader.load_for_asset(
            keywords=asset.keywords,
            start_date=start_date,
            end_date=end_date,
        )

        if articles.empty:
            return AgentResult(
                agent=self.name,
                asset=symbol,
                timestamp=date,
                data={
                    "article_count": 0,
                    "avg_sentiment": 0.0,
                    "signal": "neutral",
                    "confidence": 0,
                },
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )

        # Score sentiment
        scored = self.scorer.score_dataframe(articles, text_col="title")

        # Aggregate
        scores = scored["sent_score"].values
        avg_sent = float(scores.mean())
        strong_pos = int((scores > 0.7).sum())
        strong_neg = int((scores < -0.3).sum())

        # Determine signal
        if avg_sent > 0.15 and strong_pos > strong_neg:
            signal = "bullish"
            confidence = min(int(abs(avg_sent) * 100), 95)
        elif avg_sent < -0.15 and strong_neg > strong_pos:
            signal = "bearish"
            confidence = min(int(abs(avg_sent) * 100), 95)
        else:
            signal = "neutral"
            confidence = max(0, 50 - int(abs(avg_sent) * 100))

        # Detect event types from titles
        event_types = self._detect_events(scored["title"].tolist())

        # Top articles (most extreme sentiment)
        top_articles = []
        for _, row in scored.nlargest(3, "sent_score").iterrows():
            top_articles.append({
                "title": row.get("title", ""),
                "sentiment": round(row.get("sent_score", 0), 3),
                "label": row.get("sent_label", "neutral"),
                "date": str(row.get("date", "")),
            })

        elapsed = (time.perf_counter() - start) * 1000

        return AgentResult(
            agent=self.name,
            asset=symbol,
            timestamp=date,
            data={
                "article_count": len(articles),
                "avg_sentiment": round(avg_sent, 4),
                "strong_positive": strong_pos,
                "strong_negative": strong_neg,
                "signal": signal,
                "confidence": confidence,
                "event_types": event_types,
                "top_articles": top_articles,
            },
            processing_time_ms=elapsed,
        )

    async def analyze_all(
        self,
        symbols: list[str],
        date: str,
    ) -> dict[str, AgentResult]:
        """Analyze news for all given symbols on a date."""
        results = {}
        for symbol in symbols:
            results[symbol] = await self.analyze_asset(symbol, date)
        return results

    @staticmethod
    def _detect_events(titles: list[str]) -> list[str]:
        """Detect event types from article titles using keywords."""
        events = set()
        event_keywords = {
            "earnings": ["earnings", "revenue", "profit", "EPS", "quarterly", "guidance", "beat", "miss"],
            "merger": ["merger", "acquisition", "acquire", "takeover", "deal", "buyout"],
            "regulation": ["SEC", "regulatory", "regulation", "lawsuit", "antitrust", "fine", "compliance"],
            "product": ["launch", "release", "new product", "iPhone", "GPU", "chip"],
            "executive": ["CEO", "CFO", "resign", "appoint", "hire", "fired"],
            "macro": ["Fed", "interest rate", "inflation", "GDP", "unemployment", "CPI"],
            "crypto": ["Bitcoin", "crypto", "blockchain", "DeFi", "token"],
            "geopolitical": ["tariff", "sanction", "war", "trade war", "China"],
        }

        title_text = " ".join(titles).lower()
        for event_type, keywords in event_keywords.items():
            if any(kw.lower() in title_text for kw in keywords):
                events.add(event_type)

        return list(events)
