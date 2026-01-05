"""
Sentiment Scorer - Analyze text sentiment for trading signals.
Uses VADER for financial text and keyword-based scoring.
"""
import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config.constants import (
    BULLISH_SENTIMENT_THRESHOLD,
    BEARISH_SENTIMENT_THRESHOLD,
)


logger = logging.getLogger("sentiment_bot.sentiment")


# WSB/Reddit specific sentiment words not in standard VADER
BULLISH_WORDS = {
    # Strong bullish
    "moon": 2.0, "mooning": 2.5, "rocket": 2.0, "rockets": 2.0,
    "tendies": 1.5, "lambo": 1.5, "yolo": 1.0, "diamond": 1.5,
    "hands": 0.5, "hold": 0.5, "hodl": 1.0, "holding": 0.5,
    "bullish": 2.0, "calls": 1.0, "long": 0.8, "buy": 1.0,
    "buying": 1.0, "bought": 0.8, "squeeze": 1.5, "squeezing": 1.5,
    "gamma": 1.0, "short squeeze": 2.0, "undervalued": 1.5,
    "breakout": 1.5, "breaking out": 1.5, "ripping": 1.5, "rip": 1.0,
    "pump": 1.0, "pumping": 1.0, "green": 1.0, "gains": 1.5,
    "gain": 1.0, "profit": 1.0, "profits": 1.0, "winner": 1.0,
    "winning": 1.0, "win": 0.8, "rich": 1.0, "free money": 2.0,
    "easy money": 1.5, "print": 1.0, "printing": 1.5, "brrrr": 1.5,
    "ath": 1.0, "all time high": 1.5, "new high": 1.0,

    # Moderate bullish
    "support": 0.5, "bounce": 0.8, "bouncing": 0.8, "recovery": 0.8,
    "recovering": 0.8, "oversold": 1.0, "dip": 0.5, "buying the dip": 1.5,
    "btfd": 1.5, "accumulating": 1.0, "accumulate": 1.0,
}

BEARISH_WORDS = {
    # Strong bearish
    "crash": -2.0, "crashing": -2.5, "dump": -1.5, "dumping": -2.0,
    "tank": -1.5, "tanking": -2.0, "plunge": -2.0, "plunging": -2.0,
    "collapse": -2.5, "collapsing": -2.5, "bankrupt": -2.5,
    "bankruptcy": -2.5, "fraud": -2.0, "scam": -2.0, "ponzi": -2.5,
    "puts": -1.0, "short": -0.8, "shorting": -1.0, "shorted": -0.8,
    "bearish": -2.0, "sell": -1.0, "selling": -1.0, "sold": -0.8,
    "bag": -1.0, "bagholder": -1.5, "bagholding": -1.5, "bags": -1.0,
    "loss": -1.0, "losses": -1.0, "losing": -1.0, "lost": -1.0,
    "red": -1.0, "bleeding": -1.5, "bleed": -1.0, "dead": -1.5,
    "rip": -0.5, "rekt": -1.5, "wrecked": -1.5, "destroyed": -2.0,
    "overvalued": -1.5, "bubble": -1.5, "rug pull": -2.5, "rugpull": -2.5,

    # Moderate bearish
    "resistance": -0.5, "rejection": -1.0, "rejected": -1.0,
    "overbought": -1.0, "top": -0.5, "topping": -0.8, "peaked": -1.0,
    "fud": -0.5, "fear": -1.0, "worried": -0.8, "concern": -0.8,
}

# Emoji sentiment (common in WSB)
EMOJI_SENTIMENT = {
    "🚀": 2.0,   # Rocket - very bullish
    "🌙": 1.5,   # Moon - bullish
    "💎": 1.5,   # Diamond - holding strong
    "🙌": 1.0,   # Hands up - positive
    "🦍": 1.0,   # Ape - WSB solidarity
    "🐂": 1.5,   # Bull
    "📈": 1.5,   # Chart up
    "💰": 1.0,   # Money bag
    "🤑": 1.0,   # Money face

    "🐻": -1.5,  # Bear
    "📉": -1.5,  # Chart down
    "💩": -1.0,  # Poop
    "🤡": -1.0,  # Clown
    "☠️": -1.5,  # Skull
    "🔻": -1.0,  # Red triangle
    "😭": -1.0,  # Crying
}


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    text_preview: str
    compound_score: float  # -1 to 1
    positive: float
    negative: float
    neutral: float
    wsb_adjustment: float  # Additional WSB-specific sentiment
    final_score: float  # Combined score

    @property
    def sentiment_label(self) -> str:
        if self.final_score >= BULLISH_SENTIMENT_THRESHOLD:
            return "BULLISH"
        elif self.final_score <= BEARISH_SENTIMENT_THRESHOLD:
            return "BEARISH"
        else:
            return "NEUTRAL"


class SentimentScorer:
    """
    Financial sentiment analyzer with WSB-specific adjustments.

    Combines:
    1. VADER sentiment (general purpose)
    2. WSB/Reddit specific vocabulary
    3. Emoji sentiment
    """

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

        # Update VADER lexicon with financial terms
        for word, score in {**BULLISH_WORDS, **BEARISH_WORDS}.items():
            self.vader.lexicon[word] = score

        logger.info("SentimentScorer initialized with WSB vocabulary")

    def analyze_text(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.
        Returns score from -1 (bearish) to +1 (bullish).
        """
        if not text or not text.strip():
            return SentimentResult(
                text_preview="",
                compound_score=0,
                positive=0,
                negative=0,
                neutral=1,
                wsb_adjustment=0,
                final_score=0
            )

        # Clean text
        text_clean = text.lower()
        preview = text[:100] + "..." if len(text) > 100 else text

        # VADER analysis
        vader_scores = self.vader.polarity_scores(text_clean)

        # Calculate WSB-specific adjustment
        wsb_adj = self._calculate_wsb_adjustment(text)

        # Emoji adjustment
        emoji_adj = self._calculate_emoji_sentiment(text)

        # Combine scores (weighted average)
        final_score = (
            vader_scores['compound'] * 0.6 +
            wsb_adj * 0.25 +
            emoji_adj * 0.15
        )

        # Clamp to [-1, 1]
        final_score = max(-1, min(1, final_score))

        return SentimentResult(
            text_preview=preview,
            compound_score=vader_scores['compound'],
            positive=vader_scores['pos'],
            negative=vader_scores['neg'],
            neutral=vader_scores['neu'],
            wsb_adjustment=wsb_adj + emoji_adj,
            final_score=final_score
        )

    def analyze_posts(
        self,
        posts: List[Dict],
        weight_by_engagement: bool = True
    ) -> Tuple[float, List[SentimentResult]]:
        """
        Analyze sentiment across multiple posts.

        Args:
            posts: List of dicts with 'text' and optionally 'engagement' keys
            weight_by_engagement: Weight sentiment by post engagement

        Returns:
            (aggregate_score, individual_results)
        """
        if not posts:
            return 0.0, []

        results = []
        weighted_sum = 0
        total_weight = 0

        for post in posts:
            text = post.get('text', '') or post.get('title', '')
            engagement = post.get('engagement', 1) if weight_by_engagement else 1

            result = self.analyze_text(text)
            results.append(result)

            # Weight by engagement (log scale to reduce outlier impact)
            weight = 1 + (engagement ** 0.5) if engagement > 0 else 1
            weighted_sum += result.final_score * weight
            total_weight += weight

        aggregate = weighted_sum / total_weight if total_weight > 0 else 0

        return aggregate, results

    def get_sentiment_summary(
        self,
        posts: List[Dict]
    ) -> Dict:
        """
        Get summary statistics for sentiment across posts.
        """
        aggregate, results = self.analyze_posts(posts)

        if not results:
            return {
                "aggregate_score": 0,
                "sentiment_label": "NEUTRAL",
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "bullish_pct": 0,
                "bearish_pct": 0,
            }

        bullish = sum(1 for r in results if r.sentiment_label == "BULLISH")
        bearish = sum(1 for r in results if r.sentiment_label == "BEARISH")
        neutral = sum(1 for r in results if r.sentiment_label == "NEUTRAL")
        total = len(results)

        return {
            "aggregate_score": round(aggregate, 3),
            "sentiment_label": (
                "BULLISH" if aggregate >= BULLISH_SENTIMENT_THRESHOLD
                else "BEARISH" if aggregate <= BEARISH_SENTIMENT_THRESHOLD
                else "NEUTRAL"
            ),
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": neutral,
            "bullish_pct": round(bullish / total * 100, 1),
            "bearish_pct": round(bearish / total * 100, 1),
            "total_posts": total,
        }

    def _calculate_wsb_adjustment(self, text: str) -> float:
        """Calculate adjustment based on WSB-specific vocabulary."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        adjustment = 0
        word_count = 0

        for word in words:
            if word in BULLISH_WORDS:
                adjustment += BULLISH_WORDS[word]
                word_count += 1
            elif word in BEARISH_WORDS:
                adjustment += BEARISH_WORDS[word]
                word_count += 1

        # Normalize by word count (cap impact)
        if word_count > 0:
            adjustment = adjustment / (word_count ** 0.5)

        # Clamp
        return max(-1, min(1, adjustment))

    def _calculate_emoji_sentiment(self, text: str) -> float:
        """Calculate sentiment from emojis."""
        adjustment = 0
        emoji_count = 0

        for emoji, score in EMOJI_SENTIMENT.items():
            count = text.count(emoji)
            if count > 0:
                adjustment += score * count
                emoji_count += count

        # Normalize
        if emoji_count > 0:
            adjustment = adjustment / (emoji_count ** 0.5)

        return max(-1, min(1, adjustment))

    def score_ticker_mentions(
        self,
        ticker: str,
        posts: List
    ) -> Dict:
        """
        Score sentiment for posts mentioning a specific ticker.
        Filters and weights by ticker relevance.
        """
        # Filter to posts mentioning this ticker
        relevant_posts = []
        for post in posts:
            text = f"{post.title} {post.selftext}" if hasattr(post, 'title') else str(post)
            if ticker.upper() in text.upper():
                # Higher weight if ticker is prominent (in title)
                engagement = getattr(post, 'engagement_score', 1)
                in_title = ticker.upper() in getattr(post, 'title', '').upper()

                relevant_posts.append({
                    'text': text,
                    'engagement': engagement * (2 if in_title else 1)
                })

        if not relevant_posts:
            return {
                "ticker": ticker,
                "sentiment_score": 0,
                "post_count": 0,
                "confidence": 0
            }

        aggregate, _ = self.analyze_posts(relevant_posts)

        # Confidence based on number of posts
        confidence = min(1.0, len(relevant_posts) / 20)

        return {
            "ticker": ticker,
            "sentiment_score": round(aggregate, 3),
            "post_count": len(relevant_posts),
            "confidence": round(confidence, 2)
        }
