"""
Signal Generator - Combine sentiment, technical, and quality signals.
Produces actionable trading signals with confidence scores.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from config.constants import (
    MIN_SIGNAL_SCORE,
    MENTION_SPIKE_MULTIPLIER,
    BULLISH_SENTIMENT_THRESHOLD,
    RSI_OVERSOLD,
    RSI_OVERBOUGHT,
    VOLUME_SPIKE_MULTIPLIER,
    MIN_UNIQUE_AUTHORS,
    MIN_ENGAGEMENT_SCORE,
    MIN_REWARD_RISK_RATIO,
)


logger = logging.getLogger("sentiment_bot.signals")


@dataclass
class TradingSignal:
    """Complete trading signal with all scoring components."""
    ticker: str
    timestamp: datetime

    # Overall
    total_score: int  # 0-100
    action: str  # 'BUY', 'SELL', 'HOLD', 'SKIP'
    confidence: float  # 0-1

    # Sentiment components (0-40 points)
    sentiment_score: float  # -1 to 1
    sentiment_points: int
    mention_velocity: float
    mention_spike_points: int

    # Technical components (0-30 points)
    rsi: Optional[float]
    rsi_points: int
    volume_ratio: Optional[float]
    volume_points: int

    # Quality components (0-30 points)
    unique_authors: int
    author_points: int
    has_dd_post: bool
    dd_points: int
    total_engagement: int
    engagement_points: int

    # Reasoning
    reasons: List[str]

    def __str__(self) -> str:
        return (
            f"Signal: {self.ticker} | Score: {self.total_score}/100 | "
            f"Action: {self.action} | Confidence: {self.confidence:.0%}\n"
            f"  Sentiment: {self.sentiment_points}/40 (score={self.sentiment_score:.2f}, "
            f"velocity={self.mention_velocity:.1f}/hr)\n"
            f"  Technical: {self.rsi_points + self.volume_points}/30 "
            f"(RSI={self.rsi or 'N/A'}, Vol={self.volume_ratio or 'N/A'}x)\n"
            f"  Quality: {self.author_points + self.dd_points + self.engagement_points}/30 "
            f"(authors={self.unique_authors}, DD={self.has_dd_post})\n"
            f"  Reasons: {', '.join(self.reasons)}"
        )


class SignalGenerator:
    """
    Generate trading signals from multiple data sources.

    Scoring System (0-100):
    - Sentiment momentum: 0-40 points
      - Mention spike (>3x avg): 20 points
      - Bullish sentiment (>0.6): 20 points

    - Technical confirmation: 0-30 points
      - RSI oversold (<40): 15 points
      - Volume spike (>2x avg): 15 points

    - Quality filters: 0-30 points
      - Unique authors (>20): 10 points
      - Has DD post: 10 points
      - High engagement: 10 points

    Only trade if score >= 60
    """

    def __init__(self):
        self.min_score = MIN_SIGNAL_SCORE
        logger.info(f"SignalGenerator initialized (min_score={self.min_score})")

    def generate_signal(
        self,
        ticker: str,
        # Sentiment data
        sentiment_score: float,
        mention_velocity: float,
        avg_mention_velocity: float,
        unique_authors: int,
        has_dd_post: bool,
        total_engagement: int,
        # Technical data
        rsi: Optional[float] = None,
        volume_ratio: Optional[float] = None,
        # Context
        insider_buying: bool = False,
    ) -> TradingSignal:
        """
        Generate a trading signal for a ticker.

        Args:
            ticker: Stock symbol
            sentiment_score: -1 to 1 sentiment
            mention_velocity: Mentions per hour (recent)
            avg_mention_velocity: Average mentions per hour (baseline)
            unique_authors: Number of unique posters
            has_dd_post: Whether there's a Due Diligence post
            total_engagement: Total upvotes + comments
            rsi: Relative Strength Index (0-100)
            volume_ratio: Current volume / average volume
            insider_buying: Whether insiders are buying

        Returns:
            TradingSignal with complete scoring breakdown
        """
        reasons = []
        total_score = 0

        # =====================================================================
        # SENTIMENT MOMENTUM (0-40 points)
        # =====================================================================

        # Mention spike detection (0-20 points)
        mention_spike_points = 0
        if avg_mention_velocity > 0:
            spike_ratio = mention_velocity / avg_mention_velocity
            if spike_ratio >= MENTION_SPIKE_MULTIPLIER:
                mention_spike_points = 20
                reasons.append(f"Mention spike {spike_ratio:.1f}x")
            elif spike_ratio >= 2.0:
                mention_spike_points = 10
                reasons.append(f"Elevated mentions {spike_ratio:.1f}x")

        # Sentiment strength (0-20 points)
        sentiment_points = 0
        if sentiment_score >= BULLISH_SENTIMENT_THRESHOLD:
            sentiment_points = 20
            reasons.append(f"Strong bullish sentiment ({sentiment_score:.2f})")
        elif sentiment_score >= 0.4:
            sentiment_points = 10
            reasons.append(f"Moderate bullish ({sentiment_score:.2f})")
        elif sentiment_score <= -BULLISH_SENTIMENT_THRESHOLD:
            # Bearish - could be short signal but we skip for now
            reasons.append(f"Bearish sentiment ({sentiment_score:.2f})")

        total_score += mention_spike_points + sentiment_points

        # =====================================================================
        # TECHNICAL CONFIRMATION (0-30 points)
        # =====================================================================

        # RSI (0-15 points)
        rsi_points = 0
        if rsi is not None:
            if rsi < RSI_OVERSOLD:
                rsi_points = 15
                reasons.append(f"RSI oversold ({rsi:.0f})")
            elif rsi < 50:
                rsi_points = 7
                reasons.append(f"RSI neutral-low ({rsi:.0f})")
            elif rsi > RSI_OVERBOUGHT:
                reasons.append(f"RSI overbought ({rsi:.0f}) - caution")

        # Volume (0-15 points)
        volume_points = 0
        if volume_ratio is not None:
            if volume_ratio >= VOLUME_SPIKE_MULTIPLIER:
                volume_points = 15
                reasons.append(f"Volume spike {volume_ratio:.1f}x")
            elif volume_ratio >= 1.5:
                volume_points = 7
                reasons.append(f"Elevated volume {volume_ratio:.1f}x")

        total_score += rsi_points + volume_points

        # =====================================================================
        # QUALITY FILTERS (0-30 points)
        # =====================================================================

        # Unique authors (0-10 points) - filters out bot manipulation
        author_points = 0
        if unique_authors >= MIN_UNIQUE_AUTHORS:
            author_points = 10
            reasons.append(f"{unique_authors} unique authors (organic)")
        elif unique_authors >= 10:
            author_points = 5
            reasons.append(f"{unique_authors} authors (moderate)")
        else:
            reasons.append(f"Low author count ({unique_authors}) - possible manipulation")

        # DD post (0-10 points)
        dd_points = 10 if has_dd_post else 0
        if has_dd_post:
            reasons.append("Has DD post (quality analysis)")

        # Engagement (0-10 points)
        engagement_points = 0
        if total_engagement >= MIN_ENGAGEMENT_SCORE * 5:
            engagement_points = 10
            reasons.append(f"High engagement ({total_engagement})")
        elif total_engagement >= MIN_ENGAGEMENT_SCORE:
            engagement_points = 5

        total_score += author_points + dd_points + engagement_points

        # =====================================================================
        # BONUS POINTS
        # =====================================================================

        if insider_buying:
            total_score += 10
            reasons.append("Insider buying detected")

        # Cap at 100
        total_score = min(100, total_score)

        # =====================================================================
        # DETERMINE ACTION
        # =====================================================================

        if total_score >= 80:
            action = "BUY"
            confidence = 0.8 + (total_score - 80) * 0.01
        elif total_score >= 60:
            action = "BUY"
            confidence = 0.5 + (total_score - 60) * 0.015
        else:
            action = "SKIP"
            confidence = total_score / 100
            reasons.append(f"Score {total_score} below threshold {self.min_score}")

        # Override if bearish
        if sentiment_score <= -BULLISH_SENTIMENT_THRESHOLD:
            action = "SKIP"
            reasons.append("Skipping due to bearish sentiment")

        return TradingSignal(
            ticker=ticker,
            timestamp=datetime.utcnow(),
            total_score=total_score,
            action=action,
            confidence=min(1.0, confidence),
            sentiment_score=sentiment_score,
            sentiment_points=sentiment_points,
            mention_velocity=mention_velocity,
            mention_spike_points=mention_spike_points,
            rsi=rsi,
            rsi_points=rsi_points,
            volume_ratio=volume_ratio,
            volume_points=volume_points,
            unique_authors=unique_authors,
            author_points=author_points,
            has_dd_post=has_dd_post,
            dd_points=dd_points,
            total_engagement=total_engagement,
            engagement_points=engagement_points,
            reasons=reasons
        )

    def filter_signals(
        self,
        signals: List[TradingSignal],
        min_score: Optional[int] = None,
        action_filter: Optional[str] = None
    ) -> List[TradingSignal]:
        """Filter signals by criteria."""
        min_score = min_score or self.min_score

        filtered = [
            s for s in signals
            if s.total_score >= min_score
            and (action_filter is None or s.action == action_filter)
        ]

        # Sort by score descending
        filtered.sort(key=lambda x: x.total_score, reverse=True)

        return filtered

    def rank_signals(
        self,
        signals: List[TradingSignal]
    ) -> List[Tuple[int, TradingSignal]]:
        """Rank signals by total score with position numbers."""
        sorted_signals = sorted(signals, key=lambda x: x.total_score, reverse=True)
        return [(i + 1, s) for i, s in enumerate(sorted_signals)]
