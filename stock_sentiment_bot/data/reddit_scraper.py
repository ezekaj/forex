"""
Reddit Scraper - Fetch posts from r/wallstreetbets, r/stocks, r/options.
Extracts ticker mentions and engagement metrics.
"""
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from collections import defaultdict

import praw
from praw.models import Submission, Comment

from config.settings import RedditConfig
from config.constants import (
    REDDIT_RATE_LIMIT_SECONDS,
    REDDIT_MAX_POSTS_PER_SUB,
    MIN_UNIQUE_AUTHORS,
    MIN_ENGAGEMENT_SCORE,
)


logger = logging.getLogger("sentiment_bot.reddit")


# Common words that look like tickers but aren't
TICKER_BLACKLIST = {
    "A", "I", "AM", "PM", "CEO", "CFO", "IPO", "ETF", "SEC", "FBI", "CIA",
    "DD", "FD", "TA", "EPS", "PE", "ATH", "ATL", "YOLO", "FOMO", "FUD",
    "ITM", "OTM", "ATM", "IV", "DTE", "EOD", "AH", "PM", "ER", "PT",
    "TLDR", "IMO", "IMHO", "WSB", "DD", "USA", "USD", "EUR", "GBP",
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN",
    "HAD", "HER", "WAS", "ONE", "OUR", "OUT", "HAS", "HIS", "HOW",
    "ITS", "MAY", "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "BOY",
    "DID", "GET", "HIM", "LET", "PUT", "SAY", "SHE", "TOO", "USE",
    "EDIT", "THIS", "THAT", "WITH", "HAVE", "FROM", "WILL", "JUST",
    "BEEN", "SOME", "THEM", "THAN", "MADE", "MOST", "LONG", "VERY",
    "MUCH", "ONLY", "OVER", "SUCH", "TAKE", "COME", "YOUR", "MAKE",
}

# Known valid tickers we want to track
VALID_TICKERS = {
    "GME", "AMC", "BBBY", "PLTR", "NVDA", "TSLA", "AAPL", "AMD",
    "SOFI", "SPY", "QQQ", "MSFT", "GOOGL", "GOOG", "META", "AMZN",
    "NFLX", "DIS", "BA", "F", "GM", "NIO", "RIVN", "LCID",
    "COIN", "HOOD", "RBLX", "SNAP", "PINS", "TWTR", "UBER", "LYFT",
    "ABNB", "SQ", "PYPL", "V", "MA", "JPM", "GS", "BAC", "WFC",
    "C", "MS", "SCHW", "BRK.B", "BRK.A", "JNJ", "PFE", "MRNA",
    "BNTX", "CVS", "UNH", "WMT", "COST", "TGT", "HD", "LOW",
    "MCD", "SBUX", "NKE", "LULU", "PEP", "KO", "PM", "MO",
    "XOM", "CVX", "COP", "OXY", "SLB", "HAL", "BP", "SHEL",
}


@dataclass
class RedditPost:
    """Parsed Reddit post with extracted data."""
    post_id: str
    subreddit: str
    title: str
    selftext: str
    author: str
    created_utc: datetime
    score: int
    upvote_ratio: float
    num_comments: int
    url: str
    is_dd: bool  # Due Diligence post
    tickers_mentioned: Set[str] = field(default_factory=set)

    @property
    def engagement_score(self) -> int:
        """Combined engagement metric."""
        return self.score + (self.num_comments * 2)

    @property
    def age_hours(self) -> float:
        """Post age in hours."""
        return (datetime.utcnow() - self.created_utc).total_seconds() / 3600


@dataclass
class TickerMentions:
    """Aggregated mentions for a ticker."""
    ticker: str
    total_mentions: int = 0
    unique_authors: Set[str] = field(default_factory=set)
    total_engagement: int = 0
    posts: List[RedditPost] = field(default_factory=list)
    mention_velocity: float = 0.0  # mentions per hour
    avg_sentiment_words: float = 0.0

    @property
    def author_count(self) -> int:
        return len(self.unique_authors)


class RedditScraper:
    """
    Reddit scraper for sentiment data.

    Monitors:
    - r/wallstreetbets (meme stocks, YOLO plays)
    - r/stocks (more balanced discussion)
    - r/options (derivatives sentiment)
    """

    def __init__(self, config: RedditConfig):
        self.config = config

        # Initialize PRAW
        self.reddit = praw.Reddit(
            client_id=config.client_id,
            client_secret=config.client_secret,
            user_agent=config.user_agent
        )

        # Rate limiting
        self.last_request_time = 0

        # Ticker regex pattern: $AAPL or standalone AAPL (2-5 uppercase letters)
        self.ticker_pattern = re.compile(r'\$?([A-Z]{2,5})\b')

        logger.info(f"Reddit scraper initialized for: {config.subreddits}")

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < REDDIT_RATE_LIMIT_SECONDS:
            time.sleep(REDDIT_RATE_LIMIT_SECONDS - elapsed)
        self.last_request_time = time.time()

    def _extract_tickers(self, text: str) -> Set[str]:
        """Extract ticker symbols from text."""
        matches = self.ticker_pattern.findall(text.upper())

        # Filter out blacklisted words and validate
        tickers = set()
        for match in matches:
            if match in TICKER_BLACKLIST:
                continue
            if match in VALID_TICKERS:
                tickers.add(match)
            elif len(match) >= 3:  # Allow unknown tickers if 3+ chars
                tickers.add(match)

        return tickers

    def _is_dd_post(self, post: Submission) -> bool:
        """Check if post is Due Diligence."""
        title_lower = post.title.lower()
        flair = (post.link_flair_text or "").lower()

        dd_indicators = ["dd", "due diligence", "analysis", "deep dive", "research"]
        return any(ind in title_lower or ind in flair for ind in dd_indicators)

    def _parse_post(self, post: Submission) -> RedditPost:
        """Parse a Reddit submission into our data structure."""
        text_content = f"{post.title} {post.selftext}"
        tickers = self._extract_tickers(text_content)

        return RedditPost(
            post_id=post.id,
            subreddit=str(post.subreddit),
            title=post.title,
            selftext=post.selftext[:1000] if post.selftext else "",  # Truncate
            author=str(post.author) if post.author else "[deleted]",
            created_utc=datetime.utcfromtimestamp(post.created_utc),
            score=post.score,
            upvote_ratio=post.upvote_ratio,
            num_comments=post.num_comments,
            url=post.url,
            is_dd=self._is_dd_post(post),
            tickers_mentioned=tickers
        )

    def fetch_hot_posts(
        self,
        subreddit: str,
        limit: int = REDDIT_MAX_POSTS_PER_SUB
    ) -> List[RedditPost]:
        """Fetch hot posts from a subreddit."""
        self._rate_limit()

        posts = []
        try:
            sub = self.reddit.subreddit(subreddit)
            for submission in sub.hot(limit=limit):
                posts.append(self._parse_post(submission))
        except Exception as e:
            logger.error(f"Error fetching from r/{subreddit}: {e}")

        logger.info(f"Fetched {len(posts)} posts from r/{subreddit}")
        return posts

    def fetch_new_posts(
        self,
        subreddit: str,
        limit: int = REDDIT_MAX_POSTS_PER_SUB
    ) -> List[RedditPost]:
        """Fetch newest posts from a subreddit."""
        self._rate_limit()

        posts = []
        try:
            sub = self.reddit.subreddit(subreddit)
            for submission in sub.new(limit=limit):
                posts.append(self._parse_post(submission))
        except Exception as e:
            logger.error(f"Error fetching new from r/{subreddit}: {e}")

        return posts

    def search_ticker(
        self,
        ticker: str,
        subreddit: str = "wallstreetbets",
        time_filter: str = "day",
        limit: int = 50
    ) -> List[RedditPost]:
        """Search for posts mentioning a specific ticker."""
        self._rate_limit()

        posts = []
        try:
            sub = self.reddit.subreddit(subreddit)
            # Search with both $TICKER and plain TICKER
            for query in [f"${ticker}", ticker]:
                for submission in sub.search(
                    query,
                    time_filter=time_filter,
                    sort="hot",
                    limit=limit
                ):
                    post = self._parse_post(submission)
                    if ticker in post.tickers_mentioned:
                        posts.append(post)
        except Exception as e:
            logger.error(f"Error searching {ticker} in r/{subreddit}: {e}")

        # Deduplicate
        seen = set()
        unique_posts = []
        for p in posts:
            if p.post_id not in seen:
                seen.add(p.post_id)
                unique_posts.append(p)

        return unique_posts

    def get_all_mentions(
        self,
        hours_back: int = 24
    ) -> Dict[str, TickerMentions]:
        """
        Get all ticker mentions across configured subreddits.
        Returns aggregated mentions per ticker.
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        ticker_data: Dict[str, TickerMentions] = defaultdict(
            lambda: TickerMentions(ticker="")
        )

        # Fetch from all subreddits
        for subreddit in self.config.subreddits:
            posts = self.fetch_hot_posts(subreddit)
            posts.extend(self.fetch_new_posts(subreddit, limit=50))

            for post in posts:
                # Skip old posts
                if post.created_utc < cutoff_time:
                    continue

                # Aggregate by ticker
                for ticker in post.tickers_mentioned:
                    if ticker_data[ticker].ticker == "":
                        ticker_data[ticker].ticker = ticker

                    ticker_data[ticker].total_mentions += 1
                    ticker_data[ticker].unique_authors.add(post.author)
                    ticker_data[ticker].total_engagement += post.engagement_score
                    ticker_data[ticker].posts.append(post)

        # Calculate mention velocity
        for ticker, data in ticker_data.items():
            if data.posts:
                oldest = min(p.created_utc for p in data.posts)
                hours = max(1, (datetime.utcnow() - oldest).total_seconds() / 3600)
                data.mention_velocity = data.total_mentions / hours

        return dict(ticker_data)

    def get_trending_tickers(
        self,
        hours_back: int = 24,
        min_mentions: int = 5,
        min_authors: int = MIN_UNIQUE_AUTHORS,
        min_engagement: int = MIN_ENGAGEMENT_SCORE
    ) -> List[TickerMentions]:
        """
        Get tickers trending on Reddit.
        Sorted by mention velocity (mentions per hour).
        """
        all_mentions = self.get_all_mentions(hours_back)

        # Filter by quality thresholds
        trending = []
        for ticker, data in all_mentions.items():
            if (data.total_mentions >= min_mentions and
                data.author_count >= min_authors and
                data.total_engagement >= min_engagement):
                trending.append(data)

        # Sort by mention velocity
        trending.sort(key=lambda x: x.mention_velocity, reverse=True)

        logger.info(f"Found {len(trending)} trending tickers")
        return trending

    def detect_mention_spike(
        self,
        ticker: str,
        spike_threshold: float = 3.0,
        comparison_hours: int = 24
    ) -> Optional[float]:
        """
        Detect if ticker has unusual mention activity.
        Returns spike multiplier if detected, None otherwise.

        spike_threshold: How many times above average to consider a spike
        """
        # Get recent mentions
        recent = self.search_ticker(ticker, time_filter="hour")
        recent_count = len(recent)

        # Get baseline
        baseline = self.search_ticker(ticker, time_filter="day")
        if len(baseline) < 5:  # Not enough data
            return None

        # Calculate average hourly rate over past day
        avg_hourly = len(baseline) / comparison_hours

        if avg_hourly == 0:
            return None

        spike_ratio = recent_count / avg_hourly

        if spike_ratio >= spike_threshold:
            logger.info(f"SPIKE detected for {ticker}: {spike_ratio:.1f}x normal")
            return spike_ratio

        return None
