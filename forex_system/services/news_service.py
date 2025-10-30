"""
News and sentiment analysis service using Jina AI for web search.

Provides fundamental signal by analyzing:
- Central bank announcements (ECB, Fed)
- Economic data releases (jobs, GDP, inflation)
- Market sentiment from news articles

Inspired by AI-Trader's real-time information integration approach.
"""
import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class NewsArticle:
    """Represents a news article."""
    title: str
    snippet: str
    url: str
    published_date: Optional[datetime]
    source: str


class NewsService:
    """Service for fetching and analyzing forex news."""

    JINA_API_BASE = "https://s.jina.ai"

    # Central bank keywords by currency
    CENTRAL_BANK_KEYWORDS = {
        'EUR': ['ECB', 'European Central Bank', 'Christine Lagarde', 'eurozone'],
        'USD': ['Federal Reserve', 'Fed', 'Jerome Powell', 'FOMC'],
        'GBP': ['Bank of England', 'BoE', 'Andrew Bailey'],
        'JPY': ['Bank of Japan', 'BOJ', 'Kazuo Ueda'],
        'AUD': ['Reserve Bank of Australia', 'RBA'],
        'CAD': ['Bank of Canada', 'BoC'],
    }

    # Economic indicator keywords
    ECONOMIC_KEYWORDS = [
        'GDP', 'inflation', 'CPI', 'employment', 'jobs report',
        'unemployment', 'interest rate', 'rate hike', 'rate cut',
        'quantitative easing', 'QE', 'monetary policy'
    ]

    def __init__(self, jina_api_key: Optional[str] = None):
        """
        Initialize NewsService.

        Args:
            jina_api_key: Jina AI API key (reads from env if None)
        """
        self.jina_api_key = jina_api_key or os.getenv('JINA_API_KEY')
        if not self.jina_api_key:
            print("Warning: JINA_API_KEY not set. News service will be disabled.")

    def search_forex_news(
        self,
        pair: str,
        date: Optional[datetime] = None,
        max_results: int = 10
    ) -> List[NewsArticle]:
        """
        Search for news related to a forex pair.

        Args:
            pair: Currency pair (e.g., 'EURUSD')
            date: Date to search news for (default: today)
            max_results: Maximum number of articles to return

        Returns:
            List of NewsArticle objects
        """
        if not self.jina_api_key:
            return []

        if date is None:
            date = datetime.utcnow()

        # Extract currencies from pair
        base_currency = pair[:3]
        quote_currency = pair[3:6]

        # Build search query
        query = self._build_search_query(base_currency, quote_currency, date)

        # Search using Jina API
        articles = self._jina_search(query, max_results)

        # Filter by date (anti-look-ahead: only news BEFORE date)
        articles = [
            article for article in articles
            if article.published_date is None or article.published_date <= date
        ]

        return articles

    def get_central_bank_sentiment(
        self,
        currency: str,
        date: Optional[datetime] = None,
        lookback_days: int = 7
    ) -> Dict[str, any]:
        """
        Get sentiment from central bank news for a currency.

        Args:
            currency: Currency code (EUR, USD, GBP, etc.)
            date: Date to analyze (default: today)
            lookback_days: Days to look back for news

        Returns:
            Dict with sentiment score, article count, and key topics
        """
        if date is None:
            date = datetime.utcnow()

        start_date = date - timedelta(days=lookback_days)

        # Get central bank keywords
        cb_keywords = self.CENTRAL_BANK_KEYWORDS.get(currency, [])
        if not cb_keywords:
            return {
                'sentiment': 0.0,
                'article_count': 0,
                'topics': [],
                'error': f'No central bank keywords for {currency}'
            }

        # Build query
        query = f"{' OR '.join(cb_keywords)} monetary policy"

        # Search
        articles = self._jina_search(query, max_results=20)

        # Filter by date range
        articles = [
            article for article in articles
            if article.published_date and start_date <= article.published_date <= date
        ]

        # Analyze sentiment from titles and snippets
        sentiment_score = self._analyze_sentiment(articles)

        # Extract key topics
        topics = self._extract_topics(articles)

        return {
            'sentiment': sentiment_score,
            'article_count': len(articles),
            'topics': topics,
            'date_range': f"{start_date.date()} to {date.date()}"
        }

    def get_news_volume(
        self,
        pair: str,
        date: Optional[datetime] = None,
        lookback_hours: int = 24
    ) -> int:
        """
        Count number of news articles about a pair in recent hours.

        High news volume can indicate market-moving events.

        Args:
            pair: Currency pair
            date: Date to analyze
            lookback_hours: Hours to look back

        Returns:
            Number of articles found
        """
        if date is None:
            date = datetime.utcnow()

        start_date = date - timedelta(hours=lookback_hours)

        articles = self.search_forex_news(pair, date, max_results=100)

        # Count articles in time range
        count = sum(
            1 for article in articles
            if article.published_date and article.published_date >= start_date
        )

        return count

    def _build_search_query(
        self,
        base_currency: str,
        quote_currency: str,
        date: datetime
    ) -> str:
        """
        Build search query for forex news.

        Args:
            base_currency: Base currency code
            quote_currency: Quote currency code
            date: Date for news

        Returns:
            Search query string
        """
        # Get central bank keywords
        base_cb = self.CENTRAL_BANK_KEYWORDS.get(base_currency, [base_currency])[0]
        quote_cb = self.CENTRAL_BANK_KEYWORDS.get(quote_currency, [quote_currency])[0]

        # Format date
        date_str = date.strftime('%Y-%m-%d')

        # Build query
        query = (
            f"({base_currency}/{quote_currency} OR {base_cb} OR {quote_cb}) "
            f"AND ({' OR '.join(self.ECONOMIC_KEYWORDS[:5])}) "
            f"after:{date_str}"
        )

        return query

    def _jina_search(self, query: str, max_results: int) -> List[NewsArticle]:
        """
        Search using Jina AI API.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of NewsArticle objects
        """
        if not self.jina_api_key:
            return []

        try:
            # Jina search URL
            url = f"{self.JINA_API_BASE}/{query}"

            headers = {
                'Authorization': f'Bearer {self.jina_api_key}',
                'X-Return-Format': 'json'
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Parse results (Jina returns structured data)
            articles = []
            for item in data.get('data', [])[:max_results]:
                article = NewsArticle(
                    title=item.get('title', ''),
                    snippet=item.get('description', ''),
                    url=item.get('url', ''),
                    published_date=self._parse_date(item.get('published')),
                    source=item.get('source', '')
                )
                articles.append(article)

            return articles

        except Exception as e:
            print(f"Error searching Jina API: {str(e)}")
            return []

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """
        Parse date string to datetime.

        Args:
            date_str: Date string

        Returns:
            datetime object or None
        """
        if not date_str:
            return None

        try:
            # Try ISO format
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            try:
                # Try common formats
                for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y']:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except:
                        continue
            except:
                return None

        return None

    def _analyze_sentiment(self, articles: List[NewsArticle]) -> float:
        """
        Analyze sentiment from article titles and snippets.

        Simple keyword-based sentiment (can be improved with LLM).

        Args:
            articles: List of articles

        Returns:
            Sentiment score from -1.0 (bearish) to 1.0 (bullish)
        """
        if not articles:
            return 0.0

        # Bullish keywords
        bullish_keywords = [
            'rate hike', 'tighten', 'hawkish', 'strong', 'growth',
            'beat expectations', 'surge', 'rally', 'gain', 'rise'
        ]

        # Bearish keywords
        bearish_keywords = [
            'rate cut', 'ease', 'dovish', 'weak', 'recession',
            'miss expectations', 'plunge', 'fall', 'decline', 'drop'
        ]

        total_score = 0
        for article in articles:
            text = f"{article.title} {article.snippet}".lower()

            # Count bullish/bearish keywords
            bullish_count = sum(1 for kw in bullish_keywords if kw in text)
            bearish_count = sum(1 for kw in bearish_keywords if kw in text)

            # Calculate article sentiment
            if bullish_count + bearish_count > 0:
                article_sentiment = (bullish_count - bearish_count) / (bullish_count + bearish_count)
                total_score += article_sentiment

        # Normalize to -1.0 to 1.0
        if len(articles) > 0:
            return max(-1.0, min(1.0, total_score / len(articles)))

        return 0.0

    def _extract_topics(self, articles: List[NewsArticle]) -> List[str]:
        """
        Extract key topics from articles.

        Args:
            articles: List of articles

        Returns:
            List of topic strings
        """
        if not articles:
            return []

        # Count economic keywords
        keyword_counts = {}
        for article in articles:
            text = f"{article.title} {article.snippet}".lower()
            for keyword in self.ECONOMIC_KEYWORDS:
                if keyword.lower() in text:
                    keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1

        # Get top 5 topics
        top_topics = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return [topic[0] for topic in top_topics]

    def get_market_context(
        self,
        pair: str,
        date: Optional[datetime] = None
    ) -> Dict[str, any]:
        """
        Get comprehensive market context for a forex pair.

        Combines news sentiment, volume, and topics for LLM consumption.

        Args:
            pair: Currency pair
            date: Date to analyze

        Returns:
            Dict with market context information
        """
        if date is None:
            date = datetime.utcnow()

        # Extract currencies
        base_currency = pair[:3]
        quote_currency = pair[3:6]

        # Get central bank sentiment for both currencies
        base_sentiment = self.get_central_bank_sentiment(base_currency, date)
        quote_sentiment = self.get_central_bank_sentiment(quote_currency, date)

        # Get news volume
        news_volume = self.get_news_volume(pair, date, lookback_hours=24)

        # Get recent articles
        recent_articles = self.search_forex_news(pair, date, max_results=5)

        return {
            'pair': pair,
            'date': date.isoformat(),
            'base_currency': {
                'code': base_currency,
                'sentiment': base_sentiment['sentiment'],
                'article_count': base_sentiment['article_count'],
                'topics': base_sentiment['topics']
            },
            'quote_currency': {
                'code': quote_currency,
                'sentiment': quote_sentiment['sentiment'],
                'article_count': quote_sentiment['article_count'],
                'topics': quote_sentiment['topics']
            },
            'news_volume_24h': news_volume,
            'recent_headlines': [
                {
                    'title': article.title,
                    'snippet': article.snippet,
                    'source': article.source
                }
                for article in recent_articles
            ]
        }
