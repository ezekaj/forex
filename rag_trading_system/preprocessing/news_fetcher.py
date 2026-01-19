"""
News Fetcher
============
Fetches financial news from RSS feeds for RAG system.
"""

import json
import hashlib
import sqlite3
import feedparser
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, NEWS_LENCI_DIR

logger = logging.getLogger(__name__)


# Priority RSS feeds for forex/financial news
PRIORITY_RSS_FEEDS = [
    # Tier 1 - Breaking News
    {"name": "Reuters Business", "url": "https://feeds.reuters.com/reuters/businessNews", "tier": 1},
    {"name": "CNBC", "url": "https://www.cnbc.com/id/10000664/device/rss/rss.html", "tier": 1},
    {"name": "MarketWatch", "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories", "tier": 1},
    {"name": "Yahoo Finance", "url": "https://finance.yahoo.com/news/rssindex", "tier": 1},

    # Tier 2 - Forex Specific
    {"name": "FXStreet", "url": "https://www.fxstreet.com/rss/news", "tier": 2},
    {"name": "DailyFX", "url": "https://www.dailyfx.com/feeds/all", "tier": 2},
    {"name": "Forex Factory", "url": "https://www.forexfactory.com/rss.php", "tier": 2},
    {"name": "Investing.com", "url": "https://www.investing.com/rss/news.rss", "tier": 2},

    # Tier 3 - Central Banks & Economics
    {"name": "ECB", "url": "https://www.ecb.europa.eu/rss/press.html", "tier": 3},
    {"name": "Federal Reserve", "url": "https://www.federalreserve.gov/feeds/press_all.xml", "tier": 3},
    {"name": "Bank of England", "url": "https://www.bankofengland.co.uk/rss/news", "tier": 3},

    # Tier 4 - Analysis
    {"name": "Seeking Alpha", "url": "https://seekingalpha.com/market_currents.xml", "tier": 4},
    {"name": "Bloomberg (via Google)", "url": "https://news.google.com/rss/search?q=forex+currency&hl=en-US&gl=US&ceid=US:en", "tier": 4},
    {"name": "Forex News (Google)", "url": "https://news.google.com/rss/search?q=forex+trading+EUR+USD&hl=en-US&gl=US&ceid=US:en", "tier": 4},
]


class NewsFetcher:
    """Fetches and stores financial news from RSS feeds."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DATA_DIR / "news.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize news database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id TEXT UNIQUE,
                title TEXT,
                content TEXT,
                url TEXT,
                source TEXT,
                published_date TEXT,
                fetched_date TEXT,
                tier INTEGER,
                is_embedded INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_published
            ON articles(published_date)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_source
            ON articles(source)
        """)

        conn.commit()
        conn.close()

    def _generate_article_id(self, url: str, title: str) -> str:
        """Generate unique article ID from URL and title."""
        content = f"{url}{title}".encode('utf-8')
        return hashlib.md5(content).hexdigest()

    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse date string to ISO format."""
        if not date_str:
            return None

        try:
            # feedparser usually provides struct_time
            if hasattr(date_str, 'tm_year'):
                return datetime(*date_str[:6]).isoformat()

            # Try common formats
            formats = [
                "%a, %d %b %Y %H:%M:%S %z",
                "%a, %d %b %Y %H:%M:%S GMT",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(str(date_str), fmt).isoformat()
                except:
                    continue

            return None
        except:
            return None

    def fetch_feed(self, feed_info: Dict) -> List[Dict]:
        """Fetch articles from a single RSS feed."""
        articles = []

        try:
            feed = feedparser.parse(feed_info["url"])

            if feed.bozo and not feed.entries:
                logger.warning(f"Failed to parse {feed_info['name']}: {feed.bozo_exception}")
                return []

            for entry in feed.entries[:20]:  # Limit per feed
                title = entry.get('title', '')
                url = entry.get('link', '')

                if not title or not url:
                    continue

                # Get content/summary
                content = ''
                if 'content' in entry and entry.content:
                    content = entry.content[0].get('value', '')
                elif 'summary' in entry:
                    content = entry.summary
                elif 'description' in entry:
                    content = entry.description

                # Clean HTML tags (basic)
                import re
                content = re.sub(r'<[^>]+>', '', content)
                content = content[:5000]  # Limit content length

                # Parse date
                pub_date = None
                if 'published_parsed' in entry and entry.published_parsed:
                    pub_date = self._parse_date(entry.published_parsed)
                elif 'published' in entry:
                    pub_date = self._parse_date(entry.published)
                elif 'updated_parsed' in entry and entry.updated_parsed:
                    pub_date = self._parse_date(entry.updated_parsed)

                if not pub_date:
                    pub_date = datetime.utcnow().isoformat()

                articles.append({
                    'article_id': self._generate_article_id(url, title),
                    'title': title,
                    'content': content,
                    'url': url,
                    'source': feed_info['name'],
                    'published_date': pub_date,
                    'tier': feed_info.get('tier', 5)
                })

            logger.info(f"Fetched {len(articles)} articles from {feed_info['name']}")

        except Exception as e:
            logger.error(f"Error fetching {feed_info['name']}: {e}")

        return articles

    def fetch_all_feeds(self, feeds: List[Dict] = None, max_workers: int = 5) -> int:
        """
        Fetch articles from all RSS feeds in parallel.

        Returns:
            Number of new articles added
        """
        feeds = feeds or PRIORITY_RSS_FEEDS

        all_articles = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.fetch_feed, feed): feed for feed in feeds}

            for future in as_completed(futures):
                try:
                    articles = future.result()
                    all_articles.extend(articles)
                except Exception as e:
                    logger.error(f"Feed fetch error: {e}")

        # Store in database
        new_count = self._store_articles(all_articles)
        logger.info(f"Total: {len(all_articles)} articles, {new_count} new")

        return new_count

    def _store_articles(self, articles: List[Dict]) -> int:
        """Store articles in database. Returns count of new articles."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        new_count = 0
        fetched_date = datetime.utcnow().isoformat()

        for article in articles:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO articles
                    (article_id, title, content, url, source, published_date, fetched_date, tier)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    article['article_id'],
                    article['title'],
                    article['content'],
                    article['url'],
                    article['source'],
                    article['published_date'],
                    fetched_date,
                    article['tier']
                ))

                if cursor.rowcount > 0:
                    new_count += 1

            except Exception as e:
                logger.error(f"Failed to store article: {e}")

        conn.commit()
        conn.close()

        return new_count

    def get_articles(
        self,
        start_date: str = None,
        end_date: str = None,
        source: str = None,
        limit: int = 1000,
        not_embedded: bool = False
    ) -> List[Dict]:
        """
        Get articles from database.

        Args:
            start_date: Filter by date >= start_date (ISO format)
            end_date: Filter by date <= end_date (ISO format)
            source: Filter by source name
            limit: Maximum articles to return
            not_embedded: Only return articles not yet embedded

        Returns:
            List of article dicts
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT article_id, title, content, url, source, published_date, tier FROM articles WHERE 1=1"
        params = []

        if start_date:
            query += " AND published_date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND published_date <= ?"
            params.append(end_date)

        if source:
            query += " AND source = ?"
            params.append(source)

        if not_embedded:
            query += " AND is_embedded = 0"

        query += " ORDER BY published_date DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                'article_id': row[0],
                'title': row[1],
                'content': row[2],
                'url': row[3],
                'source': row[4],
                'published_date': row[5],
                'tier': row[6]
            }
            for row in rows
        ]

    def mark_as_embedded(self, article_ids: List[str]):
        """Mark articles as embedded in vector store."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for article_id in article_ids:
            cursor.execute(
                "UPDATE articles SET is_embedded = 1 WHERE article_id = ?",
                (article_id,)
            )

        conn.commit()
        conn.close()

    def get_stats(self) -> Dict:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(*) FROM articles")
        stats['total_articles'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM articles WHERE is_embedded = 1")
        stats['embedded_articles'] = cursor.fetchone()[0]

        cursor.execute("SELECT source, COUNT(*) FROM articles GROUP BY source ORDER BY COUNT(*) DESC")
        stats['by_source'] = dict(cursor.fetchall())

        cursor.execute("SELECT MIN(published_date), MAX(published_date) FROM articles")
        row = cursor.fetchone()
        stats['date_range'] = {'min': row[0], 'max': row[1]}

        conn.close()
        return stats


def load_feeds_from_json(json_path: str) -> List[Dict]:
    """Load RSS feed configurations from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        feeds = []
        for item in data:
            if item.get('rss_url'):
                feeds.append({
                    'name': item.get('name', 'Unknown'),
                    'url': item['rss_url'],
                    'tier': item.get('tier', 3)
                })

        return feeds
    except Exception as e:
        logger.error(f"Failed to load feeds from {json_path}: {e}")
        return []


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    fetcher = NewsFetcher()

    print("Fetching news from RSS feeds...")
    new_articles = fetcher.fetch_all_feeds()

    print(f"\nNew articles added: {new_articles}")
    print(f"\nDatabase stats:")
    stats = fetcher.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nLatest articles:")
    for article in fetcher.get_articles(limit=5):
        print(f"  [{article['source']}] {article['title'][:60]}...")
