"""
Historical Forex News Fetcher
=============================
Fetches historical forex news from GDELT for backtesting.

GDELT provides free access to news articles going back years.
This fetcher focuses on forex-relevant keywords and stores
data in the RAG system's expected format.
"""

import sqlite3
import requests
import hashlib
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

logger = logging.getLogger(__name__)

# Forex-specific keywords for GDELT queries
FOREX_KEYWORDS = [
    # Major pairs
    "EUR USD forex",
    "GBP USD forex",
    "USD JPY forex",
    "EURUSD",
    "GBPUSD",
    "USDJPY",

    # Central banks
    "federal reserve rate",
    "federal reserve policy",
    "ECB interest rate",
    "ECB monetary policy",
    "Bank of England rate",
    "Bank of Japan policy",

    # Economic indicators
    "US inflation data",
    "eurozone inflation",
    "US jobs report",
    "nonfarm payrolls",
    "GDP growth rate",
    "unemployment rate",

    # Currency moves
    "dollar strength",
    "dollar weakness",
    "euro rally",
    "yen depreciation",
    "currency exchange rate",

    # Policy
    "interest rate decision",
    "rate hike",
    "rate cut",
    "quantitative easing",
    "monetary policy meeting",

    # Market sentiment
    "forex market",
    "currency trading",
    "forex outlook",
    "currency forecast",
]


class HistoricalNewsFetcher:
    """Fetches historical forex news from GDELT."""

    GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DATA_DIR / "news.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database with RAG system schema."""
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
                tier INTEGER DEFAULT 2,
                is_embedded INTEGER DEFAULT 0
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_published ON articles(published_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON articles(source)")

        conn.commit()
        conn.close()

    def _generate_article_id(self, url: str, title: str) -> str:
        """Generate unique article ID."""
        content = f"{url}{title}".encode('utf-8')
        return hashlib.md5(content).hexdigest()

    def _fetch_gdelt_day(self, date_str: str, keywords: List[str] = None) -> List[Dict]:
        """Fetch news from GDELT for a specific date."""
        keywords = keywords or FOREX_KEYWORDS
        articles = []
        seen_titles = set()

        for keyword in keywords:
            try:
                params = {
                    'query': f'{keyword} sourcelang:eng',
                    'mode': 'artlist',
                    'maxrecords': 50,
                    'format': 'json',
                    'startdatetime': date_str.replace('-', '') + '000000',
                    'enddatetime': date_str.replace('-', '') + '235959'
                }

                resp = requests.get(self.GDELT_URL, params=params, timeout=20)

                if resp.status_code == 200:
                    data = resp.json()
                    for article in data.get('articles', []):
                        title = article.get('title', '').strip()

                        # Skip duplicates
                        if not title or title in seen_titles:
                            continue
                        seen_titles.add(title)

                        # Parse datetime
                        seendate = article.get('seendate', '')
                        if seendate:
                            try:
                                dt = datetime.strptime(seendate, '%Y%m%dT%H%M%SZ')
                                pub_date = dt.isoformat()
                            except:
                                pub_date = f"{date_str}T12:00:00"
                        else:
                            pub_date = f"{date_str}T12:00:00"

                        articles.append({
                            'article_id': self._generate_article_id(
                                article.get('url', ''), title
                            ),
                            'title': title,
                            'content': article.get('title', ''),  # GDELT only gives title
                            'url': article.get('url', ''),
                            'source': article.get('domain', 'Unknown'),
                            'published_date': pub_date,
                            'tier': 2
                        })

                elif resp.status_code == 429:
                    logger.warning(f"Rate limited, waiting...")
                    time.sleep(5)

            except Exception as e:
                logger.debug(f"Error fetching '{keyword}': {e}")

            # Rate limiting
            time.sleep(0.2)

        return articles

    def fetch_date_range(
        self,
        start_date: str,
        end_date: str,
        keywords: List[str] = None
    ) -> int:
        """
        Fetch news for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            keywords: Optional custom keywords

        Returns:
            Total new articles saved
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        total_days = (end - start).days + 1
        total_saved = 0

        print(f"\nFetching historical news: {start_date} to {end_date}")
        print(f"Total days: {total_days}")
        print("="*60)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        fetched_date = datetime.utcnow().isoformat()

        current = start
        day_num = 0

        while current <= end:
            day_num += 1
            date_str = current.strftime('%Y-%m-%d')

            print(f"  [{day_num}/{total_days}] {date_str}...", end=" ", flush=True)

            articles = self._fetch_gdelt_day(date_str, keywords)

            saved = 0
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
                        saved += 1
                except Exception as e:
                    logger.debug(f"Insert error: {e}")

            conn.commit()
            total_saved += saved
            print(f"{len(articles)} found, {saved} new")

            current += timedelta(days=1)
            time.sleep(0.5)  # Be nice to GDELT

        conn.close()

        print("="*60)
        print(f"COMPLETE: {total_saved} new articles saved")

        return total_saved

    def get_stats(self) -> Dict:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM articles")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(published_date), MAX(published_date) FROM articles")
        row = cursor.fetchone()

        cursor.execute("SELECT COUNT(DISTINCT DATE(published_date)) FROM articles")
        days = cursor.fetchone()[0]

        cursor.execute("""
            SELECT DATE(published_date), COUNT(*)
            FROM articles
            GROUP BY DATE(published_date)
            ORDER BY DATE(published_date)
            LIMIT 10
        """)
        by_date = cursor.fetchall()

        conn.close()

        return {
            'total_articles': total,
            'date_range': {'min': row[0], 'max': row[1]},
            'days_covered': days,
            'sample_by_date': by_date
        }

    def mark_all_for_embedding(self):
        """Reset embedding status for re-embedding."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE articles SET is_embedded = 0")
        conn.commit()
        conn.close()


def fetch_for_simulation(
    sim_start: str = "2024-06-01",
    sim_end: str = "2024-07-01",
    buffer_days: int = 30
) -> Dict:
    """
    Fetch news for simulation period with buffer.

    Args:
        sim_start: Simulation start date
        sim_end: Simulation end date
        buffer_days: Days before sim_start to also fetch

    Returns:
        Stats dict
    """
    fetcher = HistoricalNewsFetcher()

    # Calculate fetch range with buffer
    start_dt = datetime.strptime(sim_start, '%Y-%m-%d') - timedelta(days=buffer_days)
    fetch_start = start_dt.strftime('%Y-%m-%d')

    print(f"\n{'='*60}")
    print("HISTORICAL FOREX NEWS FETCHER")
    print(f"{'='*60}")
    print(f"Simulation period: {sim_start} to {sim_end}")
    print(f"Fetching from: {fetch_start} (includes {buffer_days} day buffer)")

    # Fetch the news
    new_articles = fetcher.fetch_date_range(fetch_start, sim_end)

    # Show stats
    stats = fetcher.get_stats()

    print(f"\n{'='*60}")
    print("DATABASE STATUS")
    print(f"{'='*60}")
    print(f"Total articles: {stats['total_articles']}")
    print(f"Date range: {stats['date_range']['min'][:10]} to {stats['date_range']['max'][:10]}")
    print(f"Days covered: {stats['days_covered']}")
    print(f"\nArticles by date (first 10 days):")
    for date, count in stats['sample_by_date']:
        print(f"  {date}: {count}")

    return stats


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Fetch for simulation period (June 2024)
    # With 30-day buffer = May 2024
    fetch_for_simulation(
        sim_start="2024-06-01",
        sim_end="2024-07-15",
        buffer_days=30
    )
