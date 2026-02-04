#!/usr/bin/env python3
"""
MEGA HISTORICAL NEWS FETCHER
============================
Fetches ALL historical news from May 2024 to present.
Handles GDELT rate limits with smart resumption.

Usage:
    python3 mega_historical_fetcher.py          # Fetch missing dates
    python3 mega_historical_fetcher.py merge    # Merge from investment-monitor
    python3 mega_historical_fetcher.py stats    # Show coverage stats
"""

import sqlite3
import requests
import hashlib
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Set

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DB_PATH = PROJECT_DIR / "databases" / "news.db"
INVESTMENT_MONITOR_DB = Path.home() / "Desktop" / "investment-monitor" / "historical_news.db"

# GDELT API
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Forex + Financial keywords for comprehensive coverage
KEYWORDS = [
    # Forex specific
    "forex EUR USD", "forex GBP USD", "forex USD JPY",
    "EURUSD", "GBPUSD", "USDJPY", "currency exchange",
    "dollar index", "euro dollar", "pound dollar", "yen dollar",

    # Central banks (critical for forex)
    "federal reserve", "fed rate", "fed policy", "Jerome Powell",
    "ECB rate", "ECB policy", "Christine Lagarde",
    "Bank of England", "BOE rate",
    "Bank of Japan", "BOJ policy",

    # Economic indicators
    "inflation data", "CPI report", "jobs report", "nonfarm payrolls",
    "GDP growth", "unemployment rate", "retail sales",
    "PMI manufacturing", "consumer confidence",

    # Market moves
    "stock market", "S&P 500", "nasdaq", "dow jones",
    "market rally", "market crash", "bull market", "bear market",

    # Macro events
    "interest rate decision", "rate hike", "rate cut",
    "quantitative easing", "monetary policy",
    "trade war", "tariff", "sanctions",

    # Risk sentiment
    "risk appetite", "safe haven", "flight to safety",
    "market volatility", "VIX",
]


def init_db():
    """Initialize database with proper schema."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
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
        );

        CREATE INDEX IF NOT EXISTS idx_published ON articles(published_date);
        CREATE INDEX IF NOT EXISTS idx_source ON articles(source);
        CREATE INDEX IF NOT EXISTS idx_date_only ON articles(DATE(published_date));

        CREATE TABLE IF NOT EXISTS fetch_progress (
            date TEXT PRIMARY KEY,
            articles_count INTEGER,
            fetched_at TEXT
        );
    """)
    conn.commit()
    conn.close()


def get_fetched_dates() -> Set[str]:
    """Get dates that have already been fetched."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT date FROM fetch_progress")
    dates = {row[0] for row in cursor.fetchall()}
    conn.close()
    return dates


def generate_article_id(url: str, title: str) -> str:
    """Generate unique article ID."""
    content = f"{url}{title}".encode('utf-8')
    return hashlib.md5(content).hexdigest()


def fetch_gdelt_day(date_str: str, max_retries: int = 3) -> List[Dict]:
    """Fetch news from GDELT for a specific date with retry logic."""
    articles = []
    seen_titles = set()

    for keyword in KEYWORDS:
        for attempt in range(max_retries):
            try:
                params = {
                    'query': f'{keyword} sourcelang:eng',
                    'mode': 'artlist',
                    'maxrecords': 75,
                    'format': 'json',
                    'startdatetime': date_str.replace('-', '') + '000000',
                    'enddatetime': date_str.replace('-', '') + '235959'
                }

                resp = requests.get(GDELT_URL, params=params, timeout=30)

                if resp.status_code == 200:
                    data = resp.json()
                    for article in data.get('articles', []):
                        title = article.get('title', '').strip()
                        if not title or title in seen_titles:
                            continue
                        seen_titles.add(title)

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
                            'article_id': generate_article_id(article.get('url', ''), title),
                            'title': title,
                            'content': title,  # GDELT only provides title
                            'url': article.get('url', ''),
                            'source': article.get('domain', 'Unknown'),
                            'published_date': pub_date,
                        })
                    break  # Success, move to next keyword

                elif resp.status_code == 429:
                    # Rate limited - wait longer
                    wait_time = 10 * (attempt + 1)
                    print(f"    Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    break

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                continue

        # Delay between keywords to avoid rate limits
        time.sleep(0.5)

    return articles


def save_articles(articles: List[Dict], date_str: str) -> int:
    """Save articles to database and mark date as fetched."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    fetched_date = datetime.now().isoformat()

    saved = 0
    for article in articles:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO articles
                (article_id, title, content, url, source, published_date, fetched_date, tier)
                VALUES (?, ?, ?, ?, ?, ?, ?, 2)
            """, (
                article['article_id'],
                article['title'],
                article['content'],
                article['url'],
                article['source'],
                article['published_date'],
                fetched_date
            ))
            if cursor.rowcount > 0:
                saved += 1
        except:
            pass

    # Mark date as fetched
    cursor.execute("""
        INSERT OR REPLACE INTO fetch_progress (date, articles_count, fetched_at)
        VALUES (?, ?, ?)
    """, (date_str, len(articles), fetched_date))

    conn.commit()
    conn.close()
    return saved


def fetch_date_range(start_date: str, end_date: str):
    """Fetch news for a date range, skipping already-fetched dates."""
    init_db()

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    # Get already-fetched dates
    fetched_dates = get_fetched_dates()

    # Calculate dates to fetch
    all_dates = []
    current = start
    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        if date_str not in fetched_dates:
            all_dates.append(date_str)
        current += timedelta(days=1)

    total_days = (end - start).days + 1
    already_done = total_days - len(all_dates)

    print(f"\n{'='*60}")
    print(f"MEGA HISTORICAL NEWS FETCHER")
    print(f"{'='*60}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Total days: {total_days}")
    print(f"Already fetched: {already_done}")
    print(f"Remaining: {len(all_dates)}")
    print(f"{'='*60}\n")

    if not all_dates:
        print("All dates already fetched!")
        return

    total_saved = 0
    for i, date_str in enumerate(all_dates, 1):
        print(f"[{i}/{len(all_dates)}] {date_str}...", end=" ", flush=True)

        articles = fetch_gdelt_day(date_str)
        saved = save_articles(articles, date_str)
        total_saved += saved

        print(f"{len(articles)} found, {saved} new")

        # Longer delay between days to be nice to GDELT
        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"COMPLETE: {total_saved} new articles saved")
    print(f"{'='*60}")


def merge_from_investment_monitor():
    """Merge existing news from investment-monitor historical_news.db."""
    if not INVESTMENT_MONITOR_DB.exists():
        print(f"Investment-monitor DB not found: {INVESTMENT_MONITOR_DB}")
        return

    init_db()

    print(f"\n{'='*60}")
    print(f"MERGING FROM INVESTMENT-MONITOR")
    print(f"{'='*60}")

    # Connect to both databases
    src_conn = sqlite3.connect(INVESTMENT_MONITOR_DB)
    dst_conn = sqlite3.connect(DB_PATH)

    # Get count from source
    src_count = src_conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    print(f"Source articles: {src_count}")

    # Fetch all from source
    cursor = src_conn.execute("SELECT date, source, title, url FROM news")

    merged = 0
    fetched_date = datetime.now().isoformat()

    batch = []
    for row in cursor:
        date, source, title, url = row
        if not title:
            continue

        article_id = generate_article_id(url or '', title)
        pub_date = f"{date}T12:00:00" if date else fetched_date

        batch.append((
            article_id, title, title, url or '', source or 'Unknown',
            pub_date, fetched_date, 2
        ))

        if len(batch) >= 1000:
            dst_conn.executemany("""
                INSERT OR IGNORE INTO articles
                (article_id, title, content, url, source, published_date, fetched_date, tier)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)
            merged += dst_conn.total_changes
            dst_conn.commit()
            print(f"  Merged {merged}...")
            batch = []

    # Final batch
    if batch:
        dst_conn.executemany("""
            INSERT OR IGNORE INTO articles
            (article_id, title, content, url, source, published_date, fetched_date, tier)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, batch)
        dst_conn.commit()

    # Get final count
    final_count = dst_conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]

    src_conn.close()
    dst_conn.close()

    print(f"\n{'='*60}")
    print(f"MERGE COMPLETE")
    print(f"Total articles now: {final_count}")
    print(f"{'='*60}")


def show_stats():
    """Show database coverage statistics."""
    init_db()
    conn = sqlite3.connect(DB_PATH)

    total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]

    date_range = conn.execute("""
        SELECT MIN(DATE(published_date)), MAX(DATE(published_date))
        FROM articles
    """).fetchone()

    by_month = conn.execute("""
        SELECT strftime('%Y-%m', published_date) as month, COUNT(*) as cnt
        FROM articles
        GROUP BY month
        ORDER BY month
    """).fetchall()

    fetched_days = conn.execute("SELECT COUNT(*) FROM fetch_progress").fetchone()[0]

    conn.close()

    print(f"\n{'='*60}")
    print(f"NEWS DATABASE COVERAGE")
    print(f"{'='*60}")
    print(f"Total articles: {total:,}")
    print(f"Date range: {date_range[0]} to {date_range[1]}")
    print(f"Days explicitly fetched: {fetched_days}")

    print(f"\nArticles by month:")
    for month, count in by_month:
        bar = '#' * min(50, count // 100)
        print(f"  {month}: {count:>6,} {bar}")

    # Show gaps
    if by_month:
        print(f"\nCoverage analysis:")
        start = datetime.strptime(by_month[0][0], '%Y-%m')
        end = datetime.strptime(by_month[-1][0], '%Y-%m')

        current = start
        months_with_data = {m[0] for m in by_month}
        gaps = []

        while current <= end:
            month_str = current.strftime('%Y-%m')
            if month_str not in months_with_data:
                gaps.append(month_str)
            current = (current.replace(day=1) + timedelta(days=32)).replace(day=1)

        if gaps:
            print(f"  Missing months: {', '.join(gaps)}")
        else:
            print(f"  No gaps in monthly coverage!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'merge':
            merge_from_investment_monitor()
        elif cmd == 'stats':
            show_stats()
        elif cmd == 'fetch':
            # Fetch specific range
            start = sys.argv[2] if len(sys.argv) > 2 else '2024-05-01'
            end = sys.argv[3] if len(sys.argv) > 3 else '2025-11-16'
            fetch_date_range(start, end)
    else:
        # Default: fetch the gap (May 2024 to Nov 2025)
        fetch_date_range('2024-05-01', '2025-11-16')
