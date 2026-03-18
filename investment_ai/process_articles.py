"""
Step 1: Process all 4,082,690 news articles.

For each article:
  1. Extract tickers (keyword matching from ASSET_REGISTRY)
  2. Score sentiment (DistilRoBERTa, cached)
  3. Classify event type (keyword-based from true_backtester)
  4. Compute importance score

Output: article_features.db with one row per article-ticker pair.

Usage:
    python3 -m investment_ai.process_articles          # process all
    python3 -m investment_ai.process_articles --test    # process 1000 articles for testing
"""

import logging
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from forex_system.training.config import ASSET_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("process_articles")

NEWS_DB = "/home/user/Backup-SSD/investment-monitor/historical_news.db"
OUTPUT_DB = str(Path(__file__).parent / "article_features.db")
SENTIMENT_CACHE = str(Path(__file__).parent.parent / "forex_system/training/artifacts/sentiment_cache_v2.db")

# ── Ticker keyword map (from ASSET_REGISTRY) ──

TICKER_KEYWORDS: dict[str, list[str]] = {}
for sym, cfg in ASSET_REGISTRY.items():
    if cfg.asset_class in ("stock", "crypto"):
        TICKER_KEYWORDS[sym] = [kw.lower() for kw in cfg.keywords]

# ── Event classification (from true_backtester._classify_event) ──

EVENT_PATTERNS = {
    "earnings": ["earnings", "quarter", "profit", "revenue", "eps", "beat", "miss", "guidance"],
    "merger": ["merger", "acquire", "acquisition", "buyout", "deal", "takeover"],
    "product": ["product", "launch", "release", "announce", "unveil", "innovation"],
    "fda": ["fda", "approval", "drug", "trial", "clinical", "pharma"],
    "analyst": ["upgrade", "downgrade", "price target", "analyst", "rating"],
    "insider": ["insider", "form 4", "bought shares", "sold shares", "ceo buy"],
    "regulation": ["regulation", "sec", "ftc", "antitrust", "lawsuit", "investigation", "fine"],
    "layoff": ["layoff", "cut", "fire", "restructur", "downsiz"],
    "executive": ["ceo", "cfo", "executive", "appoint", "resign", "step down"],
    "macro": ["fed", "inflation", "rate", "economy", "gdp", "jobs", "unemployment", "tariff"],
    "china": ["china", "tariff", "trade war", "beijing", "chinese"],
    "partnership": ["partner", "collaborat", "alliance", "joint venture"],
}


def classify_event(title: str) -> str:
    """Classify article event type from title."""
    title_lower = title.lower()
    for event_type, keywords in EVENT_PATTERNS.items():
        if any(kw in title_lower for kw in keywords):
            return event_type
    return "general"


def extract_tickers(title: str, content: str = "") -> list[str]:
    """Find which tickers are mentioned in the article."""
    content = str(content or "")
    text = f"{title} {content[:500]}".lower()
    found = []
    for ticker, keywords in TICKER_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            found.append(ticker)
    return found


def compute_importance(event_type: str, n_tickers: int, source: str) -> int:
    """Compute importance score 1-10."""
    # Base by event type
    type_scores = {
        "earnings": 8, "merger": 9, "fda": 9, "analyst": 6,
        "insider": 7, "regulation": 6, "layoff": 5, "executive": 5,
        "macro": 7, "china": 6, "partnership": 5, "product": 5, "general": 3,
    }
    score = type_scores.get(event_type, 3)

    # Boost for specific tickers
    if n_tickers >= 1:
        score += 1
    if n_tickers >= 3:
        score += 1

    # Boost for quality sources
    quality_sources = {"reuters", "bloomberg", "cnbc", "wsj", "ft.com", "marketwatch", "yahoo"}
    if source and any(qs in source.lower() for qs in quality_sources):
        score += 1

    return min(score, 10)


def init_output_db():
    """Create output database."""
    with sqlite3.connect(OUTPUT_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS article_features (
                id INTEGER PRIMARY KEY,
                article_id INTEGER,
                date TEXT,
                ticker TEXT,
                sentiment REAL,
                event_type TEXT,
                importance INTEGER,
                source TEXT,
                title TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_af_date ON article_features(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_af_ticker ON article_features(ticker)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_af_date_ticker ON article_features(date, ticker)")
    log.info(f"Output DB initialized: {OUTPUT_DB}")


def count_processed():
    """Count already processed articles."""
    try:
        with sqlite3.connect(OUTPUT_DB) as conn:
            return conn.execute("SELECT COUNT(DISTINCT article_id) FROM article_features").fetchone()[0]
    except Exception:
        return 0


def process_batch(articles: pd.DataFrame, scorer) -> list[tuple]:
    """Process a batch of articles and return rows for insertion."""
    # Score sentiment
    titles = articles["title"].fillna("").tolist()
    scores = scorer.score_batch(titles)

    rows = []
    for i, (_, article) in enumerate(articles.iterrows()):
        title = article.get("title", "") or ""
        content = article.get("content", "") or ""
        source = article.get("source", "") or ""
        date = str(article.get("date", ""))[:10]
        article_id = int(article.get("id", 0))

        if len(title) < 10:
            continue

        # Extract tickers
        tickers = extract_tickers(title, content)
        if not tickers:
            continue  # Skip articles that don't mention any tracked stock

        # Classify event
        event_type = classify_event(title)

        # Sentiment
        sentiment = scores[i]["score"]  # -1 to +1

        # Importance
        importance = compute_importance(event_type, len(tickers), source)

        # One row per ticker mentioned
        for ticker in tickers:
            rows.append((
                article_id, date, ticker, sentiment, event_type,
                importance, source, title[:200],
            ))

    return rows


def run(test_mode: bool = False):
    """Main processing loop."""
    init_output_db()

    already_done = count_processed()
    log.info(f"Already processed: {already_done} articles")

    # Load sentiment scorer
    from forex_system.models.sentiment import SentimentScorer
    scorer = SentimentScorer(cache_db=SENTIMENT_CACHE, device="cpu", batch_size=256)
    log.info(f"Sentiment scorer loaded (cache: {scorer.get_cache_stats()})")

    # Process in batches from news DB
    batch_size = 10000
    if test_mode:
        batch_size = 1000

    news_conn = sqlite3.connect(f"file:{NEWS_DB}?mode=ro", uri=True)
    total = news_conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    log.info(f"Total articles in news DB: {total:,}")

    offset = 0
    total_rows = 0
    t0 = time.perf_counter()

    while True:
        query = f"""
            SELECT id, date, source, title, content FROM news
            WHERE title IS NOT NULL AND LENGTH(title) > 10
            ORDER BY id
            LIMIT {batch_size} OFFSET {offset}
        """
        batch = pd.read_sql_query(query, news_conn)
        if batch.empty:
            break

        rows = process_batch(batch, scorer)

        if rows:
            with sqlite3.connect(OUTPUT_DB) as out_conn:
                out_conn.executemany(
                    "INSERT OR IGNORE INTO article_features "
                    "(article_id, date, ticker, sentiment, event_type, importance, source, title) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
            total_rows += len(rows)

        offset += batch_size
        elapsed = time.perf_counter() - t0
        rate = offset / elapsed if elapsed > 0 else 0
        remaining = (total - offset) / rate if rate > 0 else 0

        log.info(
            f"Processed {offset:,}/{total:,} articles ({offset/total*100:.1f}%) | "
            f"{total_rows:,} ticker-article pairs | "
            f"{rate:.0f} articles/s | ETA: {remaining/60:.0f}m"
        )

        if test_mode and offset >= 1000:
            break

    news_conn.close()
    elapsed = time.perf_counter() - t0

    log.info(f"\nDONE: {offset:,} articles processed → {total_rows:,} ticker-article pairs in {elapsed/60:.1f}m")

    # Show summary
    with sqlite3.connect(OUTPUT_DB) as conn:
        total_pairs = conn.execute("SELECT COUNT(*) FROM article_features").fetchone()[0]
        unique_dates = conn.execute("SELECT COUNT(DISTINCT date) FROM article_features").fetchone()[0]
        unique_tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM article_features").fetchone()[0]

        log.info(f"Summary: {total_pairs:,} pairs, {unique_dates} dates, {unique_tickers} tickers")

        # Event type distribution
        events = conn.execute(
            "SELECT event_type, COUNT(*) as cnt FROM article_features GROUP BY event_type ORDER BY cnt DESC"
        ).fetchall()
        log.info("Event distribution:")
        for evt, cnt in events:
            log.info(f"  {evt:15s}: {cnt:,}")

        # Top tickers
        tickers = conn.execute(
            "SELECT ticker, COUNT(*) as cnt FROM article_features GROUP BY ticker ORDER BY cnt DESC LIMIT 10"
        ).fetchall()
        log.info("Top tickers:")
        for tkr, cnt in tickers:
            log.info(f"  {tkr:10s}: {cnt:,}")


if __name__ == "__main__":
    test = "--test" in sys.argv
    run(test_mode=test)
