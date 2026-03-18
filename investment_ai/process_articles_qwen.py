"""
Process ALL news articles through Qwen3.5-35B via vLLM.

Uses the completions API with JSON prefilling to bypass thinking chain.
Concurrent processing with 16 workers for throughput.

Expected rate: ~6-14 articles/sec → 4M articles in 3-8 days.
Resumable: skips already-processed articles.

Usage:
    python3 -m investment_ai.process_articles_qwen           # full run
    python3 -m investment_ai.process_articles_qwen --test    # 100 articles
    python3 -m investment_ai.process_articles_qwen --workers 8  # fewer workers
"""

import json
import logging
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("qwen_process")

NEWS_DB = "/home/user/Backup-SSD/investment-monitor/historical_news.db"
OUTPUT_DB = str(Path(__file__).parent / "qwen_article_features.db")
VLLM_URL = "http://localhost:8000/v1/completions"
MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"


def init_db():
    with sqlite3.connect(OUTPUT_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS article_analysis (
                article_id INTEGER PRIMARY KEY,
                date TEXT,
                title TEXT,
                source TEXT,
                tickers TEXT,       -- JSON array
                sentiment TEXT,     -- BULLISH/BEARISH/NEUTRAL
                event_type TEXT,    -- earnings/merger/product/fda/analyst/regulation/layoff/macro/general
                importance INTEGER, -- 1-10
                raw_json TEXT,      -- full LLM response
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_qa_date ON article_analysis(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_qa_sentiment ON article_analysis(sentiment)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_qa_event ON article_analysis(event_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_qa_importance ON article_analysis(importance)")


def get_processed_ids() -> set:
    """Get set of already-processed article IDs."""
    try:
        with sqlite3.connect(OUTPUT_DB) as conn:
            rows = conn.execute("SELECT article_id FROM article_analysis").fetchall()
            return {r[0] for r in rows}
    except Exception:
        return set()


def classify_one(article_id: int, title: str, date: str, source: str) -> dict | None:
    """Classify a single article via Qwen."""
    json_prefix = '{"tickers":["'
    prompt = (
        "<|im_start|>user\n"
        "Classify this news for stock trading. Output ONE JSON object with these exact fields:\n"
        '- tickers: array of stock symbols mentioned (e.g. ["AAPL", "MSFT"])\n'
        "- sentiment: BULLISH or BEARISH or NEUTRAL\n"
        "- event_type: one of earnings/merger/product/fda/analyst/regulation/layoff/macro/general\n"
        "- importance: integer 1-10 (10 = massive market impact like acquisition or earnings)\n"
        "\n"
        f"Headline: {title}<|im_end|>\n"
        "<|im_start|>assistant\n"
        + json_prefix
    )

    try:
        resp = requests.post(VLLM_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 120,
            "temperature": 0.0,
            "stop": ["<|im_end|>", "\n\n"],
        }, timeout=30)

        text = json_prefix + resp.json()["choices"][0]["text"]

        # Parse JSON
        try:
            obj = json.loads(text.split("}")[0] + "}")
        except Exception:
            m = re.search(r"\{.*?\}", text, re.DOTALL)
            if m:
                obj = json.loads(m.group())
            else:
                return None

        return {
            "article_id": article_id,
            "date": date,
            "title": title[:200],
            "source": source or "",
            "tickers": json.dumps(obj.get("tickers", [])),
            "sentiment": obj.get("sentiment", "NEUTRAL"),
            "event_type": obj.get("event_type", "general"),
            "importance": int(obj.get("importance", 5)),
            "raw_json": json.dumps(obj),
        }
    except Exception as e:
        return None


def process_batch(batch: list[tuple], workers: int = 16) -> list[dict]:
    """Process a batch of articles concurrently."""
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for article_id, date, source, title in batch:
            f = executor.submit(classify_one, article_id, title, str(date)[:10], source)
            futures[f] = article_id

        for f in as_completed(futures):
            result = f.result()
            if result:
                results.append(result)

    return results


def save_results(results: list[dict]):
    """Save batch of results to DB."""
    if not results:
        return
    with sqlite3.connect(OUTPUT_DB) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO article_analysis "
            "(article_id, date, title, source, tickers, sentiment, event_type, importance, raw_json) "
            "VALUES (:article_id, :date, :title, :source, :tickers, :sentiment, :event_type, :importance, :raw_json)",
            results,
        )


def run(test_mode: bool = False, workers: int = 16):
    init_db()

    processed_ids = get_processed_ids()
    log.info(f"Already processed: {len(processed_ids):,} articles")

    news_conn = sqlite3.connect(f"file:{NEWS_DB}?mode=ro", uri=True)
    total = news_conn.execute("SELECT COUNT(*) FROM news WHERE title IS NOT NULL AND LENGTH(title) > 10").fetchone()[0]
    log.info(f"Total articles in news DB: {total:,}")
    log.info(f"Remaining: ~{total - len(processed_ids):,}")
    log.info(f"Workers: {workers}, Model: {MODEL}")
    log.info(f"vLLM: {VLLM_URL}")

    # Verify vLLM is running
    try:
        r = requests.get("http://localhost:8000/v1/models", timeout=5)
        log.info(f"vLLM status: OK ({r.json()['data'][0]['id']})")
    except Exception:
        log.error("vLLM not running at localhost:8000!")
        return

    batch_size = workers * 4  # process 4x workers at a time
    if test_mode:
        batch_size = min(100, batch_size)

    offset = 0
    total_processed = len(processed_ids)
    total_saved = 0
    t0 = time.perf_counter()
    last_log = t0

    while True:
        query = f"""
            SELECT id, date, source, title FROM news
            WHERE title IS NOT NULL AND LENGTH(title) > 10
            ORDER BY id
            LIMIT {batch_size} OFFSET {offset}
        """
        rows = news_conn.execute(query).fetchall()
        if not rows:
            break

        # Filter out already processed
        batch = [(r[0], r[1], r[2], r[3]) for r in rows if r[0] not in processed_ids]
        offset += batch_size

        if not batch:
            continue

        # Process
        results = process_batch(batch, workers=workers)
        save_results(results)
        total_saved += len(results)
        total_processed += len(batch)

        # Log progress periodically
        now = time.perf_counter()
        if now - last_log >= 30 or test_mode:  # every 30 seconds
            elapsed = now - t0
            rate = total_saved / elapsed if elapsed > 0 else 0
            remaining = (total - total_processed) / rate if rate > 0 else 0
            pct = total_processed / total * 100

            # Stats on sentiment distribution
            with sqlite3.connect(OUTPUT_DB) as conn:
                dist = conn.execute(
                    "SELECT sentiment, COUNT(*) FROM article_analysis GROUP BY sentiment"
                ).fetchall()
            dist_str = ", ".join(f"{s}:{c}" for s, c in dist)

            log.info(
                f"Progress: {total_processed:,}/{total:,} ({pct:.1f}%) | "
                f"Saved: {total_saved:,} | "
                f"Rate: {rate:.1f}/s | "
                f"ETA: {remaining/3600:.1f}h | "
                f"Dist: {dist_str}"
            )
            last_log = now

        if test_mode and total_processed >= 100:
            break

    news_conn.close()
    elapsed = time.perf_counter() - t0

    log.info(f"\nDONE: {total_processed:,} articles → {total_saved:,} analyzed in {elapsed/3600:.1f}h")

    # Final summary
    with sqlite3.connect(OUTPUT_DB) as conn:
        total_rows = conn.execute("SELECT COUNT(*) FROM article_analysis").fetchone()[0]
        events = conn.execute(
            "SELECT event_type, COUNT(*) FROM article_analysis GROUP BY event_type ORDER BY COUNT(*) DESC"
        ).fetchall()
        sentiments = conn.execute(
            "SELECT sentiment, COUNT(*) FROM article_analysis GROUP BY sentiment ORDER BY COUNT(*) DESC"
        ).fetchall()
        avg_imp = conn.execute("SELECT AVG(importance) FROM article_analysis").fetchone()[0]
        high_imp = conn.execute("SELECT COUNT(*) FROM article_analysis WHERE importance >= 8").fetchone()[0]

    log.info(f"Total analyzed: {total_rows:,}")
    log.info(f"Avg importance: {avg_imp:.1f}")
    log.info(f"High importance (≥8): {high_imp:,}")
    log.info(f"Sentiment: {dict(sentiments)}")
    log.info(f"Events: {dict(events)}")


if __name__ == "__main__":
    test = "--test" in sys.argv
    workers = 16
    for i, arg in enumerate(sys.argv):
        if arg == "--workers" and i + 1 < len(sys.argv):
            workers = int(sys.argv[i + 1])
    run(test_mode=test, workers=workers)
