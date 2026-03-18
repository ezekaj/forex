"""
Process a CHUNK of news articles through Qwen. Designed to run 4 instances in parallel.

Usage:
    python3 -m investment_ai.process_qwen_chunk 1   # chunk 1 of 4 (IDs 1-1091432)
    python3 -m investment_ai.process_qwen_chunk 2   # chunk 2 of 4
    python3 -m investment_ai.process_qwen_chunk 3   # chunk 3 of 4
    python3 -m investment_ai.process_qwen_chunk 4   # chunk 4 of 4

Each writes to its own DB: qwen_chunk_1.db, qwen_chunk_2.db, etc.
No conflicts between processes.
"""

import json
import logging
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

NEWS_DB = "/home/user/Backup-SSD/investment-monitor/historical_news.db"
VLLM_URL = "http://localhost:8000/v1/completions"
MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"
WORKERS = 16  # 4 processes × 16 = 64 total concurrent

# ID ranges for 4 chunks
CHUNKS = {
    1: (1, 1091433),
    2: (1091433, 2182865),
    3: (2182865, 3274297),
    4: (3274297, 4365733),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [chunk%(chunk)s] %(message)s",
)


def get_output_db(chunk_id: int) -> str:
    return str(Path(__file__).parent / f"qwen_chunk_{chunk_id}.db")


def init_db(db_path: str):
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS article_analysis (
                article_id INTEGER PRIMARY KEY,
                date TEXT,
                title TEXT,
                source TEXT,
                tickers TEXT,
                sentiment TEXT,
                event_type TEXT,
                importance INTEGER,
                raw_json TEXT
            )
        """)


def get_processed_count(db_path: str) -> int:
    try:
        with sqlite3.connect(db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM article_analysis").fetchone()[0]
    except Exception:
        return 0


def get_processed_ids(db_path: str) -> set:
    try:
        with sqlite3.connect(db_path) as conn:
            return {r[0] for r in conn.execute("SELECT article_id FROM article_analysis").fetchall()}
    except Exception:
        return set()


def classify_one(title: str):
    json_prefix = '{"sentiment":"'
    t = title[:100]
    prompt = (
        "<|im_start|>system\n"
        "You classify news articles for market impact analysis. "
        "Output ONE JSON object. Choose sentiment from: BULLISH, BEARISH, NEUTRAL. "
        "Choose event_type from: earnings, merger, product, fda, analyst, regulation, "
        "layoff, macro, geopolitical, tech, energy, consumer, crime, weather, political, social, general. "
        "Set importance 1-10 where 1=irrelevant to markets, 5=moderate, 8=significant, 10=market-moving. "
        "For tickers, list stock symbols directly affected. Empty array if none.<|im_end|>\n"
        f"<|im_start|>user\n{t}<|im_end|>\n"
        "<|im_start|>assistant\n"
        + json_prefix
    )
    try:
        resp = requests.post(VLLM_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 80,
            "temperature": 0.0,
            "stop": ["<|im_end|>", "\n\n"],
        }, timeout=30)
        text = json_prefix + resp.json()["choices"][0]["text"]
        try:
            return json.loads(text.split("}")[0] + "}")
        except Exception:
            m = re.search(r"\{.*?\}", text, re.DOTALL)
            return json.loads(m.group()) if m else None
    except Exception:
        return None


def run_chunk(chunk_id: int):
    log = logging.getLogger()
    extra = {"chunk": chunk_id}

    id_start, id_end = CHUNKS[chunk_id]
    db_path = get_output_db(chunk_id)
    init_db(db_path)

    processed_ids = get_processed_ids(db_path)
    log.info(f"Chunk {chunk_id}: IDs {id_start}-{id_end}, already done: {len(processed_ids):,}", extra=extra)

    news_conn = sqlite3.connect(f"file:{NEWS_DB}?mode=ro", uri=True)
    total = news_conn.execute(
        "SELECT COUNT(*) FROM news WHERE id >= ? AND id < ? AND title IS NOT NULL AND LENGTH(title) > 10",
        (id_start, id_end)
    ).fetchone()[0]
    remaining = total - len(processed_ids)
    log.info(f"Total in chunk: {total:,}, remaining: {remaining:,}", extra=extra)

    batch_size = WORKERS * 4
    offset = 0
    saved = 0
    t0 = time.perf_counter()

    while True:
        rows = news_conn.execute(
            "SELECT id, date, source, title FROM news "
            "WHERE id >= ? AND id < ? AND title IS NOT NULL AND LENGTH(title) > 10 "
            "ORDER BY id LIMIT ? OFFSET ?",
            (id_start, id_end, batch_size, offset)
        ).fetchall()

        if not rows:
            break

        offset += batch_size
        batch = [(r[0], r[1], r[2], r[3]) for r in rows if r[0] not in processed_ids]
        if not batch:
            continue

        # Process concurrently
        results = []
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {}
            for aid, date, source, title in batch:
                futures[ex.submit(classify_one, title)] = (aid, date, source, title)

            for f in as_completed(futures):
                aid, date, source, title = futures[f]
                obj = f.result()
                if obj:
                    tickers = obj.get("tickers", [])
                    if isinstance(tickers, str):
                        tickers = [tickers] if tickers else []

                    results.append((
                        aid, str(date)[:10], title[:200], source or "",
                        json.dumps(tickers),
                        obj.get("sentiment", "NEUTRAL"),
                        obj.get("event_type", "general"),
                        int(obj.get("importance", 5)),
                        json.dumps(obj),
                    ))

        # Save batch
        if results:
            with sqlite3.connect(db_path) as conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO article_analysis "
                    "(article_id, date, title, source, tickers, sentiment, event_type, importance, raw_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    results,
                )
            saved += len(results)
            processed_ids.update(r[0] for r in results)

        # Log every 30s
        elapsed = time.perf_counter() - t0
        if elapsed > 0 and saved > 0 and (saved % (batch_size * 5) == 0 or saved < batch_size * 2):
            rate = saved / elapsed
            eta = (remaining - saved) / rate if rate > 0 else 0
            log.info(
                f"Saved: {saved:,}/{remaining:,} ({saved/remaining*100:.1f}%) | "
                f"Rate: {rate:.1f}/s | ETA: {eta/3600:.1f}h",
                extra=extra,
            )

    news_conn.close()
    elapsed = time.perf_counter() - t0
    log.info(f"DONE: {saved:,} articles in {elapsed/3600:.1f}h", extra=extra)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("1", "2", "3", "4"):
        print("Usage: python -m investment_ai.process_qwen_chunk {1|2|3|4}")
        sys.exit(1)
    chunk_id = int(sys.argv[1])
    run_chunk(chunk_id)
