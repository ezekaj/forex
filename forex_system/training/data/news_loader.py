"""Load news articles from SSD SQLite database."""

import sqlite3
from datetime import datetime
from typing import Optional

import pandas as pd


class NewsLoader:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)

    def get_date_range(self) -> tuple[str, str]:
        with self._connect() as conn:
            cursor = conn.execute("SELECT MIN(date), MAX(date) FROM news")
            row = cursor.fetchone()
            return row[0], row[1]

    def get_article_count(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM news")
            return cursor.fetchone()[0]

    def get_sources(self, limit: int = 50) -> list[tuple[str, int]]:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT source, COUNT(*) as cnt FROM news GROUP BY source ORDER BY cnt DESC LIMIT ?",
                (limit,),
            )
            return cursor.fetchall()

    def load_articles(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        query = "SELECT id, date, source, title, content, news_type FROM news WHERE 1=1"
        params: list = []

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def load_for_asset(
        self,
        keywords: tuple[str, ...],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load articles mentioning any of the asset's keywords in title."""
        return self.load_forex_relevant(keywords, start_date, end_date)

    def load_forex_relevant(
        self,
        forex_keywords: tuple[str, ...],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load articles that mention forex-related keywords in title.
        Uses title-only search to avoid hitting corrupted content pages."""
        like_conditions = " OR ".join(["title LIKE ?"] * len(forex_keywords))
        params: list = [f"%{kw}%" for kw in forex_keywords]

        query = f"""
            SELECT id, date, source, title, news_type
            FROM news
            WHERE ({like_conditions})
        """

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        try:
            with self._connect() as conn:
                df = pd.read_sql_query(query, conn, params=params)
        except Exception:
            # Fallback: load in date-chunked batches to work around corruption
            df = self._load_chunked_forex(forex_keywords, start_date, end_date)

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def _load_chunked_forex(
        self,
        forex_keywords: tuple[str, ...],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load forex news in monthly chunks to work around DB corruption."""
        import logging
        log = logging.getLogger(__name__)

        all_dfs = []
        # Generate monthly date ranges
        from datetime import datetime
        start = datetime.strptime(start_date or "2024-06-01", "%Y-%m-%d")
        end = datetime.strptime(end_date or "2026-02-01", "%Y-%m-%d")

        current = start
        while current < end:
            next_month = current.replace(day=28) + pd.Timedelta(days=4)
            next_month = next_month.replace(day=1)
            chunk_start = current.strftime("%Y-%m-%d")
            chunk_end = min(next_month, end).strftime("%Y-%m-%d")

            like_conditions = " OR ".join(["title LIKE ?"] * len(forex_keywords))
            params = [f"%{kw}%" for kw in forex_keywords]
            params.extend([chunk_start, chunk_end])

            query = f"""
                SELECT id, date, source, title, news_type
                FROM news
                WHERE ({like_conditions}) AND date >= ? AND date < ?
                ORDER BY date
            """
            try:
                with self._connect() as conn:
                    chunk_df = pd.read_sql_query(query, conn, params=params)
                    all_dfs.append(chunk_df)
            except Exception as e:
                log.warning(f"Skipping corrupted chunk {chunk_start}-{chunk_end}: {e}")

            current = next_month

        if not all_dfs:
            return pd.DataFrame()
        return pd.concat(all_dfs, ignore_index=True)

    def load_articles_batch(
        self,
        batch_size: int = 10000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """Generator that yields batches of articles for memory-efficient processing."""
        query = "SELECT id, date, source, title, content, news_type FROM news WHERE 1=1"
        params: list = []

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        with self._connect() as conn:
            offset = 0
            while True:
                batch_query = f"{query} LIMIT ? OFFSET ?"
                batch_params = params + [batch_size, offset]
                df = pd.read_sql_query(batch_query, conn, params=batch_params)
                if df.empty:
                    break
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                yield df
                offset += batch_size

    def get_stats(self) -> dict:
        min_date, max_date = self.get_date_range()
        count = self.get_article_count()
        return {
            "total_articles": count,
            "date_range": (min_date, max_date),
        }
