"""Download and cache stock/crypto price data from Yahoo Finance."""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from forex_system.training.config import TrainingConfig, ASSET_REGISTRY, get_symbols_by_class

log = logging.getLogger(__name__)


class YahooDownloader:
    """
    Downloads OHLCV data from Yahoo Finance and caches to local SQLite.
    First call downloads; subsequent calls read from cache.
    """

    def __init__(self, cache_db: str = None):
        config = TrainingConfig()
        self.cache_db = cache_db or config.YAHOO_CACHE_DB_PATH
        Path(self.cache_db).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_prices (
                    symbol TEXT NOT NULL,
                    datetime TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL, volume REAL,
                    UNIQUE(symbol, datetime)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_yahoo_daily_symbol ON daily_prices(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_yahoo_daily_dt ON daily_prices(datetime)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hourly_prices (
                    symbol TEXT NOT NULL,
                    datetime TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL, volume REAL,
                    UNIQUE(symbol, datetime)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_yahoo_hourly_symbol ON hourly_prices(symbol)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS download_log (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    last_download TEXT NOT NULL,
                    bar_count INTEGER,
                    UNIQUE(symbol, timeframe)
                )
            """)

    def download_symbol(
        self,
        symbol: str,
        period: str = "5y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Download OHLCV for a single symbol from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                log.warning(f"No data returned for {symbol} ({interval})")
                return pd.DataFrame()

            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            df = df[["open", "high", "low", "close", "volume"]]
            df.index = df.index.tz_localize(None) if df.index.tz else df.index
            return df
        except Exception as e:
            log.error(f"Failed to download {symbol}: {e}")
            return pd.DataFrame()

    def download_and_cache(
        self,
        symbol: str,
        daily_period: str = "5y",
        hourly_period: str = "1y",
        force: bool = False,
    ) -> dict:
        """Download daily + hourly data and cache to SQLite."""
        stats = {"symbol": symbol, "daily_bars": 0, "hourly_bars": 0}

        # Check if already cached (within last 24h)
        if not force and self._is_recent(symbol, "1d"):
            stats["daily_bars"] = self._get_cached_count(symbol, "1d")
            stats["hourly_bars"] = self._get_cached_count(symbol, "1h")
            stats["cached"] = True
            return stats

        # Download daily
        daily = self.download_symbol(symbol, period=daily_period, interval="1d")
        if not daily.empty:
            self._save_to_cache(symbol, daily, "daily_prices")
            stats["daily_bars"] = len(daily)
            self._log_download(symbol, "1d", len(daily))

        # Download hourly (Yahoo limits to ~730 days)
        hourly = self.download_symbol(symbol, period=hourly_period, interval="1h")
        if not hourly.empty:
            self._save_to_cache(symbol, hourly, "hourly_prices")
            stats["hourly_bars"] = len(hourly)
            self._log_download(symbol, "1h", len(hourly))

        stats["cached"] = False
        return stats

    def download_all(
        self,
        asset_classes: list[str] = None,
        force: bool = False,
    ) -> list[dict]:
        """Download data for all registered assets of given classes."""
        if asset_classes is None:
            asset_classes = ["stock", "crypto"]

        symbols = []
        for cls in asset_classes:
            for sym in get_symbols_by_class(cls):
                asset = ASSET_REGISTRY[sym]
                if asset.data_source == "yahoo":
                    symbols.append(sym)

        # Also add yahoo-sourced commodities and indices
        for sym, asset in ASSET_REGISTRY.items():
            if asset.data_source == "yahoo" and sym not in symbols:
                symbols.append(sym)

        results = []
        for i, symbol in enumerate(symbols):
            log.info(f"Downloading {i+1}/{len(symbols)}: {symbol}")
            result = self.download_and_cache(symbol, force=force)
            results.append(result)
            log.info(f"  {symbol}: {result['daily_bars']} daily, {result['hourly_bars']} hourly"
                     + (" (cached)" if result.get("cached") else ""))

        return results

    def load_cached(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Load cached data from local SQLite."""
        table = "daily_prices" if timeframe == "1d" else "hourly_prices"

        query = f"SELECT datetime, open, high, low, close, volume FROM {table} WHERE symbol = ?"
        params: list = [symbol]

        if start_date:
            query += " AND datetime >= ?"
            params.append(start_date.strftime("%Y-%m-%d %H:%M:%S"))
        if end_date:
            query += " AND datetime <= ?"
            params.append(end_date.strftime("%Y-%m-%d %H:%M:%S"))

        query += " ORDER BY datetime"

        with sqlite3.connect(self.cache_db) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return df

        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        df.index.name = "timestamp"
        return df

    def _save_to_cache(self, symbol: str, df: pd.DataFrame, table: str):
        rows = []
        for dt, row in df.iterrows():
            rows.append((
                symbol,
                dt.strftime("%Y-%m-%d %H:%M:%S"),
                float(row["open"]), float(row["high"]),
                float(row["low"]), float(row["close"]),
                float(row.get("volume", 0)),
            ))
        with sqlite3.connect(self.cache_db) as conn:
            conn.executemany(
                f"INSERT OR REPLACE INTO {table} VALUES (?,?,?,?,?,?,?)",
                rows,
            )

    def _is_recent(self, symbol: str, timeframe: str, hours: int = 24) -> bool:
        with sqlite3.connect(self.cache_db) as conn:
            cursor = conn.execute(
                "SELECT last_download FROM download_log WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe),
            )
            row = cursor.fetchone()
            if not row:
                return False
            last = datetime.fromisoformat(row[0])
            return (datetime.now() - last).total_seconds() < hours * 3600

    def _get_cached_count(self, symbol: str, timeframe: str) -> int:
        table = "daily_prices" if timeframe == "1d" else "hourly_prices"
        with sqlite3.connect(self.cache_db) as conn:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE symbol = ?", (symbol,))
            return cursor.fetchone()[0]

    def _log_download(self, symbol: str, timeframe: str, count: int):
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO download_log VALUES (?,?,?,?)",
                (symbol, timeframe, datetime.now().isoformat(), count),
            )

    def get_stats(self) -> dict:
        with sqlite3.connect(self.cache_db) as conn:
            daily = conn.execute("SELECT COUNT(DISTINCT symbol) FROM daily_prices").fetchone()[0]
            hourly = conn.execute("SELECT COUNT(DISTINCT symbol) FROM hourly_prices").fetchone()[0]
            total_daily = conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
            total_hourly = conn.execute("SELECT COUNT(*) FROM hourly_prices").fetchone()[0]
        return {
            "symbols_with_daily": daily,
            "symbols_with_hourly": hourly,
            "total_daily_bars": total_daily,
            "total_hourly_bars": total_hourly,
        }
