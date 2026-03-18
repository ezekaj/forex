"""Load OHLCV price data from multiple sources (SSD DB + Yahoo Finance cache)."""

import logging
import sqlite3
from datetime import datetime
from typing import Optional

import pandas as pd

from forex_system.training.config import ASSET_REGISTRY, get_asset, TrainingConfig

log = logging.getLogger(__name__)


TIMEFRAME_TABLE_MAP = {
    "1d": "daily_prices",
    "4h": "h4_prices",
    "1h": "hourly_prices",
    "30m": "m30_prices",
    "15m": "m15_prices",
    "5m": "m5_prices",
    "1m": "m1_prices",
}


class PriceLoader:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)

    def get_available_tables(self) -> list[str]:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            return [row[0] for row in cursor.fetchall()]

    def get_symbols(self, timeframe: str = "1d") -> list[str]:
        table = TIMEFRAME_TABLE_MAP.get(timeframe)
        if not table:
            raise ValueError(f"Unknown timeframe: {timeframe}. Available: {list(TIMEFRAME_TABLE_MAP.keys())}")
        with self._connect() as conn:
            cursor = conn.execute(f"SELECT DISTINCT symbol FROM {table} ORDER BY symbol")
            return [row[0] for row in cursor.fetchall()]

    def get_date_range(self, symbol: str, timeframe: str = "1d") -> tuple[datetime, datetime]:
        table = TIMEFRAME_TABLE_MAP.get(timeframe)
        if not table:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        with self._connect() as conn:
            cursor = conn.execute(
                f"SELECT MIN(datetime), MAX(datetime) FROM {table} WHERE symbol = ?",
                (symbol,),
            )
            row = cursor.fetchone()
            return self._parse_datetime(row[0]), self._parse_datetime(row[1])

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        table = TIMEFRAME_TABLE_MAP.get(timeframe)
        if not table:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        query = f"SELECT datetime, open, high, low, close, volume FROM {table} WHERE symbol = ?"
        params: list = [symbol]

        if start_date:
            ts = self._to_db_format(start_date)
            query += " AND datetime >= ?"
            params.append(ts)
        if end_date:
            ts = self._to_db_format(end_date)
            query += " AND datetime <= ?"
            params.append(ts)

        query += " ORDER BY datetime"

        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return df

        df["datetime"] = df["datetime"].apply(self._parse_datetime)
        df = df.set_index("datetime").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        df.index.name = "timestamp"
        return df

    def load_multi_timeframe(
        self,
        symbol: str,
        timeframes: list[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, pd.DataFrame]:
        if timeframes is None:
            timeframes = ["1h", "4h", "1d"]
        return {
            tf: self.load_ohlcv(symbol, tf, start_date, end_date)
            for tf in timeframes
            if TIMEFRAME_TABLE_MAP.get(tf) in self.get_available_tables()
        }

    def get_bar_count(self, symbol: str, timeframe: str = "1d") -> int:
        table = TIMEFRAME_TABLE_MAP.get(timeframe)
        if not table:
            return 0
        with self._connect() as conn:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE symbol = ?", (symbol,))
            return cursor.fetchone()[0]

    def get_stats(self) -> dict:
        tables = self.get_available_tables()
        stats = {"tables": tables, "symbols": {}, "total_bars": {}}
        for tf, table in TIMEFRAME_TABLE_MAP.items():
            if table in tables:
                with self._connect() as conn:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    stats["total_bars"][tf] = cursor.fetchone()[0]
        stats["symbols"] = self.get_symbols("1d")
        return stats

    @staticmethod
    def _parse_datetime(val) -> datetime:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            # Millisecond timestamp
            return datetime.fromtimestamp(val / 1000)
        if isinstance(val, str):
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
                try:
                    return datetime.strptime(val, fmt)
                except ValueError:
                    continue
            # Try as numeric string
            try:
                return datetime.fromtimestamp(int(val) / 1000)
            except (ValueError, OSError):
                pass
        raise ValueError(f"Cannot parse datetime: {val}")

    @staticmethod
    def _to_db_format(dt: datetime) -> str:
        # DB stores millisecond timestamps
        return str(int(dt.timestamp() * 1000))


class UniversalPriceLoader:
    """
    Routes price requests to the correct data source based on asset class.
    - Forex/commodities/indices from SSD → PriceLoader
    - Stocks/crypto from Yahoo Finance → YahooDownloader cache
    """

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self._ssd_loader = PriceLoader(self.config.PRICE_DB_PATH)
        self._yahoo = None

    @property
    def yahoo(self):
        if self._yahoo is None:
            from forex_system.training.data.yahoo_downloader import YahooDownloader
            self._yahoo = YahooDownloader(self.config.YAHOO_CACHE_DB_PATH)
        return self._yahoo

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        auto_download: bool = True,
    ) -> pd.DataFrame:
        """Load OHLCV for any registered asset, routing to correct source."""
        asset = get_asset(symbol)

        if asset.data_source == "ssd_db":
            return self._ssd_loader.load_ohlcv(symbol, timeframe, start_date, end_date)
        else:
            # Yahoo Finance cache
            df = self.yahoo.load_cached(symbol, timeframe, start_date, end_date)
            if df.empty and auto_download:
                log.info(f"No cached data for {symbol}, downloading from Yahoo...")
                self.yahoo.download_and_cache(symbol)
                df = self.yahoo.load_cached(symbol, timeframe, start_date, end_date)
            return df

    def load_multi_timeframe(
        self,
        symbol: str,
        timeframes: list[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, pd.DataFrame]:
        if timeframes is None:
            timeframes = ["1h", "1d"]
        return {
            tf: self.load_ohlcv(symbol, tf, start_date, end_date)
            for tf in timeframes
        }

    def get_available_symbols(self) -> list[str]:
        """Return all symbols that have data available."""
        available = []
        # SSD symbols
        try:
            available.extend(self._ssd_loader.get_symbols("1d"))
        except Exception:
            pass
        # Yahoo cached symbols
        try:
            stats = self.yahoo.get_stats()
            with sqlite3.connect(self.config.YAHOO_CACHE_DB_PATH) as conn:
                cursor = conn.execute("SELECT DISTINCT symbol FROM daily_prices")
                available.extend([row[0] for row in cursor.fetchall()])
        except Exception:
            pass
        return sorted(set(available))
