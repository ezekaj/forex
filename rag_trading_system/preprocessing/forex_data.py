"""
Forex Historical Data Fetcher
=============================
Downloads and manages historical forex price data.

Sources:
- yfinance (free, decent quality)
- OANDA API (production quality, requires account)
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Try to import yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.warning("yfinance not installed. Run: pip install yfinance")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, PRIMARY_PAIRS


class ForexDataFetcher:
    """
    Fetches and caches historical forex data.

    Uses yfinance for free data, with OANDA as optional production source.
    """

    # Yahoo Finance forex symbols
    YAHOO_SYMBOLS = {
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "USDJPY=X",
        "AUDUSD": "AUDUSD=X",
        "USDCAD": "USDCAD=X",
        "USDCHF": "USDCHF=X",
        "NZDUSD": "NZDUSD=X",
        "EURGBP": "EURGBP=X",
        "EURJPY": "EURJPY=X",
        "GBPJPY": "GBPJPY=X",
    }

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DATA_DIR / "forex_prices.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the price database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL,
                timeframe TEXT DEFAULT '4H',
                source TEXT DEFAULT 'yfinance',
                UNIQUE(pair, timestamp, timeframe)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_pair ON prices(pair)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_timestamp ON prices(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_pair_time ON prices(pair, timestamp)")

        conn.commit()
        conn.close()

    def fetch_from_yfinance(
        self,
        pair: str,
        start_date: str,
        end_date: str = None,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch forex data from Yahoo Finance.

        Args:
            pair: Currency pair (e.g., "EURUSD")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            interval: Data interval (1h, 4h, 1d)

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not HAS_YFINANCE:
            logger.error("yfinance not available")
            return None

        symbol = self.YAHOO_SYMBOLS.get(pair.upper())
        if not symbol:
            logger.error(f"Unknown pair: {pair}")
            return None

        end_date = end_date or datetime.now().strftime("%Y-%m-%d")

        try:
            logger.info(f"Fetching {pair} from Yahoo Finance: {start_date} to {end_date} ({interval})")

            ticker = yf.Ticker(symbol)

            # For intraday data, Yahoo limits history
            # 1h data: max 730 days
            # 1d data: no practical limit
            yf_interval = interval
            if interval == "4h":
                # Try 1h first, fall back to 1d
                yf_interval = "1h"

            # Fetch data
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=yf_interval,
                auto_adjust=True
            )

            # If 1h failed, try daily
            if df.empty and yf_interval == "1h":
                logger.info(f"1h data unavailable, trying daily for {pair}")
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    auto_adjust=True
                )
                yf_interval = "1d"

            if df.empty:
                logger.warning(f"No data returned for {pair}")
                return None

            # Normalize column names
            df.columns = [c.lower() for c in df.columns]

            # Keep only OHLCV
            cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
            df = df[cols].copy()

            # Add volume if missing
            if 'volume' not in df.columns:
                df['volume'] = 0

            # Resample to 4H if we got 1h data
            if interval == "4h" and yf_interval == "1h":
                df = self._resample_to_4h(df)

            # Remove any NaN rows
            df = df.dropna()

            logger.info(f"Fetched {len(df)} bars for {pair} ({yf_interval})")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch {pair}: {e}")
            return None

    def _resample_to_4h(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample hourly data to 4-hour bars."""
        resampled = df.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        return resampled.dropna()

    def save_to_db(self, pair: str, df: pd.DataFrame, timeframe: str = "4H", source: str = "yfinance"):
        """Save price data to database."""
        if df is None or df.empty:
            return 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        count = 0
        for timestamp, row in df.iterrows():
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO prices
                    (pair, timestamp, open, high, low, close, volume, timeframe, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pair.upper(),
                    timestamp.isoformat(),
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    float(row['volume']) if pd.notna(row['volume']) else 0,
                    timeframe,
                    source
                ))
                count += 1
            except Exception as e:
                logger.warning(f"Failed to insert row: {e}")

        conn.commit()
        conn.close()
        logger.info(f"Saved {count} bars for {pair}")
        return count

    def get_prices(
        self,
        pair: str,
        start_date: str = None,
        end_date: str = None,
        timeframe: str = "4H"
    ) -> Optional[pd.DataFrame]:
        """
        Get price data from database.

        Args:
            pair: Currency pair
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            timeframe: Timeframe filter

        Returns:
            DataFrame with OHLCV data
        """
        conn = sqlite3.connect(self.db_path)

        query = "SELECT timestamp, open, high, low, close, volume FROM prices WHERE pair = ? AND timeframe = ?"
        params = [pair.upper(), timeframe]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            return None

        # Convert timestamp to datetime index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        return df

    def get_stats(self) -> Dict:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT pair, COUNT(*), MIN(timestamp), MAX(timestamp) FROM prices GROUP BY pair")
        rows = cursor.fetchall()

        conn.close()

        stats = {}
        for pair, count, min_date, max_date in rows:
            stats[pair] = {
                "count": count,
                "start": min_date,
                "end": max_date
            }
        return stats

    def fetch_and_save_all(
        self,
        pairs: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        interval: str = "1d"
    ) -> Dict:
        """
        Fetch and save data for multiple pairs.

        Args:
            pairs: List of pairs (defaults to PRIMARY_PAIRS)
            start_date: Start date (defaults to 2 years ago)
            end_date: End date (defaults to today)
            interval: Data interval (1d, 4h, 1h)

        Returns:
            Dict with results per pair
        """
        pairs = pairs or PRIMARY_PAIRS

        if not start_date:
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        results = {}

        for pair in pairs:
            logger.info(f"Processing {pair}...")

            # Fetch from yfinance
            df = self.fetch_from_yfinance(pair, start_date, end_date, interval=interval)

            if df is not None and not df.empty:
                # Determine timeframe label
                tf_label = "1D" if interval == "1d" else "4H" if interval == "4h" else "1H"
                count = self.save_to_db(pair, df, tf_label, "yfinance")
                results[pair] = {"status": "success", "bars": count, "timeframe": tf_label}
            else:
                results[pair] = {"status": "failed", "bars": 0}

        return results


def download_historical_data(
    pairs: List[str] = None,
    years: int = 2
) -> Dict:
    """
    Convenience function to download historical data.

    Args:
        pairs: Currency pairs to download
        years: Number of years of history

    Returns:
        Results dict
    """
    fetcher = ForexDataFetcher()

    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

    logger.info(f"Downloading {years} years of data for {pairs or PRIMARY_PAIRS}")

    results = fetcher.fetch_and_save_all(pairs, start_date)

    # Print summary
    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)

    for pair, result in results.items():
        status = "✓" if result["status"] == "success" else "✗"
        print(f"  {status} {pair}: {result['bars']} bars")

    stats = fetcher.get_stats()
    print("\nDatabase contents:")
    for pair, info in stats.items():
        print(f"  {pair}: {info['count']} bars ({info['start'][:10]} to {info['end'][:10]})")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Download 2 years of data for primary pairs
    results = download_historical_data(years=2)
