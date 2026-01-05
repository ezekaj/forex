"""
Yahoo Finance client for market data.
No API key required - just works.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from dataclasses import dataclass

import yfinance as yf
import pandas as pd

logger = logging.getLogger("sentiment_bot.yahoo")


@dataclass
class Quote:
    """Current price quote."""
    ticker: str
    bid: float
    ask: float
    last: float
    timestamp: datetime

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2 if self.bid and self.ask else self.last

    @property
    def spread(self) -> float:
        return (self.ask - self.bid) if self.bid and self.ask else 0


@dataclass
class Bar:
    """OHLCV bar data."""
    ticker: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class StockInfo:
    """Basic stock information."""
    ticker: str
    name: str
    sector: Optional[str]
    market_cap: Optional[float]
    avg_volume: Optional[int]
    fifty_two_week_high: Optional[float]
    fifty_two_week_low: Optional[float]


class YahooClient:
    """
    Yahoo Finance client for market data.

    Advantages:
    - No API key required
    - Unlimited historical data
    - Reliable and well-maintained

    Limitations:
    - 15-minute delay on quotes (fine for swing trading)
    - No order execution (use paper trading simulation)
    """

    def __init__(self):
        self._cache: Dict[str, yf.Ticker] = {}
        logger.info("Yahoo Finance client initialized (no API key needed)")

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """Get or create cached ticker object."""
        if symbol not in self._cache:
            self._cache[symbol] = yf.Ticker(symbol)
        return self._cache[symbol]

    def get_quote(self, ticker: str) -> Quote:
        """Get latest quote for ticker."""
        t = self._get_ticker(ticker)
        info = t.info

        # Get current price from fast_info (more reliable)
        try:
            fast = t.fast_info
            last_price = fast.last_price
        except:
            last_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

        return Quote(
            ticker=ticker,
            bid=info.get('bid', last_price),
            ask=info.get('ask', last_price),
            last=last_price,
            timestamp=datetime.now()
        )

    def get_bars(
        self,
        ticker: str,
        period: str = "3mo",
        interval: str = "1d"
    ) -> List[Bar]:
        """
        Get historical bars.

        Args:
            ticker: Stock symbol
            period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        """
        t = self._get_ticker(ticker)
        df = t.history(period=period, interval=interval)

        bars = []
        for idx, row in df.iterrows():
            bars.append(Bar(
                ticker=ticker,
                timestamp=idx.to_pydatetime(),
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=int(row['Volume'])
            ))

        return bars

    def get_bars_range(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str = "1d"
    ) -> List[Bar]:
        """Get bars for specific date range."""
        t = self._get_ticker(ticker)
        df = t.history(start=start, end=end, interval=interval)

        bars = []
        for idx, row in df.iterrows():
            bars.append(Bar(
                ticker=ticker,
                timestamp=idx.to_pydatetime(),
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=int(row['Volume'])
            ))

        return bars

    def get_latest_bar(self, ticker: str) -> Optional[Bar]:
        """Get most recent daily bar."""
        bars = self.get_bars(ticker, period="5d", interval="1d")
        return bars[-1] if bars else None

    def get_info(self, ticker: str) -> StockInfo:
        """Get stock information."""
        t = self._get_ticker(ticker)
        info = t.info

        return StockInfo(
            ticker=ticker,
            name=info.get('longName', ticker),
            sector=info.get('sector'),
            market_cap=info.get('marketCap'),
            avg_volume=info.get('averageVolume'),
            fifty_two_week_high=info.get('fiftyTwoWeekHigh'),
            fifty_two_week_low=info.get('fiftyTwoWeekLow')
        )

    def get_rsi(self, ticker: str, period: int = 14) -> Optional[float]:
        """Calculate RSI for ticker."""
        bars = self.get_bars(ticker, period="3mo", interval="1d")
        if len(bars) < period + 1:
            return None

        closes = [b.close for b in bars]
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]

        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return round(rsi, 2)

    def get_atr(self, ticker: str, period: int = 14) -> Optional[float]:
        """Calculate Average True Range."""
        bars = self.get_bars(ticker, period="3mo", interval="1d")
        if len(bars) < period + 1:
            return None

        true_ranges = []
        for i in range(1, len(bars)):
            high = bars[i].high
            low = bars[i].low
            prev_close = bars[i-1].close

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        atr = sum(true_ranges[-period:]) / period
        return round(atr, 2)

    def get_volume_ratio(self, ticker: str) -> Optional[float]:
        """Get today's volume vs average volume."""
        t = self._get_ticker(ticker)
        info = t.info

        avg_volume = info.get('averageVolume', 0)
        current_volume = info.get('volume', 0)

        if avg_volume == 0:
            return None

        return round(current_volume / avg_volume, 2)

    def is_market_open(self) -> bool:
        """Check if US market is currently open (approximate)."""
        now = datetime.now()
        # Simple check - weekday and between 9:30 AM and 4 PM ET
        # Note: Doesn't account for holidays
        if now.weekday() >= 5:  # Weekend
            return False

        # Rough market hours (adjust for your timezone)
        hour = now.hour
        return 9 <= hour < 16

    def validate_ticker(self, ticker: str) -> bool:
        """Check if ticker is valid."""
        try:
            t = self._get_ticker(ticker)
            info = t.info
            return 'symbol' in info or 'shortName' in info
        except:
            return False

    def get_multiple_quotes(self, tickers: List[str]) -> Dict[str, Quote]:
        """Get quotes for multiple tickers efficiently."""
        quotes = {}
        for ticker in tickers:
            try:
                quotes[ticker] = self.get_quote(ticker)
            except Exception as e:
                logger.warning(f"Failed to get quote for {ticker}: {e}")
        return quotes
