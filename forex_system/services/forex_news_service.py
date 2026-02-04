#!/usr/bin/env python3
"""
FOREX NEWS SERVICE - Integrated from news_lenci_forex
Provides fundamental analysis layer for forex trading.

Key Sources:
1. Economic Calendar Events (High Impact Only)
2. Central Bank Sentiment (Fed, ECB, BOJ, BOE, etc.)
3. ForexFactory-style calendar integration
4. Geopolitical Event Detection
"""
import asyncio
import aiohttp
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import re


@dataclass
class EconomicEvent:
    """Represents a scheduled economic event."""
    datetime: datetime
    currency: str
    impact: str  # HIGH, MEDIUM, LOW
    event_name: str
    forecast: Optional[float] = None
    previous: Optional[float] = None
    actual: Optional[float] = None

    @property
    def is_high_impact(self) -> bool:
        return self.impact.upper() == 'HIGH'

    @property
    def surprise_factor(self) -> Optional[float]:
        """Calculate surprise vs forecast (actual - forecast) / |forecast|"""
        if self.actual is not None and self.forecast is not None and self.forecast != 0:
            return (self.actual - self.forecast) / abs(self.forecast)
        return None


@dataclass
class CurrencySentiment:
    """Aggregate sentiment for a currency."""
    currency: str
    hawkish_score: float = 0.0  # Central bank hawkishness (-1 to 1)
    economic_strength: float = 0.0  # Recent data vs expectations
    risk_sentiment: float = 0.0  # Risk-on/risk-off impact
    news_volume: int = 0
    key_drivers: List[str] = field(default_factory=list)

    @property
    def composite_score(self) -> float:
        """Weighted composite sentiment score."""
        return (
            self.hawkish_score * 0.40 +
            self.economic_strength * 0.35 +
            self.risk_sentiment * 0.25
        )


class ForexNewsService:
    """
    Integrated news service for forex trading.
    Combines news analysis with economic calendar data.
    """

    # Central bank keywords for hawkish/dovish detection
    HAWKISH_WORDS = [
        'hike', 'raise', 'tighten', 'inflation', 'hawkish', 'restrictive',
        'strong', 'robust', 'accelerate', 'overheat', 'reduce balance sheet'
    ]
    DOVISH_WORDS = [
        'cut', 'lower', 'ease', 'dovish', 'accommodative', 'patient',
        'transitory', 'slow', 'weakness', 'recession', 'support'
    ]

    # Currency to central bank mapping
    CURRENCY_BANKS = {
        'USD': 'Federal Reserve',
        'EUR': 'European Central Bank',
        'GBP': 'Bank of England',
        'JPY': 'Bank of Japan',
        'CHF': 'Swiss National Bank',
        'AUD': 'Reserve Bank of Australia',
        'NZD': 'Reserve Bank of New Zealand',
        'CAD': 'Bank of Canada'
    }

    # High-impact event patterns
    HIGH_IMPACT_EVENTS = {
        'USD': ['Non-Farm Payroll', 'FOMC', 'CPI', 'GDP', 'Retail Sales', 'ISM'],
        'EUR': ['ECB Rate', 'German IFO', 'German ZEW', 'CPI', 'PMI'],
        'GBP': ['BOE Rate', 'CPI', 'GDP', 'Employment', 'PMI'],
        'JPY': ['BOJ Rate', 'CPI', 'GDP', 'Tankan'],
        'AUD': ['RBA Rate', 'Employment', 'CPI', 'GDP'],
        'CAD': ['BOC Rate', 'Employment', 'CPI', 'GDP'],
        'CHF': ['SNB Rate', 'CPI', 'KOF'],
        'NZD': ['RBNZ Rate', 'CPI', 'GDP', 'Employment']
    }

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(__file__).parent.parent.parent / 'data' / 'forex_news.db'
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._session: Optional[aiohttp.ClientSession] = None

    def _init_db(self):
        """Initialize database for news and calendar data."""
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS economic_events (
                id INTEGER PRIMARY KEY,
                datetime TEXT,
                currency TEXT,
                impact TEXT,
                event_name TEXT,
                forecast REAL,
                previous REAL,
                actual REAL,
                UNIQUE(datetime, currency, event_name)
            );

            CREATE TABLE IF NOT EXISTS forex_news (
                id INTEGER PRIMARY KEY,
                datetime TEXT,
                currencies TEXT,
                title TEXT,
                content TEXT,
                sentiment REAL,
                source TEXT,
                UNIQUE(datetime, title)
            );

            CREATE TABLE IF NOT EXISTS central_bank_sentiment (
                id INTEGER PRIMARY KEY,
                currency TEXT,
                date TEXT,
                hawkish_score REAL,
                confidence REAL,
                key_statements TEXT,
                UNIQUE(currency, date)
            );

            CREATE INDEX IF NOT EXISTS idx_events_datetime ON economic_events(datetime);
            CREATE INDEX IF NOT EXISTS idx_news_datetime ON forex_news(datetime);
        """)
        conn.commit()
        conn.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    # =========================================================================
    # ECONOMIC CALENDAR
    # =========================================================================

    async def fetch_economic_calendar(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: Optional[List[str]] = None
    ) -> List[EconomicEvent]:
        """
        Fetch economic calendar events.
        Uses investing.com calendar API (free).
        """
        session = await self._get_session()
        events = []

        # Try investing.com economic calendar
        try:
            url = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"

            # Format dates
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'X-Requested-With': 'XMLHttpRequest',
                'Accept': 'application/json'
            }

            params = {
                'dateFrom': start_str,
                'dateTo': end_str,
                'importance': '3',  # High impact only
            }

            async with session.get(url, headers=headers, params=params, timeout=15) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Parse the calendar data
                    events = self._parse_calendar_data(data, currencies)
        except Exception as e:
            print(f"[ForexNews] Calendar fetch error: {e}")

        return events

    def _parse_calendar_data(
        self,
        data: dict,
        currencies: Optional[List[str]] = None
    ) -> List[EconomicEvent]:
        """Parse calendar response into EconomicEvent objects."""
        events = []

        try:
            for item in data.get('data', []):
                currency = item.get('currency', '').upper()

                # Filter by currency if specified
                if currencies and currency not in currencies:
                    continue

                event = EconomicEvent(
                    datetime=datetime.strptime(item['datetime'], '%Y-%m-%d %H:%M:%S'),
                    currency=currency,
                    impact=item.get('importance', 'LOW'),
                    event_name=item.get('event', ''),
                    forecast=float(item['forecast']) if item.get('forecast') else None,
                    previous=float(item['previous']) if item.get('previous') else None,
                    actual=float(item['actual']) if item.get('actual') else None
                )
                events.append(event)
        except Exception as e:
            print(f"[ForexNews] Parse error: {e}")

        return events

    def get_upcoming_events(
        self,
        pair: str,
        hours_ahead: int = 24
    ) -> List[EconomicEvent]:
        """Get upcoming high-impact events for a currency pair."""
        base = pair[:3].upper()
        quote = pair[3:6].upper()

        now = datetime.utcnow()
        end = now + timedelta(hours=hours_ahead)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        events = conn.execute("""
            SELECT * FROM economic_events
            WHERE datetime BETWEEN ? AND ?
            AND currency IN (?, ?)
            AND impact = 'HIGH'
            ORDER BY datetime
        """, (now.isoformat(), end.isoformat(), base, quote)).fetchall()

        conn.close()

        return [
            EconomicEvent(
                datetime=datetime.fromisoformat(e['datetime']),
                currency=e['currency'],
                impact=e['impact'],
                event_name=e['event_name'],
                forecast=e['forecast'],
                previous=e['previous'],
                actual=e['actual']
            )
            for e in events
        ]

    def should_avoid_trading(
        self,
        pair: str,
        minutes_before: int = 30,
        minutes_after: int = 30
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if trading should be avoided due to upcoming/recent events.
        Returns (should_avoid, reason).
        """
        base = pair[:3].upper()
        quote = pair[3:6].upper()

        now = datetime.utcnow()
        window_start = now - timedelta(minutes=minutes_after)
        window_end = now + timedelta(minutes=minutes_before)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        events = conn.execute("""
            SELECT * FROM economic_events
            WHERE datetime BETWEEN ? AND ?
            AND currency IN (?, ?)
            AND impact = 'HIGH'
            ORDER BY datetime
            LIMIT 1
        """, (window_start.isoformat(), window_end.isoformat(), base, quote)).fetchall()

        conn.close()

        if events:
            event = events[0]
            return True, f"High-impact event: {event['event_name']} ({event['currency']})"

        return False, None

    # =========================================================================
    # CENTRAL BANK SENTIMENT
    # =========================================================================

    async def analyze_central_bank_sentiment(
        self,
        currency: str,
        lookback_days: int = 14
    ) -> CurrencySentiment:
        """
        Analyze central bank sentiment from recent news/statements.
        """
        sentiment = CurrencySentiment(currency=currency)
        bank_name = self.CURRENCY_BANKS.get(currency, currency)

        session = await self._get_session()

        # Fetch recent news about the central bank
        try:
            # Using DuckDuckGo search (free, no API key needed)
            query = f"{bank_name} monetary policy rate decision"
            url = f"https://api.duckduckgo.com/?q={query}&format=json&t=ForexBot"

            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    # Analyze abstracts and related topics
                    texts = []
                    if data.get('Abstract'):
                        texts.append(data['Abstract'])
                    for topic in data.get('RelatedTopics', [])[:5]:
                        if isinstance(topic, dict) and topic.get('Text'):
                            texts.append(topic['Text'])

                    # Calculate hawkish/dovish score
                    hawkish_count = 0
                    dovish_count = 0

                    for text in texts:
                        text_lower = text.lower()
                        hawkish_count += sum(1 for w in self.HAWKISH_WORDS if w in text_lower)
                        dovish_count += sum(1 for w in self.DOVISH_WORDS if w in text_lower)

                    total = hawkish_count + dovish_count
                    if total > 0:
                        sentiment.hawkish_score = (hawkish_count - dovish_count) / total
                        sentiment.key_drivers.append(
                            f"{'Hawkish' if sentiment.hawkish_score > 0 else 'Dovish'} "
                            f"({hawkish_count}H/{dovish_count}D)"
                        )
        except Exception as e:
            print(f"[ForexNews] Central bank sentiment error: {e}")

        return sentiment

    def get_pair_sentiment(
        self,
        pair: str,
        timestamp: Optional[datetime] = None
    ) -> Tuple[float, List[str]]:
        """
        Get sentiment differential for a currency pair.
        Returns (sentiment_score, key_drivers).

        Positive = bullish for the pair (base stronger than quote)
        Negative = bearish for the pair (quote stronger than base)
        """
        base = pair[:3].upper()
        quote = pair[3:6].upper()

        if timestamp is None:
            timestamp = datetime.utcnow()

        date_str = timestamp.strftime('%Y-%m-%d')

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        base_sent = conn.execute("""
            SELECT hawkish_score, confidence, key_statements
            FROM central_bank_sentiment
            WHERE currency = ? AND date <= ?
            ORDER BY date DESC LIMIT 1
        """, (base, date_str)).fetchone()

        quote_sent = conn.execute("""
            SELECT hawkish_score, confidence, key_statements
            FROM central_bank_sentiment
            WHERE currency = ? AND date <= ?
            ORDER BY date DESC LIMIT 1
        """, (quote, date_str)).fetchone()

        conn.close()

        base_score = base_sent['hawkish_score'] if base_sent else 0.0
        quote_score = quote_sent['hawkish_score'] if quote_sent else 0.0

        # Sentiment differential: positive = bullish for pair
        differential = base_score - quote_score

        drivers = []
        if base_sent and base_sent['key_statements']:
            drivers.append(f"{base}: {base_sent['key_statements'][:50]}")
        if quote_sent and quote_sent['key_statements']:
            drivers.append(f"{quote}: {quote_sent['key_statements'][:50]}")

        return differential, drivers

    # =========================================================================
    # NEWS ANALYSIS
    # =========================================================================

    async def fetch_forex_news(
        self,
        pair: str,
        hours_lookback: int = 24
    ) -> List[Dict]:
        """Fetch recent forex news for a pair."""
        base = pair[:3].upper()
        quote = pair[3:6].upper()

        session = await self._get_session()
        news_items = []

        # Try multiple free sources
        sources = [
            f"https://www.google.com/search?q={pair}+forex+news&tbm=nws",
            f"https://api.duckduckgo.com/?q={pair}+forex&format=json"
        ]

        try:
            # DuckDuckGo instant answers
            url = f"https://api.duckduckgo.com/?q={base}{quote}+forex+news&format=json&t=ForexBot"
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    for topic in data.get('RelatedTopics', [])[:10]:
                        if isinstance(topic, dict) and topic.get('Text'):
                            sentiment = self._analyze_news_sentiment(topic['Text'])
                            news_items.append({
                                'title': topic.get('Text', '')[:200],
                                'sentiment': sentiment,
                                'source': 'DuckDuckGo'
                            })
        except Exception as e:
            print(f"[ForexNews] News fetch error: {e}")

        return news_items

    def _analyze_news_sentiment(self, text: str) -> float:
        """Simple sentiment analysis for forex news."""
        text_lower = text.lower()

        # Forex-specific bullish words
        bullish_words = [
            'rise', 'climb', 'surge', 'rally', 'gain', 'strength', 'higher',
            'hawkish', 'hike', 'beat', 'exceed', 'improve', 'grow', 'expand'
        ]

        # Forex-specific bearish words
        bearish_words = [
            'fall', 'drop', 'plunge', 'decline', 'weaken', 'lower', 'dovish',
            'cut', 'miss', 'disappoint', 'contract', 'shrink', 'concern'
        ]

        bullish_count = sum(1 for w in bullish_words if w in text_lower)
        bearish_count = sum(1 for w in bearish_words if w in text_lower)

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0

        return (bullish_count - bearish_count) / total

    # =========================================================================
    # INTEGRATED SIGNAL GENERATION
    # =========================================================================

    def get_fundamental_signal(
        self,
        pair: str,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Generate a fundamental signal for the pair.
        Combines calendar, central bank sentiment, and news.

        Returns:
            {
                'signal': 'BUY' | 'SELL' | 'NEUTRAL',
                'strength': float (0-1),
                'avoid_trading': bool,
                'reasons': List[str]
            }
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        result = {
            'signal': 'NEUTRAL',
            'strength': 0.0,
            'avoid_trading': False,
            'reasons': []
        }

        # Check if we should avoid trading
        should_avoid, avoid_reason = self.should_avoid_trading(pair)
        if should_avoid:
            result['avoid_trading'] = True
            result['reasons'].append(f"AVOID: {avoid_reason}")
            return result

        # Get sentiment differential
        sentiment_diff, drivers = self.get_pair_sentiment(pair, timestamp)

        # Calculate signal based on sentiment
        if abs(sentiment_diff) > 0.2:
            if sentiment_diff > 0:
                result['signal'] = 'BUY'
            else:
                result['signal'] = 'SELL'
            result['strength'] = min(abs(sentiment_diff), 1.0)
            result['reasons'].extend(drivers)

        # Check recent economic data surprises
        base = pair[:3].upper()
        quote = pair[3:6].upper()

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Get recent events with actual data (last 7 days)
        week_ago = (timestamp - timedelta(days=7)).isoformat()
        recent_events = conn.execute("""
            SELECT currency, event_name, actual, forecast, previous
            FROM economic_events
            WHERE datetime > ?
            AND currency IN (?, ?)
            AND actual IS NOT NULL
            AND impact = 'HIGH'
        """, (week_ago, base, quote)).fetchall()

        conn.close()

        # Calculate economic surprise differential
        base_surprises = []
        quote_surprises = []

        for event in recent_events:
            if event['forecast'] is not None and event['forecast'] != 0:
                surprise = (event['actual'] - event['forecast']) / abs(event['forecast'])

                if event['currency'] == base:
                    base_surprises.append(surprise)
                else:
                    quote_surprises.append(surprise)

        if base_surprises or quote_surprises:
            avg_base = sum(base_surprises) / len(base_surprises) if base_surprises else 0
            avg_quote = sum(quote_surprises) / len(quote_surprises) if quote_surprises else 0

            economic_diff = avg_base - avg_quote

            if abs(economic_diff) > 0.05:  # 5% surprise differential
                if economic_diff > 0:
                    if result['signal'] == 'BUY':
                        result['strength'] = min(result['strength'] + 0.2, 1.0)
                    elif result['signal'] == 'NEUTRAL':
                        result['signal'] = 'BUY'
                        result['strength'] = 0.3
                else:
                    if result['signal'] == 'SELL':
                        result['strength'] = min(result['strength'] + 0.2, 1.0)
                    elif result['signal'] == 'NEUTRAL':
                        result['signal'] = 'SELL'
                        result['strength'] = 0.3

                result['reasons'].append(
                    f"Economic surprise: {base} {avg_base:+.1%} vs {quote} {avg_quote:+.1%}"
                )

        return result

    def get_market_context(
        self,
        pair: str,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Get comprehensive market context for LLM review.
        Used by HybridLLMStrategy.
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        fundamental = self.get_fundamental_signal(pair, timestamp)
        sentiment_diff, drivers = self.get_pair_sentiment(pair, timestamp)
        upcoming_events = self.get_upcoming_events(pair, hours_ahead=48)

        return {
            'fundamental_signal': fundamental['signal'],
            'fundamental_strength': fundamental['strength'],
            'sentiment_differential': sentiment_diff,
            'key_drivers': drivers,
            'upcoming_events': [
                {
                    'datetime': e.datetime.isoformat(),
                    'currency': e.currency,
                    'event': e.event_name,
                    'impact': e.impact
                }
                for e in upcoming_events[:5]
            ],
            'avoid_trading': fundamental['avoid_trading'],
            'avoid_reason': fundamental['reasons'][0] if fundamental['avoid_trading'] else None,
            'timestamp': timestamp.isoformat()
        }


# Convenience function for synchronous use
def get_forex_news_service() -> ForexNewsService:
    """Get a ForexNewsService instance."""
    return ForexNewsService()
