#!/usr/bin/env python3
"""
LLM TRADING AGENT
=================

An AI-powered trading agent that:
1. ✅ Learns from every trade (memory system)
2. ✅ Reads news in real-time
3. ✅ Analyzes charts visually (multimodal LLM)
4. ✅ Makes decisions with reasoning
5. ✅ Remembers past mistakes and successes

ARCHITECTURE:
┌──────────────────────────────────────────────────────────────────┐
│                        LLM TRADING AGENT                         │
├──────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   NEWS      │  │   CHART     │  │   MEMORY    │              │
│  │   FETCHER   │  │   ANALYZER  │  │   SYSTEM    │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                      │
│         ▼                ▼                ▼                      │
│  ┌────────────────────────────────────────────────────┐         │
│  │              LOCAL LLM (Qwen / LLaMA)              │         │
│  │                                                    │         │
│  │  "Looking at EUR/USD chart, I see a double top     │         │
│  │   pattern forming. News shows ECB hawkish stance.  │         │
│  │   My last 3 EUR/USD shorts were profitable.        │         │
│  │   DECISION: SELL with 65% confidence"             │         │
│  └────────────────────────────────────────────────────┘         │
│                           │                                      │
│                           ▼                                      │
│                    ┌─────────────┐                              │
│                    │   BROKER    │                              │
│                    │  (OANDA)    │                              │
│                    └─────────────┘                              │
└──────────────────────────────────────────────────────────────────┘

LLM OPTIONS (Local, Free):
1. Qwen2-VL (7B) - Multimodal, can see charts
2. LLaMA 3.2 Vision (11B) - Best for reasoning
3. Mistral (7B) - Fast, good for text analysis
4. Phi-3-vision (4B) - Lightweight, chart analysis

REQUIREMENTS:
pip install ollama transformers torch pillow mplfinance
"""

import os
import sys
import json
import time
import asyncio
import sqlite3
import requests
import mplfinance as mpf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from io import BytesIO
import base64
import warnings
warnings.filterwarnings('ignore')

# Optional: Ollama for local LLM
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  Ollama not installed. Run: pip install ollama")


# ═══════════════════════════════════════════════════════════════════════════════
#                       CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentConfig:
    """Configuration for the LLM Trading Agent"""

    # Capital & Risk
    STARTING_CAPITAL: float = 30000
    RISK_PER_TRADE: float = 0.01
    MAX_POSITIONS: int = 2
    MIN_CONFIDENCE: float = 0.50  # Higher threshold for LLM decisions

    # Trading pairs
    PAIRS: List[str] = field(default_factory=lambda: ["EUR_USD", "GBP_USD"])

    # LLM Settings
    LLM_MODEL: str = "llama3.2-vision"  # Options: qwen2-vl, llama3.2-vision, mistral
    LLM_TEMPERATURE: float = 0.3  # Lower = more deterministic
    USE_VISION: bool = True  # Enable chart analysis

    # Data APIs
    TWELVE_DATA_KEY: str = "6e0c3f6868b443ba8d3515a8def07244"

    # OANDA (broker)
    OANDA_ACCOUNT_ID: str = ""
    OANDA_API_KEY: str = ""
    OANDA_ENVIRONMENT: str = "practice"

    # Memory settings
    MEMORY_DB: Path = field(default_factory=lambda: Path(__file__).parent / "agent_memory.db")
    MAX_MEMORY_TRADES: int = 100  # Remember last N trades

    # Paths
    CHART_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "charts")

    def __post_init__(self):
        self.CHART_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#                       MEMORY SYSTEM - LEARN FROM TRADES
# ═══════════════════════════════════════════════════════════════════════════════

class MemorySystem:
    """
    Persistent memory for the trading agent.
    Stores and retrieves:
    - Past trades and their outcomes
    - Market patterns and what worked
    - News events and their market impact
    - Lessons learned
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.conn = sqlite3.connect(config.MEMORY_DB)
        self._init_db()

    def _init_db(self):
        """Initialize database tables"""
        cursor = self.conn.cursor()

        # Trade history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                pair TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                confidence REAL,
                reasoning TEXT,
                news_context TEXT,
                chart_pattern TEXT,
                lesson_learned TEXT
            )
        ''')

        # Market observations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                pair TEXT,
                observation TEXT,
                market_condition TEXT,
                importance INTEGER
            )
        ''')

        # Lessons learned
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lessons (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                category TEXT,
                lesson TEXT,
                success_rate REAL
            )
        ''')

        self.conn.commit()

    def record_trade(self, trade: Dict):
        """Record a completed trade"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO trades
            (timestamp, pair, direction, entry_price, exit_price, pnl,
             confidence, reasoning, news_context, chart_pattern, lesson_learned)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            trade.get('pair'),
            trade.get('direction'),
            trade.get('entry_price'),
            trade.get('exit_price'),
            trade.get('pnl'),
            trade.get('confidence'),
            trade.get('reasoning'),
            trade.get('news_context'),
            trade.get('chart_pattern'),
            trade.get('lesson_learned')
        ))
        self.conn.commit()

    def get_recent_trades(self, pair: str = None, limit: int = 10) -> List[Dict]:
        """Get recent trades for context"""
        cursor = self.conn.cursor()

        if pair:
            cursor.execute('''
                SELECT * FROM trades WHERE pair = ?
                ORDER BY timestamp DESC LIMIT ?
            ''', (pair, limit))
        else:
            cursor.execute('''
                SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?
            ''', (limit,))

        columns = ['id', 'timestamp', 'pair', 'direction', 'entry_price',
                   'exit_price', 'pnl', 'confidence', 'reasoning',
                   'news_context', 'chart_pattern', 'lesson_learned']

        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_win_rate(self, pair: str = None) -> Dict:
        """Calculate win rate statistics"""
        cursor = self.conn.cursor()

        if pair:
            cursor.execute('''
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    AVG(pnl) as avg_pnl,
                    SUM(pnl) as total_pnl
                FROM trades WHERE pair = ?
            ''', (pair,))
        else:
            cursor.execute('''
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    AVG(pnl) as avg_pnl,
                    SUM(pnl) as total_pnl
                FROM trades
            ''')

        row = cursor.fetchone()
        total, wins, avg_pnl, total_pnl = row

        return {
            'total_trades': total or 0,
            'wins': wins or 0,
            'win_rate': (wins / total) if total else 0,
            'avg_pnl': avg_pnl or 0,
            'total_pnl': total_pnl or 0
        }

    def add_lesson(self, category: str, lesson: str, success_rate: float = None):
        """Add a learned lesson"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO lessons (timestamp, category, lesson, success_rate)
            VALUES (?, ?, ?, ?)
        ''', (datetime.now().isoformat(), category, lesson, success_rate))
        self.conn.commit()

    def get_lessons(self, category: str = None) -> List[Dict]:
        """Retrieve learned lessons"""
        cursor = self.conn.cursor()

        if category:
            cursor.execute('''
                SELECT * FROM lessons WHERE category = ?
                ORDER BY timestamp DESC
            ''', (category,))
        else:
            cursor.execute('SELECT * FROM lessons ORDER BY timestamp DESC')

        columns = ['id', 'timestamp', 'category', 'lesson', 'success_rate']
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_memory_context(self, pair: str) -> str:
        """Generate memory context for LLM prompt"""
        # Recent trades
        trades = self.get_recent_trades(pair, 5)
        stats = self.get_win_rate(pair)
        lessons = self.get_lessons()[:5]

        context = f"""
## My Trading Memory for {pair}

### Performance Stats
- Total trades: {stats['total_trades']}
- Win rate: {stats['win_rate']:.1%}
- Total P&L: ${stats['total_pnl']:.2f}

### Recent Trades
"""
        for t in trades:
            result = "✅ WIN" if (t['pnl'] or 0) > 0 else "❌ LOSS"
            context += f"- {t['timestamp'][:10]}: {t['direction']} → {result} (${t['pnl']:.2f})\n"
            if t['lesson_learned']:
                context += f"  Lesson: {t['lesson_learned']}\n"

        context += "\n### Key Lessons Learned\n"
        for lesson in lessons:
            context += f"- [{lesson['category']}] {lesson['lesson']}\n"

        return context


# ═══════════════════════════════════════════════════════════════════════════════
#                       NEWS FETCHER - REAL-TIME NEWS
# ═══════════════════════════════════════════════════════════════════════════════

class NewsFetcher:
    """
    Fetch real-time financial news from multiple sources.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.cache = {}
        self.cache_time = {}

    def get_forex_news(self, pair: str = None) -> List[Dict]:
        """
        Get latest forex news.
        Sources: Google News RSS, Yahoo Finance, FX news sites
        """
        news = []

        # Google News RSS (free, no API key)
        news.extend(self._fetch_google_news(pair))

        # Forex Factory calendar (important events)
        news.extend(self._fetch_economic_calendar())

        return sorted(news, key=lambda x: x.get('time', ''), reverse=True)[:20]

    def _fetch_google_news(self, pair: str = None) -> List[Dict]:
        """Fetch from Google News RSS"""
        try:
            # Build search query
            if pair:
                base_pair = pair.replace('_', ' ')
                query = f"{base_pair} forex"
            else:
                query = "forex currency trading"

            url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return []

            # Parse RSS XML
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)

            news = []
            for item in root.findall('.//item')[:10]:
                title = item.find('title')
                pub_date = item.find('pubDate')
                link = item.find('link')

                if title is not None:
                    news.append({
                        'title': title.text,
                        'time': pub_date.text if pub_date is not None else '',
                        'url': link.text if link is not None else '',
                        'source': 'Google News'
                    })

            return news
        except Exception as e:
            print(f"  ⚠️  Google News error: {e}")
            return []

    def _fetch_economic_calendar(self) -> List[Dict]:
        """Fetch economic calendar events"""
        # This would connect to Forex Factory or similar
        # For now, return placeholder
        return []

    def get_news_summary(self, pair: str) -> str:
        """Generate a news summary for LLM context"""
        news = self.get_forex_news(pair)

        if not news:
            return "No recent news available."

        summary = f"## Latest News for {pair}\n\n"
        for item in news[:5]:
            summary += f"- **{item['title']}**\n"
            summary += f"  Source: {item['source']} | {item['time'][:20] if item['time'] else 'Recent'}\n\n"

        return summary


# ═══════════════════════════════════════════════════════════════════════════════
#                       CHART ANALYZER - VISUAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class ChartAnalyzer:
    """
    Generate and analyze charts using multimodal LLM.
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    def generate_chart(self, df: pd.DataFrame, pair: str) -> str:
        """
        Generate a candlestick chart and save as image.
        Returns path to the image.
        """
        try:
            # Prepare data for mplfinance
            df_plot = df.copy()
            df_plot.index = pd.to_datetime(df_plot.index)

            # Calculate indicators
            df_plot['SMA20'] = df_plot['close'].rolling(20).mean()
            df_plot['SMA50'] = df_plot['close'].rolling(50).mean()

            # Create chart
            chart_path = self.config.CHART_DIR / f"{pair}_chart.png"

            # Add moving averages
            ap = [
                mpf.make_addplot(df_plot['SMA20'], color='blue', width=1),
                mpf.make_addplot(df_plot['SMA50'], color='red', width=1),
            ]

            # Style
            mc = mpf.make_marketcolors(
                up='green', down='red',
                edge='inherit',
                wick='inherit',
                volume='in'
            )
            style = mpf.make_mpf_style(marketcolors=mc, gridstyle='-', gridcolor='lightgray')

            # Plot
            mpf.plot(
                df_plot.tail(100),
                type='candle',
                style=style,
                addplot=ap,
                volume=False,
                title=f'{pair} - 4H Chart',
                savefig=str(chart_path),
                figsize=(12, 8)
            )

            return str(chart_path)

        except Exception as e:
            print(f"  ⚠️  Chart generation error: {e}")
            return None

    def encode_image_base64(self, image_path: str) -> str:
        """Encode image to base64 for LLM"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')


# ═══════════════════════════════════════════════════════════════════════════════
#                       LLM BRAIN - DECISION MAKING
# ═══════════════════════════════════════════════════════════════════════════════

class LLMBrain:
    """
    The LLM-powered decision making brain.
    Uses local LLM (Ollama) for privacy and speed.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.available = OLLAMA_AVAILABLE

        if self.available:
            self._check_model()

    def _check_model(self):
        """Check if required model is available"""
        try:
            models = ollama.list()
            model_names = [m['name'] for m in models.get('models', [])]

            if self.config.LLM_MODEL not in model_names:
                print(f"  ⚠️  Model {self.config.LLM_MODEL} not found.")
                print(f"     Available: {model_names}")
                print(f"     Run: ollama pull {self.config.LLM_MODEL}")
        except Exception as e:
            print(f"  ⚠️  Ollama check error: {e}")

    def analyze(self, context: Dict) -> Dict:
        """
        Analyze market data and make a trading decision.

        Args:
            context: {
                'pair': str,
                'price_data': DataFrame,
                'news': str,
                'memory': str,
                'chart_path': str (optional)
            }

        Returns:
            {
                'decision': 'BUY' | 'SELL' | 'HOLD',
                'confidence': float (0-1),
                'reasoning': str,
                'patterns_seen': List[str],
                'risk_assessment': str
            }
        """
        if not self.available:
            return self._fallback_analysis(context)

        # Build prompt
        prompt = self._build_analysis_prompt(context)

        try:
            # Call LLM
            if self.config.USE_VISION and context.get('chart_path'):
                # Multimodal call with chart image
                response = ollama.chat(
                    model=self.config.LLM_MODEL,
                    messages=[{
                        'role': 'user',
                        'content': prompt,
                        'images': [context['chart_path']]
                    }],
                    options={'temperature': self.config.LLM_TEMPERATURE}
                )
            else:
                # Text-only call
                response = ollama.chat(
                    model=self.config.LLM_MODEL,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': self.config.LLM_TEMPERATURE}
                )

            return self._parse_response(response['message']['content'])

        except Exception as e:
            print(f"  ⚠️  LLM error: {e}")
            return self._fallback_analysis(context)

    def _build_analysis_prompt(self, context: Dict) -> str:
        """Build the analysis prompt for the LLM"""
        pair = context['pair']
        df = context.get('price_data')

        # Price summary
        if df is not None and len(df) > 0:
            current_price = df['close'].iloc[-1]
            high_24h = df['high'].tail(6).max()  # 6 bars = 24h for 4h chart
            low_24h = df['low'].tail(6).min()
            change_24h = ((current_price - df['close'].iloc[-6]) / df['close'].iloc[-6]) * 100

            price_info = f"""
Current Price: {current_price:.5f}
24H High: {high_24h:.5f}
24H Low: {low_24h:.5f}
24H Change: {change_24h:+.2f}%
"""
        else:
            price_info = "Price data unavailable"

        prompt = f"""You are an expert forex trader analyzing {pair}.

## Current Market Data
{price_info}

## Recent News
{context.get('news', 'No news available')}

## Your Trading Memory
{context.get('memory', 'No previous trades')}

## Chart Analysis
{"I've attached the current 4H chart. Please analyze the candlestick patterns, support/resistance levels, and trend direction." if context.get('chart_path') else "No chart available"}

## Your Task
Based on ALL the above information:
1. Identify any chart patterns (double top, head & shoulders, flags, etc.)
2. Assess the news sentiment and its likely market impact
3. Consider your past performance and lessons learned
4. Make a trading decision

## Response Format (REQUIRED)
You MUST respond in this exact JSON format:
```json
{{
    "decision": "BUY" or "SELL" or "HOLD",
    "confidence": 0.0 to 1.0,
    "reasoning": "Explain your decision in 2-3 sentences",
    "patterns_seen": ["pattern1", "pattern2"],
    "risk_assessment": "low" or "medium" or "high",
    "lesson_to_remember": "What should I remember from this analysis?"
}}
```

Be conservative. Only trade when you see clear signals. Remember past mistakes.
"""
        return prompt

    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response into structured decision"""
        try:
            # Find JSON in response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)

            if json_match:
                data = json.loads(json_match.group())
                return {
                    'decision': data.get('decision', 'HOLD').upper(),
                    'confidence': float(data.get('confidence', 0.3)),
                    'reasoning': data.get('reasoning', 'No reasoning provided'),
                    'patterns_seen': data.get('patterns_seen', []),
                    'risk_assessment': data.get('risk_assessment', 'unknown'),
                    'lesson': data.get('lesson_to_remember', '')
                }
        except Exception as e:
            print(f"  ⚠️  Parse error: {e}")

        # Fallback: try to extract decision from text
        response_upper = response.upper()
        if 'STRONG BUY' in response_upper or 'DEFINITELY BUY' in response_upper:
            return {'decision': 'BUY', 'confidence': 0.7, 'reasoning': response[:200]}
        elif 'BUY' in response_upper and 'SELL' not in response_upper:
            return {'decision': 'BUY', 'confidence': 0.5, 'reasoning': response[:200]}
        elif 'STRONG SELL' in response_upper:
            return {'decision': 'SELL', 'confidence': 0.7, 'reasoning': response[:200]}
        elif 'SELL' in response_upper:
            return {'decision': 'SELL', 'confidence': 0.5, 'reasoning': response[:200]}

        return {'decision': 'HOLD', 'confidence': 0.3, 'reasoning': 'Unclear signal'}

    def _fallback_analysis(self, context: Dict) -> Dict:
        """Simple fallback when LLM is unavailable"""
        return {
            'decision': 'HOLD',
            'confidence': 0.0,
            'reasoning': 'LLM unavailable - no trade',
            'patterns_seen': [],
            'risk_assessment': 'unknown'
        }

    def learn_from_trade(self, trade_result: Dict) -> str:
        """
        Ask LLM to analyze a completed trade and extract lessons.
        """
        if not self.available:
            return "LLM unavailable"

        prompt = f"""Analyze this completed trade and extract a lesson:

Trade Details:
- Pair: {trade_result.get('pair')}
- Direction: {trade_result.get('direction')}
- Entry: {trade_result.get('entry_price')}
- Exit: {trade_result.get('exit_price')}
- P&L: ${trade_result.get('pnl', 0):.2f}
- Original Reasoning: {trade_result.get('reasoning', 'N/A')}

What went right or wrong? What should I remember for future trades?
Respond in one concise sentence.
"""

        try:
            response = ollama.chat(
                model=self.config.LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.5}
            )
            return response['message']['content'].strip()
        except:
            return "Unable to analyze trade"


# ═══════════════════════════════════════════════════════════════════════════════
#                       MAIN TRADING AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class LLMTradingAgent:
    """
    The complete LLM-powered trading agent.
    """

    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()

        print("\n" + "="*70)
        print("   LLM TRADING AGENT")
        print("="*70)

        self.memory = MemorySystem(self.config)
        self.news = NewsFetcher(self.config)
        self.charts = ChartAnalyzer(self.config)
        self.brain = LLMBrain(self.config)

        # Data manager (reuse from production system)
        sys.path.insert(0, str(Path(__file__).parent))
        try:
            from PRODUCTION_TRADING_SYSTEM import DataManager, OandaBroker, Config
            prod_config = Config(TWELVE_DATA_KEY=self.config.TWELVE_DATA_KEY)
            self.data_manager = DataManager(prod_config)

            oanda_config = Config(
                OANDA_ACCOUNT_ID=self.config.OANDA_ACCOUNT_ID,
                OANDA_API_KEY=self.config.OANDA_API_KEY,
                OANDA_ENVIRONMENT=self.config.OANDA_ENVIRONMENT
            )
            self.broker = OandaBroker(oanda_config)
        except ImportError:
            print("  ⚠️  Could not import production system components")
            self.data_manager = None
            self.broker = None

        print(f"  LLM Model: {self.config.LLM_MODEL}")
        print(f"  Vision: {'Enabled' if self.config.USE_VISION else 'Disabled'}")
        print(f"  Memory DB: {self.config.MEMORY_DB}")
        print("="*70)

    async def analyze_pair(self, pair: str) -> Dict:
        """
        Full analysis of a currency pair.
        Combines: Chart + News + Memory + LLM
        """
        print(f"\n🔍 Analyzing {pair}...")

        # 1. Get price data
        df = None
        if self.data_manager:
            try:
                df = self.data_manager.fetch_historical(pair, '4h', 200)
                print(f"  📊 Got {len(df)} bars of price data")
            except Exception as e:
                print(f"  ⚠️  Price data error: {e}")

        # 2. Generate chart
        chart_path = None
        if df is not None and self.config.USE_VISION:
            chart_path = self.charts.generate_chart(df, pair)
            if chart_path:
                print(f"  📈 Generated chart: {chart_path}")

        # 3. Get news
        news_summary = self.news.get_news_summary(pair)
        print(f"  📰 Fetched news")

        # 4. Get memory context
        memory_context = self.memory.get_memory_context(pair)
        print(f"  🧠 Retrieved memory")

        # 5. LLM Analysis
        print(f"  🤖 Asking LLM for analysis...")

        context = {
            'pair': pair,
            'price_data': df,
            'news': news_summary,
            'memory': memory_context,
            'chart_path': chart_path
        }

        analysis = self.brain.analyze(context)

        print(f"\n  Decision: {analysis['decision']}")
        print(f"  Confidence: {analysis['confidence']:.0%}")
        print(f"  Reasoning: {analysis['reasoning']}")
        print(f"  Patterns: {analysis.get('patterns_seen', [])}")
        print(f"  Risk: {analysis.get('risk_assessment', 'unknown')}")

        return analysis

    async def run_cycle(self):
        """Run one analysis cycle"""
        print(f"\n{'─'*70}")
        print(f"🔄 Agent Cycle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'─'*70}")

        for pair in self.config.PAIRS:
            analysis = await self.analyze_pair(pair)

            # Check if we should trade
            if (analysis['decision'] != 'HOLD' and
                analysis['confidence'] >= self.config.MIN_CONFIDENCE):

                print(f"\n  ✅ Trade signal: {analysis['decision']} {pair}")

                # Would execute trade here via broker
                # For now, just log it

            else:
                print(f"\n  ⏸️  No trade: confidence too low or HOLD")

    async def run(self, interval_minutes: int = 60):
        """Run the agent continuously"""
        print(f"\n🚀 Starting LLM Trading Agent...")
        print(f"   Pairs: {', '.join(self.config.PAIRS)}")
        print(f"   Interval: {interval_minutes} minutes")

        while True:
            try:
                await self.run_cycle()
                print(f"\n⏰ Next cycle in {interval_minutes} minutes...")
                await asyncio.sleep(interval_minutes * 60)
            except KeyboardInterrupt:
                print("\n🛑 Shutting down...")
                break
            except Exception as e:
                print(f"\n⚠️  Error: {e}")
                await asyncio.sleep(60)


# ═══════════════════════════════════════════════════════════════════════════════
#                       MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    config = AgentConfig(
        PAIRS=["EUR_USD", "GBP_USD"],
        LLM_MODEL="llama3.2-vision",  # or "qwen2-vl", "mistral"
        USE_VISION=True,
        MIN_CONFIDENCE=0.50,
    )

    agent = LLMTradingAgent(config)

    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    LLM TRADING AGENT SETUP                       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Install Ollama:                                              ║
║     curl -fsSL https://ollama.com/install.sh | sh                ║
║                                                                  ║
║  2. Pull a vision-capable model:                                 ║
║     ollama pull llama3.2-vision                                  ║
║     OR                                                           ║
║     ollama pull qwen2-vl                                         ║
║                                                                  ║
║  3. Run the agent:                                               ║
║     python LLM_TRADING_AGENT.py                                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Run single analysis
    asyncio.run(agent.run_cycle())


if __name__ == "__main__":
    main()
