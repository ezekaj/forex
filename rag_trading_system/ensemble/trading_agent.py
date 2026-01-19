"""
Trading Agent
=============
The main LLM-powered trading agent that makes decisions.

Uses:
- Qwen for reasoning
- LLaVA for chart analysis
- RAG for news and memory retrieval
- Technical indicators for data
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MIN_CONFIDENCE,
    MIN_ENSEMBLE_AGREEMENT,
    SL_ATR_MULTIPLIER,
    TP_ATR_MULTIPLIER,
    RAG_TOP_K,
    TRADE_MEMORY_K
)

logger = logging.getLogger(__name__)


TRADING_SYSTEM_PROMPT = """You are an expert forex trading analyst. Your job is to analyze market data and make trading decisions.

RULES:
1. You can only see data UP TO the current time - never future data
2. Be conservative - only trade when multiple signals align
3. Always provide clear reasoning for your decisions
4. Consider both technical and fundamental factors
5. Risk management is paramount

OUTPUT FORMAT (JSON):
{
    "action": "BUY" | "SELL" | "HOLD",
    "confidence": 0.0 to 1.0,
    "reasoning": "Your detailed analysis",
    "key_factors": ["factor1", "factor2", ...],
    "risk_assessment": "low" | "medium" | "high"
}
"""

CHART_ANALYSIS_PROMPT = """Analyze this forex chart image and describe what you see.

Focus on:
1. Current trend direction (bullish/bearish/sideways)
2. Key price levels (support/resistance)
3. Chart patterns (if any)
4. Candlestick patterns at current price
5. Overall market structure

Be specific and concise. This analysis will be used for trading decisions.
"""


class TradingAgent:
    """
    LLM-powered trading agent that makes decisions based on:
    - Technical indicators
    - Chart analysis (vision)
    - News sentiment (RAG)
    - Past trade memory
    """

    def __init__(self):
        # Import components
        from ensemble.ollama_client import get_client

        self.llm = get_client()

        # RAG components (lazy load)
        self._embedder = None
        self._news_fetcher = None

        logger.info(f"TradingAgent initialized with model: {self.llm.main_model}")

    @property
    def embedder(self):
        if self._embedder is None:
            from rag.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder

    @property
    def news_fetcher(self):
        if self._news_fetcher is None:
            from preprocessing.news_fetcher import NewsFetcher
            self._news_fetcher = NewsFetcher()
        return self._news_fetcher

    def analyze_chart(self, chart_path: str) -> str:
        """
        Analyze chart image using vision model.

        Args:
            chart_path: Path to chart image

        Returns:
            Analysis text
        """
        if not self.llm.vision_model:
            return "Vision model not available"

        try:
            analysis = self.llm.generate_with_image(
                prompt=CHART_ANALYSIS_PROMPT,
                image_path=chart_path,
                temperature=0.3
            )
            return analysis
        except Exception as e:
            logger.error(f"Chart analysis failed: {e}")
            return f"Chart analysis failed: {e}"

    def get_relevant_news(self, pair: str, current_date: str, top_k: int = None) -> List[Dict]:
        """
        Get relevant news articles using RAG.

        Args:
            pair: Currency pair
            current_date: Current simulated date (YYYY-MM-DD)
            top_k: Number of articles to retrieve

        Returns:
            List of relevant articles
        """
        top_k = top_k or RAG_TOP_K

        # Build query
        base_currency = pair[:3]
        quote_currency = pair[3:]
        query = f"{pair} {base_currency} {quote_currency} forex currency trading"

        try:
            articles = self.embedder.search_relevant_news(
                query=query,
                max_date=current_date,
                top_k=top_k
            )
            return articles
        except Exception as e:
            logger.error(f"News retrieval failed: {e}")
            return []

    def get_similar_trades(self, pair: str, situation: str, current_date: str) -> List[Dict]:
        """
        Get similar past trades from memory.

        Args:
            pair: Currency pair
            situation: Description of current situation
            current_date: Current simulated date

        Returns:
            List of similar past trades
        """
        try:
            trades = self.embedder.search_similar_trades(
                query=f"{pair} {situation}",
                max_date=current_date,
                pair=pair,
                top_k=TRADE_MEMORY_K
            )
            return trades
        except Exception as e:
            logger.error(f"Trade memory retrieval failed: {e}")
            return []

    def make_decision(
        self,
        pair: str,
        current_time: str,
        indicators: Dict,
        chart_analysis: str = None,
        news_context: List[Dict] = None,
        past_trades: List[Dict] = None
    ) -> Dict:
        """
        Make a trading decision.

        Args:
            pair: Currency pair
            current_time: Current simulated time
            indicators: Technical indicators
            chart_analysis: Vision model's chart analysis
            news_context: Relevant news articles
            past_trades: Similar past trades

        Returns:
            Decision dict with action, confidence, reasoning
        """
        # Build context for LLM
        context = self._build_context(
            pair, current_time, indicators, chart_analysis, news_context, past_trades
        )

        # Generate decision
        prompt = f"""
{context}

Based on the above analysis, make a trading decision for {pair}.

Remember:
- Only trade if confidence is high (>60%)
- Consider risk/reward ratio
- Account for current market volatility
- Learn from similar past trades

Respond with valid JSON only:
"""

        try:
            response = self.llm.generate(
                prompt=prompt,
                system=TRADING_SYSTEM_PROMPT,
                temperature=0.2,
                json_mode=True
            )

            # Parse response
            decision = self._parse_decision(response, pair, indicators)
            return decision

        except Exception as e:
            logger.error(f"Decision generation failed: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": f"Error: {e}",
                "key_factors": [],
                "risk_assessment": "high"
            }

    def _build_context(
        self,
        pair: str,
        current_time: str,
        indicators: Dict,
        chart_analysis: str,
        news_context: List[Dict],
        past_trades: List[Dict]
    ) -> str:
        """Build context string for LLM."""

        context_parts = [f"=== TRADING ANALYSIS FOR {pair} ==="]
        context_parts.append(f"Current Time: {current_time}")

        # Technical indicators
        context_parts.append("\n--- TECHNICAL INDICATORS ---")
        if indicators:
            for key, value in indicators.items():
                if not key.endswith("_series") and value is not None:
                    if isinstance(value, float):
                        context_parts.append(f"{key}: {value:.5f}")
                    else:
                        context_parts.append(f"{key}: {value}")

        # Chart analysis
        if chart_analysis:
            context_parts.append("\n--- CHART ANALYSIS (Vision Model) ---")
            context_parts.append(chart_analysis)

        # News context
        if news_context:
            context_parts.append(f"\n--- RELEVANT NEWS ({len(news_context)} articles) ---")
            for i, article in enumerate(news_context[:5], 1):
                context_parts.append(f"{i}. [{article.get('source', 'Unknown')}] {article.get('title', '')[:100]}")
                if article.get('content'):
                    context_parts.append(f"   Summary: {article['content'][:200]}...")

        # Past trades
        if past_trades:
            context_parts.append(f"\n--- SIMILAR PAST TRADES ({len(past_trades)} found) ---")
            for trade in past_trades[:5]:
                context_parts.append(
                    f"- {trade.get('direction', '?')} @ {trade.get('trade_date', '?')}: "
                    f"{trade.get('outcome', '?')} | Lesson: {trade.get('lesson', 'N/A')[:100]}"
                )

        return "\n".join(context_parts)

    def _parse_decision(self, response: str, pair: str, indicators: Dict) -> Dict:
        """Parse LLM response into decision dict."""
        try:
            # Try to parse JSON
            decision = json.loads(response)

            # Validate and normalize
            action = decision.get("action", "HOLD").upper()
            if action not in ["BUY", "SELL", "HOLD"]:
                action = "HOLD"

            confidence = float(decision.get("confidence", 0))
            confidence = max(0, min(1, confidence))

            # Calculate SL/TP levels if trading
            entry_price = indicators.get("current_price", 0)
            atr = indicators.get("atr", entry_price * 0.001)

            if action == "BUY":
                stop_loss = entry_price - (atr * SL_ATR_MULTIPLIER)
                take_profit = entry_price + (atr * TP_ATR_MULTIPLIER)
            elif action == "SELL":
                stop_loss = entry_price + (atr * SL_ATR_MULTIPLIER)
                take_profit = entry_price - (atr * TP_ATR_MULTIPLIER)
            else:
                stop_loss = None
                take_profit = None

            return {
                "action": action,
                "confidence": confidence,
                "reasoning": decision.get("reasoning", "No reasoning provided"),
                "key_factors": decision.get("key_factors", []),
                "risk_assessment": decision.get("risk_assessment", "medium"),
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "pair": pair
            }

        except json.JSONDecodeError:
            # Try to extract from text
            logger.warning("Failed to parse JSON, extracting from text")

            action = "HOLD"
            confidence = 0.3

            response_lower = response.lower()
            if "buy" in response_lower and "don't buy" not in response_lower:
                action = "BUY"
                confidence = 0.5
            elif "sell" in response_lower and "don't sell" not in response_lower:
                action = "SELL"
                confidence = 0.5

            entry_price = indicators.get("current_price", 0)
            atr = indicators.get("atr", entry_price * 0.001)

            return {
                "action": action,
                "confidence": confidence,
                "reasoning": response[:500],
                "key_factors": [],
                "risk_assessment": "high",
                "entry_price": entry_price,
                "stop_loss": entry_price - atr * SL_ATR_MULTIPLIER if action == "BUY" else entry_price + atr * SL_ATR_MULTIPLIER if action == "SELL" else None,
                "take_profit": entry_price + atr * TP_ATR_MULTIPLIER if action == "BUY" else entry_price - atr * TP_ATR_MULTIPLIER if action == "SELL" else None,
                "pair": pair
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test agent
    agent = TradingAgent()

    # Mock indicators
    indicators = {
        "current_price": 1.0850,
        "rsi": 35.5,
        "macd": 0.0002,
        "macd_signal": 0.0001,
        "trend": "bullish",
        "atr": 0.0045,
        "ema_50": 1.0820,
        "support_levels": [1.0800, 1.0750],
        "resistance_levels": [1.0900, 1.0950],
    }

    print("Testing trading decision...")
    decision = agent.make_decision(
        pair="EURUSD",
        current_time="2025-06-15T10:00:00",
        indicators=indicators,
        chart_analysis="Bullish trend with price above EMA50. RSI showing oversold conditions.",
        news_context=[{"title": "ECB signals hawkish stance", "source": "Reuters", "content": "ECB expected to raise rates..."}],
        past_trades=[{"direction": "BUY", "outcome": "WIN", "lesson": "ECB hawkish + oversold RSI = good entry"}]
    )

    print(f"\nDecision: {json.dumps(decision, indent=2)}")
