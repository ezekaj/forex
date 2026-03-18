"""Fund Manager Agent — makes final investment decisions using LLM reasoning."""

import json
import logging
from dataclasses import dataclass
from typing import Optional

from forex_system.agents.base_agent import BaseAgent, AgentResult

log = logging.getLogger(__name__)

FUND_MANAGER_SYSTEM = """You are a quantitative trader. You follow PRICE ACTION first, news second.

CORE RULE: TECHNICALS DECIDE DIRECTION. NEWS DECIDES SIZE.
- If price is ABOVE SMA20 with bullish MACD → default BUY
- If price is BELOW SMA20 with bearish MACD → default SELL
- Positive news + uptrend = HIGH confidence BUY
- Negative news + downtrend = HIGH confidence SELL
- News AGAINST the trend = LOWER confidence, but still follow the trend
- HOLD only when trend is flat AND news is neutral

SCORING (do this step by step):
  Trend (weight 3x): price vs SMA20 → +2 if above and rising, -2 if below and falling
  Momentum: RSI > 60 = +1, RSI < 40 = -1
  MACD: positive histogram = +1, negative = -1
  News: avg sentiment > 0.2 = +1, < -0.2 = -1
  TOTAL = (trend * 3) + momentum + MACD + news

  TOTAL > 0 → BUY
  TOTAL < 0 → SELL
  TOTAL = 0 → HOLD

Confidence = 40 + abs(TOTAL) * 5, capped at 85.

IMPORTANT: Markets go DOWN as often as UP. Do NOT default to BUY. Follow the trend direction shown by the technical indicators.

Respond ONLY in JSON format."""


@dataclass
class Decision:
    symbol: str
    direction: str      # "BUY", "SELL", "HOLD"
    confidence: int     # 0-100
    reasoning: str
    risk_factors: list
    time_horizon_days: int
    bull_strength: int  # 0-100, how convincing was the bull case
    bear_strength: int  # 0-100, how convincing was the bear case


class FundManager(BaseAgent):
    """
    Makes final investment decisions by weighing Bull/Bear arguments
    and quantitative signals. Uses the 235B reasoning model.
    """

    def __init__(self, llm_endpoint: str = "http://localhost:8000"):
        super().__init__(name="fund_manager", llm_endpoint=llm_endpoint)

    async def decide(
        self,
        symbol: str,
        date: str,
        news_data: dict,
        tech_data: dict,
        bull_argument: str,
        bear_argument: str,
        learned_rules: list[dict] = None,
    ) -> Decision:
        """
        Make the final BUY/SELL/HOLD decision for an asset.
        Weighs all evidence and the Bull/Bear debate.
        """
        prompt = self._build_prompt(
            symbol, date, news_data, tech_data,
            bull_argument, bear_argument, learned_rules
        )

        response = await self.llm.ask_json(
            prompt, system=FUND_MANAGER_SYSTEM,
            max_tokens=1200,
        )

        decision = Decision(
            symbol=symbol,
            direction=response.get("direction", "HOLD"),
            confidence=response.get("confidence", 0),
            reasoning=response.get("reasoning", ""),
            risk_factors=response.get("risk_factors", []),
            time_horizon_days=response.get("time_horizon_days", 3),
            bull_strength=response.get("bull_strength", 50),
            bear_strength=response.get("bear_strength", 50),
        )

        self.log_action(
            "decision",
            f"{symbol}: {decision.direction} (conf={decision.confidence})"
        )
        return decision

    def _build_prompt(self, symbol, date, news, tech, bull, bear, rules):
        parts = [
            f"ASSET: {symbol}",
            f"DATE: {date}",
            "",
        ]

        # News signals
        if news:
            parts.append(f"NEWS ANALYSIS ({news.get('article_count', 0)} articles):")
            parts.append(f"  Signal: {news.get('signal', 'neutral')}")
            parts.append(f"  Avg sentiment: {news.get('avg_sentiment', 0):.3f}")
            parts.append(f"  Confidence: {news.get('confidence', 0)}")
            events = news.get("event_types", [])
            if events:
                parts.append(f"  Events detected: {', '.join(events)}")
            parts.append("")

        # Technical signals
        if tech:
            parts.append("TECHNICAL ANALYSIS:")
            parts.append(f"  Direction: {tech.get('direction', '?')}")
            parts.append(f"  Trend: {tech.get('trend', '?')}")
            parts.append(f"  Momentum: {tech.get('momentum', '?')}")
            parts.append(f"  Volatility: {tech.get('volatility', '?')}")
            ind = tech.get("indicators", {})
            if ind:
                parts.append(f"  RSI: {ind.get('rsi', '?')}")
                parts.append(f"  MACD histogram: {ind.get('macd_hist', '?')}")
                parts.append(f"  Price vs SMA20: {ind.get('price_vs_sma20', '?')}%")
                parts.append(f"  5-day return: {ind.get('ret_5d', '?')}%")
                parts.append(f"  ATR: {ind.get('atr_pct', '?')}%")
            parts.append("")

        # Bull/Bear debate
        parts.append("BULL RESEARCHER ARGUMENT:")
        parts.append(bull[:800] if bull else "  (no argument provided)")
        parts.append("")
        parts.append("BEAR RESEARCHER ARGUMENT:")
        parts.append(bear[:800] if bear else "  (no argument provided)")
        parts.append("")

        # Learned rules
        if rules:
            parts.append("LEARNED RULES FROM PAST PREDICTIONS:")
            for r in rules[:5]:
                acc = r.get("correct", 0) / max(r.get("samples", 1), 1) * 100
                parts.append(f"  - {r.get('condition', '?')} → {r.get('prediction', '?')} "
                            f"({acc:.0f}% accuracy, {r.get('samples', 0)} samples)")
            parts.append("")

        parts.append("EXAMPLES:")
        parts.append("""
Example 1 - BUY (uptrend confirmed by news):
  Technicals: price 2% above SMA20, RSI=58, MACD positive
  News: sentiment=0.7, earnings beat
  Scoring: Trend=+2(x3)=+6, Momentum=+1, MACD=+1, News=+1 → TOTAL=+9
  {"direction":"BUY","confidence":85,"reasoning":"Strong uptrend + positive earnings"}

Example 2 - SELL (downtrend, negative news):
  Technicals: price 3% below SMA20, RSI=35, MACD negative
  News: sentiment=-0.3, regulation concerns
  Scoring: Trend=-2(x3)=-6, Momentum=-1, MACD=-1, News=-1 → TOTAL=-9
  {"direction":"SELL","confidence":85,"reasoning":"Clear downtrend + negative news confirms"}

Example 3 - SELL (downtrend, mixed news):
  Technicals: price 1% below SMA20, RSI=45, MACD negative
  News: sentiment=0.3, minor positive
  Scoring: Trend=-1(x3)=-3, Momentum=0, MACD=-1, News=+1 → TOTAL=-3
  {"direction":"SELL","confidence":55,"reasoning":"Downtrend dominates despite slightly positive news"}
""")
        parts.append("Now make YOUR decision. Follow the scoring. Respond ONLY in JSON:")
        parts.append(json.dumps({
            "direction": "BUY or SELL or HOLD",
            "confidence": "0-100",
            "reasoning": "your detailed reasoning",
            "risk_factors": ["what could go wrong"],
            "time_horizon_days": "1-5",
            "bull_strength": "0-100 how convincing was the bull case",
            "bear_strength": "0-100 how convincing was the bear case",
        }, indent=2))

        return "\n".join(parts)
