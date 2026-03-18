"""Bull and Bear Researcher Agents — adversarial debate via LLM reasoning."""

import logging
from forex_system.agents.base_agent import BaseAgent, AgentResult

log = logging.getLogger(__name__)


BULL_SYSTEM = """You are a BULLISH analyst. Score each factor from 0 to +2 (how much it supports the BULL case):
1. News sentiment (positive articles, bullish events)
2. News volume (high volume = more attention = potential movement)
3. Event impact (earnings beat=+2, general news=0)
4. Price trend alignment (bullish trend confirms)
5. Technical levels (near support = bounce opportunity)

Give your TOTAL SCORE (0-10) and a concise one-paragraph bullish argument.
Be direct and quantitative. No hedging."""

BEAR_SYSTEM = """You are a BEARISH analyst. Score each factor from 0 to +2 (how much it supports the BEAR case):
1. News sentiment (negative articles, bearish events)
2. News volume (high negative volume = panic selling)
3. Event impact (earnings miss=+2, rate hike=+2)
4. Price trend alignment (bearish trend confirms)
5. Technical levels (near resistance = rejection risk)

Give your TOTAL SCORE (0-10) and a concise one-paragraph bearish argument.
Be direct and quantitative. No hedging."""


class BullResearcher(BaseAgent):
    """Argues the bullish case for a trade. Uses the 235B reasoning model."""

    def __init__(self, llm_endpoint: str = "http://localhost:8000"):
        super().__init__(name="bull_researcher", llm_endpoint=llm_endpoint)

    async def argue(
        self,
        symbol: str,
        news_data: dict,
        tech_data: dict,
        learned_rules: list[dict] = None,
    ) -> AgentResult:
        """Build the bullish argument for this asset."""
        prompt = self._build_prompt(symbol, news_data, tech_data, learned_rules)
        response = await self.llm.ask(prompt, system=BULL_SYSTEM, max_tokens=800)

        return AgentResult(
            agent=self.name,
            asset=symbol,
            timestamp="",
            data={"argument": response, "bias": "bullish"},
        )

    def _build_prompt(self, symbol, news, tech, rules):
        parts = [f"ASSET: {symbol}\n"]

        if news:
            parts.append(f"NEWS: {news.get('article_count', 0)} articles, "
                        f"avg sentiment: {news.get('avg_sentiment', 0):.3f}, "
                        f"signal: {news.get('signal', 'neutral')}")
            articles = news.get("top_articles", [])
            for a in articles[:3]:
                parts.append(f"  - '{a.get('title', '')}' (sentiment: {a.get('sentiment', 0)})")
            events = news.get("event_types", [])
            if events:
                parts.append(f"  Event types: {', '.join(events)}")

        if tech:
            parts.append(f"\nTECHNICAL: trend={tech.get('trend', '?')}, "
                        f"momentum={tech.get('momentum', '?')}, "
                        f"volatility={tech.get('volatility', '?')}")
            ind = tech.get("indicators", {})
            if ind:
                parts.append(f"  RSI={ind.get('rsi', '?')}, MACD hist={ind.get('macd_hist', '?')}, "
                            f"ADX={ind.get('adx', '?')}, Price vs SMA20={ind.get('price_vs_sma20', '?')}%")

        if rules:
            parts.append("\nLEARNED RULES:")
            for r in rules[:5]:
                parts.append(f"  - {r.get('condition', '?')} → {r.get('prediction', '?')} "
                            f"(conf: {r.get('confidence', 0)}, samples: {r.get('samples', 0)})")

        parts.append("\nMake the BULLISH case. Why will this asset go UP in the next 3-5 days?")
        return "\n".join(parts)


class BearResearcher(BaseAgent):
    """Argues the bearish case for a trade. Uses the 235B reasoning model."""

    def __init__(self, llm_endpoint: str = "http://localhost:8000"):
        super().__init__(name="bear_researcher", llm_endpoint=llm_endpoint)

    async def argue(
        self,
        symbol: str,
        news_data: dict,
        tech_data: dict,
        learned_rules: list[dict] = None,
    ) -> AgentResult:
        """Build the bearish argument for this asset."""
        prompt = self._build_prompt(symbol, news_data, tech_data, learned_rules)
        response = await self.llm.ask(prompt, system=BEAR_SYSTEM, max_tokens=800)

        return AgentResult(
            agent=self.name,
            asset=symbol,
            timestamp=news_data.get("timestamp", ""),
            data={"argument": response, "bias": "bearish"},
        )

    def _build_prompt(self, symbol, news, tech, rules):
        parts = [f"ASSET: {symbol}\n"]

        if news:
            parts.append(f"NEWS: {news.get('article_count', 0)} articles, "
                        f"avg sentiment: {news.get('avg_sentiment', 0):.3f}")
            articles = news.get("top_articles", [])
            for a in articles[:3]:
                parts.append(f"  - '{a.get('title', '')}' (sentiment: {a.get('sentiment', 0)})")

        if tech:
            ind = tech.get("indicators", {})
            parts.append(f"\nTECHNICAL: RSI={ind.get('rsi', '?')}, "
                        f"ret_5d={ind.get('ret_5d', '?')}%, "
                        f"BB position={ind.get('bb_position', '?')}")

        if rules:
            parts.append("\nLEARNED RULES:")
            for r in rules[:5]:
                parts.append(f"  - {r.get('condition', '?')} → {r.get('prediction', '?')}")

        parts.append("\nMake the BEARISH case. Why will this asset go DOWN in the next 3-5 days?")
        return "\n".join(parts)
