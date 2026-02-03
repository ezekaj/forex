"""
Fundamental Analysis Agent
==========================
LLM-powered agent for fundamental stock analysis.
Integrates with the trading system to provide fundamental signals.

Based on Dexter's financial research agent architecture.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .tools import (
    get_full_fundamental_analysis,
    get_quick_valuation,
    get_earnings_analysis,
    get_company_facts,
    get_analyst_estimates,
    get_insider_trades,
    get_key_ratios,
    get_income_statements,
)

logger = logging.getLogger(__name__)


FUNDAMENTAL_SYSTEM_PROMPT = """You are an expert fundamental analyst specializing in equity research.
Your job is to analyze company financials and provide actionable investment insights.

ANALYSIS FRAMEWORK:
1. **Valuation**: Is the stock cheap or expensive relative to peers and history?
   - P/E ratio vs industry average
   - P/B ratio for asset-heavy companies
   - EV/EBITDA for comparing across capital structures

2. **Quality**: Is this a high-quality business?
   - ROE > 15% indicates efficient use of equity
   - Consistent revenue and earnings growth
   - Strong free cash flow generation
   - Manageable debt levels (debt/equity < 1)

3. **Growth**: What's the growth trajectory?
   - Revenue growth rate
   - EPS growth rate
   - Analyst estimates vs historical performance

4. **Sentiment**: What does smart money think?
   - Insider buying = bullish signal
   - Insider selling = potential concern (but often routine)
   - Analyst consensus and price targets

5. **Catalysts**: What could move the stock?
   - Upcoming earnings
   - Recent news developments
   - Industry trends

OUTPUT FORMAT (JSON):
{
    "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
    "confidence": 0.0 to 1.0,
    "fair_value_estimate": <price or null>,
    "upside_potential": <percentage or null>,
    "key_strengths": ["strength1", "strength2"],
    "key_risks": ["risk1", "risk2"],
    "thesis": "Your investment thesis in 2-3 sentences",
    "catalysts": ["catalyst1", "catalyst2"]
}
"""


class FundamentalAnalysisAgent:
    """
    Agent for fundamental stock analysis.

    Can be used standalone or integrated with TradingAgent
    for multi-signal decision making.
    """

    def __init__(self, llm_client=None):
        """
        Initialize the agent.

        Args:
            llm_client: LLM client for analysis. If None, will try to use
                       the system's default Ollama client.
        """
        self.llm = llm_client
        self._ensure_llm()

    def _ensure_llm(self):
        """Ensure LLM client is available."""
        if self.llm is None:
            try:
                from ensemble.ollama_client import get_client
                self.llm = get_client()
                logger.info("Using Ollama client for fundamental analysis")
            except ImportError:
                logger.warning("No LLM client available. Analysis will be data-only.")

    def analyze(
        self,
        ticker: str,
        analysis_type: str = "full",
        include_llm_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Perform fundamental analysis on a stock.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            analysis_type: 'full', 'quick', or 'earnings'
            include_llm_analysis: Whether to run LLM analysis on the data

        Returns:
            Analysis results with signal, confidence, and reasoning
        """
        logger.info(f"Analyzing {ticker} ({analysis_type} analysis)")

        # Fetch data based on analysis type
        if analysis_type == "full":
            data = get_full_fundamental_analysis(ticker)
        elif analysis_type == "quick":
            data = get_quick_valuation(ticker)
        elif analysis_type == "earnings":
            data = get_earnings_analysis(ticker)
        else:
            data = get_quick_valuation(ticker)

        # Add timestamp
        data["analysis_timestamp"] = datetime.now().isoformat()
        data["analysis_type"] = analysis_type

        # Run LLM analysis if requested and available
        if include_llm_analysis and self.llm:
            llm_analysis = self._run_llm_analysis(ticker, data)
            data["llm_analysis"] = llm_analysis

        return data

    def _run_llm_analysis(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run LLM analysis on the fundamental data."""
        # Build context for LLM
        context = self._format_data_for_llm(ticker, data)

        prompt = f"""
{context}

Based on the above fundamental data for {ticker}, provide your analysis.

Consider:
1. Is the stock fairly valued, overvalued, or undervalued?
2. What are the key strengths and risks?
3. What's your investment thesis?
4. What catalysts could move the stock?

Respond with valid JSON only:
"""

        try:
            response = self.llm.generate(
                prompt=prompt,
                system=FUNDAMENTAL_SYSTEM_PROMPT,
                temperature=0.2,
                json_mode=True
            )

            return self._parse_llm_response(response)

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": str(e)
            }

    def _format_data_for_llm(self, ticker: str, data: Dict[str, Any]) -> str:
        """Format fundamental data for LLM consumption."""
        parts = [f"=== FUNDAMENTAL ANALYSIS: {ticker} ===\n"]

        # Company overview
        if "company" in data and isinstance(data["company"], dict):
            company = data["company"]
            parts.append("--- COMPANY OVERVIEW ---")
            parts.append(f"Sector: {company.get('sector', 'N/A')}")
            parts.append(f"Industry: {company.get('industry', 'N/A')}")
            parts.append(f"Market Cap: ${company.get('market_cap', 'N/A'):,.0f}" if company.get('market_cap') else "Market Cap: N/A")
            parts.append(f"Employees: {company.get('employees', 'N/A'):,}" if company.get('employees') else "")
            parts.append("")

        # Current price
        if "price" in data and isinstance(data["price"], dict):
            price = data["price"]
            parts.append("--- CURRENT PRICE ---")
            parts.append(f"Price: ${price.get('price', 'N/A')}")
            parts.append(f"Change: {price.get('change_percent', 'N/A')}%")
            parts.append("")

        # Key ratios
        if "ratios" in data:
            ratios = data["ratios"]
            if isinstance(ratios, list) and ratios:
                ratios = ratios[0]
            if isinstance(ratios, dict):
                parts.append("--- KEY RATIOS ---")
                for key in ['pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_to_ebitda', 'roe', 'roa', 'debt_to_equity', 'current_ratio']:
                    if key in ratios and ratios[key] is not None:
                        parts.append(f"{key}: {ratios[key]:.2f}")
                parts.append("")

        # Financials summary
        if "financials" in data:
            fins = data["financials"]
            if "income_statements" in fins:
                income = fins["income_statements"]
                if isinstance(income, list) and income:
                    latest = income[0]
                    parts.append("--- LATEST INCOME STATEMENT ---")
                    for key in ['revenue', 'gross_profit', 'operating_income', 'net_income', 'eps']:
                        if key in latest and latest[key] is not None:
                            if key == 'eps':
                                parts.append(f"{key}: ${latest[key]:.2f}")
                            else:
                                parts.append(f"{key}: ${latest[key]:,.0f}")
                    parts.append("")

        # Analyst estimates
        if "analyst_estimates" in data:
            estimates = data["analyst_estimates"]
            if isinstance(estimates, list) and estimates:
                estimates = estimates[0]
            if isinstance(estimates, dict):
                parts.append("--- ANALYST ESTIMATES ---")
                for key in ['estimated_eps', 'estimated_revenue', 'number_of_analysts']:
                    if key in estimates and estimates[key] is not None:
                        parts.append(f"{key}: {estimates[key]}")
                parts.append("")

        # Insider activity
        if "insider_trades" in data:
            trades = data["insider_trades"]
            if isinstance(trades, list) and trades:
                parts.append(f"--- INSIDER ACTIVITY (Last {len(trades)} trades) ---")
                buys = sum(1 for t in trades if isinstance(t, dict) and t.get('transaction_type') == 'buy')
                sells = sum(1 for t in trades if isinstance(t, dict) and t.get('transaction_type') == 'sell')
                parts.append(f"Buys: {buys}, Sells: {sells}")
                parts.append(f"Net sentiment: {'Bullish' if buys > sells else 'Bearish' if sells > buys else 'Neutral'}")
                parts.append("")

        # Recent news headlines
        if "recent_news" in data:
            news = data["recent_news"]
            if isinstance(news, list) and news:
                parts.append(f"--- RECENT NEWS ({len(news)} articles) ---")
                for article in news[:5]:
                    if isinstance(article, dict):
                        parts.append(f"• {article.get('headline', article.get('title', 'N/A'))[:100]}")
                parts.append("")

        return "\n".join(parts)

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Extract what we can from text
            response_lower = response.lower()

            signal = "NEUTRAL"
            if "bullish" in response_lower or "buy" in response_lower:
                signal = "BULLISH"
            elif "bearish" in response_lower or "sell" in response_lower:
                signal = "BEARISH"

            return {
                "signal": signal,
                "confidence": 0.5,
                "thesis": response[:500],
                "key_strengths": [],
                "key_risks": [],
                "catalysts": []
            }

    def get_trading_signal(
        self,
        ticker: str,
        current_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get a trading signal for integration with TradingAgent.

        Returns a signal compatible with the multi-agent ensemble.

        Args:
            ticker: Stock ticker
            current_price: Current market price (optional, will fetch if not provided)

        Returns:
            {
                "action": "BUY" | "SELL" | "HOLD",
                "confidence": 0.0 to 1.0,
                "reasoning": "...",
                "source": "fundamental_analysis"
            }
        """
        analysis = self.analyze(ticker, analysis_type="quick", include_llm_analysis=True)

        llm = analysis.get("llm_analysis", {})
        signal = llm.get("signal", "NEUTRAL")
        confidence = llm.get("confidence", 0.5)

        # Convert signal to action
        if signal == "BULLISH":
            action = "BUY"
        elif signal == "BEARISH":
            action = "SELL"
        else:
            action = "HOLD"

        # Adjust confidence based on data quality
        if analysis.get("ratios") and analysis.get("analyst_estimates"):
            confidence = min(confidence * 1.1, 1.0)  # Boost if we have good data

        return {
            "action": action,
            "confidence": confidence,
            "reasoning": llm.get("thesis", "Fundamental analysis"),
            "key_factors": llm.get("key_strengths", []) + llm.get("catalysts", []),
            "risks": llm.get("key_risks", []),
            "fair_value": llm.get("fair_value_estimate"),
            "upside_potential": llm.get("upside_potential"),
            "source": "fundamental_analysis",
            "raw_data": analysis
        }


# ========== Integration with TradingAgent ==========

def integrate_with_trading_agent(trading_agent, fundamental_weight: float = 0.25):
    """
    Integrate FundamentalAnalysisAgent with existing TradingAgent.

    This monkey-patches the TradingAgent to include fundamental signals
    in its decision-making process.

    Args:
        trading_agent: Instance of TradingAgent
        fundamental_weight: Weight for fundamental signals (0-1)
    """
    fundamental_agent = FundamentalAnalysisAgent(trading_agent.llm)

    # Store original make_decision method
    original_make_decision = trading_agent.make_decision

    def enhanced_make_decision(
        pair: str,
        current_time: str,
        indicators: Dict,
        chart_analysis: str = None,
        news_context: List[Dict] = None,
        past_trades: List[Dict] = None
    ) -> Dict:
        """Enhanced decision with fundamental analysis."""

        # Check if this is a stock (not forex)
        is_stock = not any(x in pair.upper() for x in ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD'])

        # Get original decision
        decision = original_make_decision(
            pair, current_time, indicators, chart_analysis, news_context, past_trades
        )

        # Add fundamental analysis for stocks
        if is_stock:
            try:
                fundamental_signal = fundamental_agent.get_trading_signal(
                    pair,
                    current_price=indicators.get("current_price")
                )

                # Blend signals
                tech_confidence = decision.get("confidence", 0.5)
                fund_confidence = fundamental_signal.get("confidence", 0.5)

                # Check for signal agreement
                tech_action = decision.get("action", "HOLD")
                fund_action = fundamental_signal.get("action", "HOLD")

                if tech_action == fund_action:
                    # Signals agree - boost confidence
                    blended_confidence = min(
                        tech_confidence * (1 - fundamental_weight) +
                        fund_confidence * fundamental_weight + 0.1,
                        1.0
                    )
                else:
                    # Signals disagree - reduce confidence
                    blended_confidence = max(
                        tech_confidence * (1 - fundamental_weight) +
                        fund_confidence * fundamental_weight - 0.1,
                        0.0
                    )

                # Update decision
                decision["confidence"] = blended_confidence
                decision["fundamental_signal"] = fundamental_signal
                decision["reasoning"] = (
                    f"Technical: {decision.get('reasoning', 'N/A')}\n"
                    f"Fundamental: {fundamental_signal.get('reasoning', 'N/A')}"
                )

                logger.info(
                    f"{pair}: Tech={tech_action} ({tech_confidence:.0%}), "
                    f"Fund={fund_action} ({fund_confidence:.0%}), "
                    f"Blended={blended_confidence:.0%}"
                )

            except Exception as e:
                logger.error(f"Fundamental analysis failed for {pair}: {e}")
                decision["fundamental_error"] = str(e)

        return decision

    # Replace method
    trading_agent.make_decision = enhanced_make_decision
    trading_agent._fundamental_agent = fundamental_agent

    logger.info(f"Integrated fundamental analysis with weight={fundamental_weight}")
    return trading_agent


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test standalone analysis
    agent = FundamentalAnalysisAgent()

    print("Testing AAPL analysis...")
    result = agent.analyze("AAPL", analysis_type="quick")
    print(json.dumps(result, indent=2, default=str))

    print("\nGetting trading signal...")
    signal = agent.get_trading_signal("AAPL")
    print(json.dumps(signal, indent=2, default=str))
