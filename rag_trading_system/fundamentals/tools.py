"""
Financial Tools
===============
Convenience functions for fundamental analysis.
Mirrors Dexter's tool structure for easy use.
"""

from typing import Dict, Any, Optional, List
from .api import get_client, APIResponse
import logging

logger = logging.getLogger(__name__)


def _extract_data(response: APIResponse, key: str) -> Dict[str, Any]:
    """Extract data from API response with error handling."""
    if not response.success:
        return {"error": response.error, "data": []}
    return response.data.get(key, response.data)


# ========== Financial Statements ==========

def get_income_statements(
    ticker: str,
    period: str = "annual",
    limit: int = 5
) -> Dict[str, Any]:
    """
    Get income statements for analyzing profitability.

    Returns: revenue, gross_profit, operating_income, net_income, eps, etc.
    """
    response = get_client().get_income_statements(ticker, period, limit)
    return _extract_data(response, "income_statements")


def get_balance_sheets(
    ticker: str,
    period: str = "annual",
    limit: int = 5
) -> Dict[str, Any]:
    """
    Get balance sheets for analyzing financial position.

    Returns: total_assets, total_liabilities, shareholders_equity, cash, debt, etc.
    """
    response = get_client().get_balance_sheets(ticker, period, limit)
    return _extract_data(response, "balance_sheets")


def get_cash_flow_statements(
    ticker: str,
    period: str = "annual",
    limit: int = 5
) -> Dict[str, Any]:
    """
    Get cash flow statements for analyzing liquidity.

    Returns: operating_cash_flow, free_cash_flow, capex, dividends, etc.
    """
    response = get_client().get_cash_flow_statements(ticker, period, limit)
    return _extract_data(response, "cash_flow_statements")


# ========== Analysis Tools ==========

def get_analyst_estimates(
    ticker: str,
    period: str = "annual"
) -> Dict[str, Any]:
    """
    Get Wall Street analyst estimates.

    Returns: estimated_eps, estimated_revenue, number_of_analysts, etc.
    Useful for: Understanding market expectations, identifying potential surprises.
    """
    response = get_client().get_analyst_estimates(ticker, period)
    return _extract_data(response, "analyst_estimates")


def get_insider_trades(
    ticker: str,
    limit: int = 30
) -> Dict[str, Any]:
    """
    Get insider trading activity.

    Returns: transaction_type (buy/sell), shares, value, insider_name, title
    Useful for: Detecting insider sentiment, following smart money.
    """
    response = get_client().get_insider_trades(ticker, limit)
    return _extract_data(response, "insider_trades")


def get_company_facts(ticker: str) -> Dict[str, Any]:
    """
    Get company fundamental facts.

    Returns: sector, industry, market_cap, employees, description, etc.
    Useful for: Quick company overview, sector analysis.
    """
    response = get_client().get_company_facts(ticker)
    return _extract_data(response, "company_facts")


def get_key_ratios(
    ticker: str,
    period: str = "annual",
    limit: int = 5
) -> Dict[str, Any]:
    """
    Get key financial ratios.

    Returns: pe_ratio, price_to_book, roe, roa, debt_to_equity, current_ratio, etc.
    Useful for: Valuation analysis, financial health assessment.
    """
    response = get_client().get_key_ratios(ticker, period, limit)
    return _extract_data(response, "key_ratios")


def get_sec_filings(
    ticker: str,
    form_type: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Get SEC filings.

    Args:
        form_type: '10-K' (annual), '10-Q' (quarterly), '8-K' (material events)

    Returns: filing_date, form_type, url, description
    Useful for: Deep dive into company disclosures, risk factors.
    """
    response = get_client().get_filings(ticker, form_type, limit)
    return _extract_data(response, "filings")


def get_company_news(
    ticker: str,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Get recent company news.

    Returns: headline, summary, source, published_at, url
    Useful for: Sentiment analysis, event-driven trading.
    """
    response = get_client().get_news(ticker, limit)
    return _extract_data(response, "news")


def get_price_snapshot(ticker: str) -> Dict[str, Any]:
    """
    Get current price data.

    Returns: price, change, change_percent, volume, market_cap
    """
    response = get_client().get_price_snapshot(ticker)
    return _extract_data(response, "snapshot")


# ========== Composite Analysis Functions ==========

def get_full_fundamental_analysis(ticker: str) -> Dict[str, Any]:
    """
    Get comprehensive fundamental analysis for a stock.

    Combines:
    - Company facts (sector, industry)
    - Latest financials (income, balance sheet, cash flow)
    - Key ratios (valuation, profitability)
    - Analyst estimates (expectations)
    - Insider activity (smart money signals)
    - Recent news (sentiment)

    Returns a structured dict ready for LLM analysis.
    """
    logger.info(f"Running full fundamental analysis for {ticker}")

    analysis = {
        "ticker": ticker,
        "company": get_company_facts(ticker),
        "financials": {
            "income_statements": get_income_statements(ticker, "annual", 3),
            "balance_sheets": get_balance_sheets(ticker, "annual", 3),
            "cash_flow": get_cash_flow_statements(ticker, "annual", 3),
        },
        "ratios": get_key_ratios(ticker, "annual", 3),
        "analyst_estimates": get_analyst_estimates(ticker),
        "insider_trades": get_insider_trades(ticker, 20),
        "recent_news": get_company_news(ticker, 10),
        "price": get_price_snapshot(ticker),
    }

    return analysis


def get_quick_valuation(ticker: str) -> Dict[str, Any]:
    """
    Get quick valuation metrics for fast decision making.

    Returns key metrics for value assessment:
    - P/E ratio
    - P/B ratio
    - EV/EBITDA
    - Analyst target prices
    - Insider sentiment
    """
    ratios = get_key_ratios(ticker, "annual", 1)
    estimates = get_analyst_estimates(ticker)
    price = get_price_snapshot(ticker)
    insiders = get_insider_trades(ticker, 10)

    # Calculate insider sentiment
    insider_buys = sum(1 for t in insiders if isinstance(t, dict) and t.get("transaction_type") == "buy")
    insider_sells = sum(1 for t in insiders if isinstance(t, dict) and t.get("transaction_type") == "sell")

    return {
        "ticker": ticker,
        "current_price": price.get("price") if isinstance(price, dict) else None,
        "ratios": ratios[0] if isinstance(ratios, list) and ratios else ratios,
        "analyst_consensus": estimates[0] if isinstance(estimates, list) and estimates else estimates,
        "insider_sentiment": {
            "recent_buys": insider_buys,
            "recent_sells": insider_sells,
            "net_sentiment": "bullish" if insider_buys > insider_sells else "bearish" if insider_sells > insider_buys else "neutral"
        }
    }


def get_earnings_analysis(ticker: str) -> Dict[str, Any]:
    """
    Analyze earnings trends and surprises.

    Returns:
    - Historical EPS trend
    - Revenue growth
    - Analyst estimates vs actuals
    - Upcoming earnings expectations
    """
    income = get_income_statements(ticker, "quarterly", 8)
    estimates = get_analyst_estimates(ticker, "quarterly")

    return {
        "ticker": ticker,
        "quarterly_income": income,
        "analyst_estimates": estimates,
    }
