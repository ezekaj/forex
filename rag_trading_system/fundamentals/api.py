"""
Financial Datasets API Client
=============================
Python wrapper for financialdatasets.ai API.
Ported from Dexter's TypeScript implementation.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from urllib.parse import urlencode
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)

BASE_URL = "https://api.financialdatasets.ai"


@dataclass
class APIResponse:
    """API response container."""
    data: Dict[str, Any]
    url: str
    success: bool
    error: Optional[str] = None


class FinancialDataAPI:
    """
    Client for financialdatasets.ai API.

    Provides access to:
    - Financial statements (income, balance sheet, cash flow)
    - SEC filings (10-K, 10-Q, 8-K)
    - Analyst estimates
    - Insider trades
    - Company facts
    - Key ratios
    - Price data
    - News
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the API client.

        Args:
            api_key: API key for financialdatasets.ai.
                    Falls back to FINANCIAL_DATASETS_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("FINANCIAL_DATASETS_API_KEY")
        if not self.api_key:
            logger.warning("No FINANCIAL_DATASETS_API_KEY found. API calls will fail.")

        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": self.api_key or "",
            "Content-Type": "application/json"
        })

    def _call(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """
        Make an API call.

        Args:
            endpoint: API endpoint (e.g., '/financials/income-statements/')
            params: Query parameters

        Returns:
            APIResponse with data and metadata
        """
        url = f"{BASE_URL}{endpoint}"

        # Filter out None values from params
        if params:
            params = {k: v for k, v in params.items() if v is not None}

        try:
            response = self.session.get(url, params=params, timeout=30)
            full_url = response.url

            if not response.ok:
                return APIResponse(
                    data={},
                    url=full_url,
                    success=False,
                    error=f"API error: {response.status_code} {response.reason}"
                )

            data = response.json()
            return APIResponse(data=data, url=full_url, success=True)

        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return APIResponse(
                data={},
                url=url,
                success=False,
                error=str(e)
            )

    # ========== Financial Statements ==========

    def get_income_statements(
        self,
        ticker: str,
        period: str = "annual",
        limit: int = 10,
        report_period_gte: Optional[str] = None,
        report_period_lte: Optional[str] = None
    ) -> APIResponse:
        """
        Get income statements for a company.

        Args:
            ticker: Stock ticker (e.g., 'AAPL')
            period: 'annual', 'quarterly', or 'ttm'
            limit: Max number of periods to return
            report_period_gte: Filter for periods >= this date (YYYY-MM-DD)
            report_period_lte: Filter for periods <= this date (YYYY-MM-DD)
        """
        return self._call("/financials/income-statements/", {
            "ticker": ticker,
            "period": period,
            "limit": limit,
            "report_period_gte": report_period_gte,
            "report_period_lte": report_period_lte
        })

    def get_balance_sheets(
        self,
        ticker: str,
        period: str = "annual",
        limit: int = 10,
        report_period_gte: Optional[str] = None,
        report_period_lte: Optional[str] = None
    ) -> APIResponse:
        """Get balance sheets for a company."""
        return self._call("/financials/balance-sheets/", {
            "ticker": ticker,
            "period": period,
            "limit": limit,
            "report_period_gte": report_period_gte,
            "report_period_lte": report_period_lte
        })

    def get_cash_flow_statements(
        self,
        ticker: str,
        period: str = "annual",
        limit: int = 10,
        report_period_gte: Optional[str] = None,
        report_period_lte: Optional[str] = None
    ) -> APIResponse:
        """Get cash flow statements for a company."""
        return self._call("/financials/cash-flow-statements/", {
            "ticker": ticker,
            "period": period,
            "limit": limit,
            "report_period_gte": report_period_gte,
            "report_period_lte": report_period_lte
        })

    def get_all_financials(
        self,
        ticker: str,
        period: str = "annual",
        limit: int = 10
    ) -> APIResponse:
        """Get all financial statements in one call."""
        return self._call("/financials/", {
            "ticker": ticker,
            "period": period,
            "limit": limit
        })

    # ========== SEC Filings ==========

    def get_filings(
        self,
        ticker: str,
        form_type: Optional[str] = None,
        limit: int = 10
    ) -> APIResponse:
        """
        Get SEC filings for a company.

        Args:
            ticker: Stock ticker
            form_type: Filter by form type ('10-K', '10-Q', '8-K', etc.)
            limit: Max filings to return
        """
        return self._call("/sec/filings/", {
            "ticker": ticker,
            "form_type": form_type,
            "limit": limit
        })

    def get_10k_items(self, ticker: str, limit: int = 5) -> APIResponse:
        """Get 10-K filing items (annual reports)."""
        return self._call("/sec/10k-items/", {
            "ticker": ticker,
            "limit": limit
        })

    def get_10q_items(self, ticker: str, limit: int = 5) -> APIResponse:
        """Get 10-Q filing items (quarterly reports)."""
        return self._call("/sec/10q-items/", {
            "ticker": ticker,
            "limit": limit
        })

    def get_8k_items(self, ticker: str, limit: int = 10) -> APIResponse:
        """Get 8-K filing items (current reports/material events)."""
        return self._call("/sec/8k-items/", {
            "ticker": ticker,
            "limit": limit
        })

    # ========== Analyst & Insider Data ==========

    def get_analyst_estimates(
        self,
        ticker: str,
        period: str = "annual"
    ) -> APIResponse:
        """
        Get analyst estimates (EPS, revenue forecasts).

        Args:
            ticker: Stock ticker
            period: 'annual' or 'quarterly'
        """
        return self._call("/analyst-estimates/", {
            "ticker": ticker,
            "period": period
        })

    def get_insider_trades(
        self,
        ticker: str,
        limit: int = 50
    ) -> APIResponse:
        """Get recent insider trading activity."""
        return self._call("/insider-trades/", {
            "ticker": ticker,
            "limit": limit
        })

    # ========== Company Info ==========

    def get_company_facts(self, ticker: str) -> APIResponse:
        """
        Get company facts (sector, industry, market cap, employees, etc.)
        """
        return self._call("/company/facts/", {"ticker": ticker})

    def get_key_ratios(
        self,
        ticker: str,
        period: str = "annual",
        limit: int = 10
    ) -> APIResponse:
        """
        Get key financial ratios (P/E, ROE, debt/equity, etc.)
        """
        return self._call("/key-ratios/", {
            "ticker": ticker,
            "period": period,
            "limit": limit
        })

    def get_segmented_revenues(
        self,
        ticker: str,
        period: str = "annual",
        limit: int = 5
    ) -> APIResponse:
        """Get revenue breakdown by segment/geography."""
        return self._call("/segmented-revenues/", {
            "ticker": ticker,
            "period": period,
            "limit": limit
        })

    # ========== Prices ==========

    def get_price_snapshot(self, ticker: str) -> APIResponse:
        """Get current price snapshot."""
        return self._call("/prices/snapshot/", {"ticker": ticker})

    def get_prices(
        self,
        ticker: str,
        interval: str = "day",
        interval_multiplier: int = 1,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> APIResponse:
        """
        Get historical price data.

        Args:
            ticker: Stock ticker
            interval: 'minute', 'hour', 'day', 'week', 'month'
            interval_multiplier: Number of intervals per data point
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Max data points
        """
        return self._call("/prices/", {
            "ticker": ticker,
            "interval": interval,
            "interval_multiplier": interval_multiplier,
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit
        })

    # ========== News ==========

    def get_news(
        self,
        ticker: str,
        limit: int = 20,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> APIResponse:
        """Get company news articles."""
        return self._call("/news/", {
            "ticker": ticker,
            "limit": limit,
            "start_date": start_date,
            "end_date": end_date
        })

    # ========== Crypto ==========

    def get_crypto_price(self, ticker: str) -> APIResponse:
        """Get crypto price snapshot (e.g., 'BTC', 'ETH')."""
        return self._call("/crypto/prices/snapshot/", {"ticker": ticker})

    def get_crypto_prices(
        self,
        ticker: str,
        interval: str = "day",
        limit: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> APIResponse:
        """Get historical crypto prices."""
        return self._call("/crypto/prices/", {
            "ticker": ticker,
            "interval": interval,
            "limit": limit,
            "start_date": start_date,
            "end_date": end_date
        })


# Convenience functions for direct use
_default_client: Optional[FinancialDataAPI] = None

def get_client() -> FinancialDataAPI:
    """Get or create the default API client."""
    global _default_client
    if _default_client is None:
        _default_client = FinancialDataAPI()
    return _default_client
