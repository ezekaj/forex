"""
Fundamental Analysis Module
===========================
Python port of Dexter's financial research tools.
Uses financialdatasets.ai API for comprehensive fundamental analysis.

Features:
- Income statements, balance sheets, cash flow
- SEC filings (10-K, 10-Q, 8-K)
- Analyst estimates and consensus
- Insider trades
- Company facts and key ratios
- News sentiment

Adapted from: github.com/virattt/dexter
"""

from .api import FinancialDataAPI
from .tools import (
    get_income_statements,
    get_balance_sheets,
    get_cash_flow_statements,
    get_analyst_estimates,
    get_insider_trades,
    get_company_facts,
    get_key_ratios,
    get_sec_filings,
    get_company_news,
    get_price_snapshot,
)
from .agent import FundamentalAnalysisAgent

__all__ = [
    'FinancialDataAPI',
    'FundamentalAnalysisAgent',
    'get_income_statements',
    'get_balance_sheets',
    'get_cash_flow_statements',
    'get_analyst_estimates',
    'get_insider_trades',
    'get_company_facts',
    'get_key_ratios',
    'get_sec_filings',
    'get_company_news',
    'get_price_snapshot',
]
