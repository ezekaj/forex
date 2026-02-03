#!/usr/bin/env python3
"""
Test Fundamental Analysis Integration
=====================================
Quick test to verify the Dexter financial tools integration.

Usage:
    export FINANCIAL_DATASETS_API_KEY="your_key_here"
    python test_fundamentals.py
"""

import os
import sys
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_api_connection():
    """Test basic API connection."""
    print("\n" + "="*60)
    print("TEST 1: API Connection")
    print("="*60)

    from fundamentals.api import FinancialDataAPI

    api = FinancialDataAPI()

    if not api.api_key:
        print("❌ No API key found!")
        print("   Set FINANCIAL_DATASETS_API_KEY environment variable")
        return False

    print(f"✅ API key found: {api.api_key[:8]}...")
    return True


def test_price_snapshot():
    """Test fetching price data."""
    print("\n" + "="*60)
    print("TEST 2: Price Snapshot (AAPL)")
    print("="*60)

    from fundamentals.tools import get_price_snapshot

    result = get_price_snapshot("AAPL")
    print(f"Result: {json.dumps(result, indent=2, default=str)[:500]}")

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return False

    print("✅ Price snapshot successful")
    return True


def test_company_facts():
    """Test fetching company facts."""
    print("\n" + "="*60)
    print("TEST 3: Company Facts (MSFT)")
    print("="*60)

    from fundamentals.tools import get_company_facts

    result = get_company_facts("MSFT")
    print(f"Result: {json.dumps(result, indent=2, default=str)[:500]}")

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return False

    print("✅ Company facts successful")
    return True


def test_financial_statements():
    """Test fetching financial statements."""
    print("\n" + "="*60)
    print("TEST 4: Income Statement (GOOGL)")
    print("="*60)

    from fundamentals.tools import get_income_statements

    result = get_income_statements("GOOGL", period="annual", limit=2)
    print(f"Result: {json.dumps(result, indent=2, default=str)[:500]}")

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return False

    print("✅ Financial statements successful")
    return True


def test_analyst_estimates():
    """Test fetching analyst estimates."""
    print("\n" + "="*60)
    print("TEST 5: Analyst Estimates (NVDA)")
    print("="*60)

    from fundamentals.tools import get_analyst_estimates

    result = get_analyst_estimates("NVDA")
    print(f"Result: {json.dumps(result, indent=2, default=str)[:500]}")

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return False

    print("✅ Analyst estimates successful")
    return True


def test_insider_trades():
    """Test fetching insider trades."""
    print("\n" + "="*60)
    print("TEST 6: Insider Trades (TSLA)")
    print("="*60)

    from fundamentals.tools import get_insider_trades

    result = get_insider_trades("TSLA", limit=5)
    print(f"Result: {json.dumps(result, indent=2, default=str)[:500]}")

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return False

    print("✅ Insider trades successful")
    return True


def test_quick_valuation():
    """Test quick valuation analysis."""
    print("\n" + "="*60)
    print("TEST 7: Quick Valuation (AMZN)")
    print("="*60)

    from fundamentals.tools import get_quick_valuation

    result = get_quick_valuation("AMZN")
    print(f"Result: {json.dumps(result, indent=2, default=str)[:1000]}")

    if "error" in str(result):
        print(f"⚠️  Partial data, some errors present")
    else:
        print("✅ Quick valuation successful")

    return True


def test_fundamental_agent():
    """Test the FundamentalAnalysisAgent."""
    print("\n" + "="*60)
    print("TEST 8: Fundamental Analysis Agent (META)")
    print("="*60)

    try:
        from fundamentals.agent import FundamentalAnalysisAgent

        agent = FundamentalAnalysisAgent()

        # Quick analysis without LLM
        result = agent.analyze("META", analysis_type="quick", include_llm_analysis=False)
        print(f"Data-only analysis: {json.dumps(result, indent=2, default=str)[:1000]}")

        # Get trading signal
        signal = agent.get_trading_signal("META")
        print(f"\nTrading signal: {json.dumps(signal, indent=2, default=str)[:500]}")

        print("✅ Fundamental agent working")
        return True

    except Exception as e:
        print(f"❌ Agent error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  FUNDAMENTAL ANALYSIS INTEGRATION TEST")
    print("  Dexter Financial Tools for Forex/Stock Trading")
    print("="*60)

    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    tests = [
        ("API Connection", test_api_connection),
        ("Price Snapshot", test_price_snapshot),
        ("Company Facts", test_company_facts),
        ("Financial Statements", test_financial_statements),
        ("Analyst Estimates", test_analyst_estimates),
        ("Insider Trades", test_insider_trades),
        ("Quick Valuation", test_quick_valuation),
        ("Fundamental Agent", test_fundamental_agent),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} FAILED: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("  TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Fundamental analysis is ready to use.")
        print("\nNext steps:")
        print("1. Get API key from https://financialdatasets.ai")
        print("2. export FINANCIAL_DATASETS_API_KEY='your_key'")
        print("3. Integrate with TradingAgent using:")
        print("   from fundamentals.agent import integrate_with_trading_agent")
        print("   integrate_with_trading_agent(your_trading_agent)")
    else:
        print("\n⚠️  Some tests failed. Check API key and network connection.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
