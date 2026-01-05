"""
Demo script to test the sentiment bot without Reddit API.
Uses Yahoo Finance (no API key) to show the system works.
"""
import sys
import time
from pathlib import Path

# Fix Windows console encoding for emojis
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from data.yahoo_client import YahooClient
from trading.paper_trader import PaperTrader
from trading.signal_generator import SignalGenerator
from trading.risk_manager import RiskManager
from trading.position_sizer import PositionSizer
from analysis.sentiment_scorer import SentimentScorer


def demo_yahoo_data():
    """Test Yahoo Finance data fetching."""
    print("\n" + "=" * 60)
    print("YAHOO FINANCE DATA TEST (No API Key Needed)")
    print("=" * 60)

    client = YahooClient()

    # Test with popular tickers
    tickers = ["NVDA", "GME", "TSLA", "AAPL", "SPY"]

    for ticker in tickers:
        try:
            time.sleep(1)  # Rate limiting
            quote = client.get_quote(ticker)
            rsi = client.get_rsi(ticker)
            atr = client.get_atr(ticker)
            vol_ratio = client.get_volume_ratio(ticker)

            print(f"\n{ticker}:")
            print(f"  Price:  ${quote.last:.2f}")
            print(f"  RSI:    {rsi}")
            print(f"  ATR:    ${atr:.2f}" if atr else "  ATR:    N/A")
            print(f"  Volume: {vol_ratio:.1f}x avg" if vol_ratio else "  Volume: N/A")
        except Exception as e:
            print(f"\n{ticker}: Skipped (rate limited)")


def demo_sentiment_scoring():
    """Test sentiment scoring with sample text."""
    print("\n" + "=" * 60)
    print("SENTIMENT SCORING TEST")
    print("=" * 60)

    scorer = SentimentScorer()

    # Sample WSB-style posts (no emojis for Windows compatibility)
    samples = [
        ("NVDA to the moon! Diamond hands, we're not selling! Tendies incoming!", "Bullish WSB"),
        ("GME is dead, sold everything. Total bagholder moment. Going to zero.", "Bearish"),
        ("Just bought some AAPL shares. Solid long-term investment.", "Neutral"),
        ("SHORT SQUEEZE INCOMING ON AMC! Hedgies are REKT! Buy calls NOW!", "Very Bullish"),
        ("Market looking weak, might crash soon. Being cautious.", "Cautious"),
    ]

    for text, label in samples:
        result = scorer.analyze_text(text)
        print(f"\n[{label}]: \"{text[:50]}...\"")
        print(f"  Score: {result.final_score:+.2f} ({result.sentiment_label})")


def demo_signal_generation():
    """Test signal generation with mock data."""
    print("\n" + "=" * 60)
    print("SIGNAL GENERATION TEST")
    print("=" * 60)

    generator = SignalGenerator()

    # Simulate different scenarios
    scenarios = [
        {
            "name": "Strong Buy Signal (High sentiment + volume spike)",
            "ticker": "NVDA",
            "sentiment_score": 0.75,
            "mention_velocity": 50,
            "avg_mention_velocity": 10,
            "unique_authors": 35,
            "has_dd_post": True,
            "total_engagement": 500,
            "rsi": 35,
            "volume_ratio": 2.5,
        },
        {
            "name": "Moderate Signal (Some positive indicators)",
            "ticker": "AAPL",
            "sentiment_score": 0.45,
            "mention_velocity": 20,
            "avg_mention_velocity": 15,
            "unique_authors": 25,
            "has_dd_post": False,
            "total_engagement": 150,
            "rsi": 50,
            "volume_ratio": 1.3,
        },
        {
            "name": "Skip Signal (Low quality)",
            "ticker": "XYZ",
            "sentiment_score": 0.3,
            "mention_velocity": 5,
            "avg_mention_velocity": 5,
            "unique_authors": 8,
            "has_dd_post": False,
            "total_engagement": 20,
            "rsi": 55,
            "volume_ratio": 0.8,
        },
    ]

    for scenario in scenarios:
        name = scenario.pop("name")
        signal = generator.generate_signal(**scenario)
        print(f"\n{name}")
        print(f"  {signal.ticker}: Score {signal.total_score}/100 → {signal.action}")
        print(f"  Reasons: {', '.join(signal.reasons[:3])}")


def demo_paper_trading():
    """Test paper trading simulation."""
    print("\n" + "=" * 60)
    print("PAPER TRADING TEST")
    print("=" * 60)

    # Initialize with $10,000
    trader = PaperTrader(initial_cash=10000)
    client = YahooClient()

    # Set up price getter
    trader.set_price_getter(lambda t: client.get_quote(t).last)

    # Simulate some trades
    print("\nExecuting simulated trades...")

    try:
        # Buy NVDA
        nvda_price = client.get_quote("NVDA").last
        shares = int(1000 / nvda_price)  # ~$1000 position
        trader.buy("NVDA", shares, stop_loss=nvda_price * 0.95)
        print(f"  Bought {shares} NVDA @ ${nvda_price:.2f}")

        # Buy AAPL
        aapl_price = client.get_quote("AAPL").last
        shares = int(1000 / aapl_price)
        trader.buy("AAPL", shares)
        print(f"  Bought {shares} AAPL @ ${aapl_price:.2f}")

        # Get current prices for summary
        prices = {
            "NVDA": client.get_quote("NVDA").last,
            "AAPL": client.get_quote("AAPL").last,
        }

        # Print summary
        trader.print_summary(prices)

    except Exception as e:
        print(f"  Error: {e}")


def demo_risk_manager():
    """Test risk manager circuit breakers."""
    print("\n" + "=" * 60)
    print("RISK MANAGER TEST")
    print("=" * 60)

    rm = RiskManager(initial_equity=10000)

    # Show initial state
    print("\nInitial state:")
    print(f"  Can trade: {rm.can_trade()}")
    print(f"  Risk multiplier: {rm.get_risk_multiplier()}")

    # Simulate some losses
    print("\nSimulating 3 consecutive losses...")
    for i in range(3):
        rm.record_trade_result(pnl=-100, is_win=False)
        rm.update_equity(rm.state.current_equity - 100)

    status = rm.get_status()
    print(f"  Consecutive losses: {status['consecutive_losses']}")
    print(f"  Risk multiplier: {status['risk_multiplier']} (reduced!)")
    print(f"  Effective risk: {status['effective_risk_pct']:.1f}%")

    # Simulate drawdown
    print("\nSimulating 20% drawdown...")
    rm.update_equity(8000)
    status = rm.get_status()
    print(f"  Drawdown: {status['drawdown_pct']:.1f}%")
    print(f"  Is halted: {status['is_halted']}")

    # Simulate halt
    print("\nSimulating 26% drawdown (should halt)...")
    rm.update_equity(7400)
    can_trade, reason = rm.can_trade()
    print(f"  Can trade: {can_trade}")
    print(f"  Reason: {reason}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  STOCK SENTIMENT BOT - DEMO")
    print("  Testing all components (no Reddit API needed)")
    print("=" * 60)

    demo_yahoo_data()
    demo_sentiment_scoring()
    demo_signal_generation()
    demo_risk_manager()
    demo_paper_trading()

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Add your Reddit API keys to .env")
    print("2. Run: python main.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
