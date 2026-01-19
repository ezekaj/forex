#!/usr/bin/env python3
"""
RAG Trading System - Main Entry Point
=====================================

This script runs the trading simulation "game" to train the bot.

Usage:
    python run_simulation.py              # Run full simulation
    python run_simulation.py --test       # Run quick test
    python run_simulation.py --fetch-news # Only fetch news
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    SIM_START_DATE,
    SIM_END_DATE,
    SIM_STARTING_CAPITAL,
    PRIMARY_PAIRS,
    DATA_DIR,
    RESULTS_DIR
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "simulation.log")
    ]
)
logger = logging.getLogger(__name__)


def fetch_news():
    """Fetch latest news from RSS feeds."""
    from preprocessing.news_fetcher import NewsFetcher

    logger.info("Fetching news from RSS feeds...")
    fetcher = NewsFetcher()
    new_articles = fetcher.fetch_all_feeds()

    stats = fetcher.get_stats()
    logger.info(f"News database stats: {stats}")

    return new_articles


def test_components():
    """Test all system components."""
    print("\n" + "="*60)
    print("   RAG TRADING SYSTEM - COMPONENT TEST")
    print("="*60)

    # Test Ollama
    print("\n[1/5] Testing Ollama models...")
    try:
        from ensemble.ollama_client import get_client
        client = get_client()
        print(f"  ✓ Models available: {client.available_models}")
        print(f"  ✓ Main model: {client.main_model}")
        print(f"  ✓ Vision model: {client.vision_model}")

        # Quick generation test
        response = client.generate("Say 'OK' if you're working.", max_tokens=10)
        print(f"  ✓ Generation test: {response.strip()}")

        # Embedding test
        emb = client.embed("test")
        print(f"  ✓ Embedding test: {len(emb)} dimensions")
    except Exception as e:
        print(f"  ✗ Ollama error: {e}")
        return False

    # Test News Fetcher
    print("\n[2/5] Testing News Fetcher...")
    try:
        from preprocessing.news_fetcher import NewsFetcher
        fetcher = NewsFetcher()
        stats = fetcher.get_stats()
        print(f"  ✓ News database: {stats['total_articles']} articles")
        if stats['total_articles'] == 0:
            print("  → Fetching news...")
            fetch_news()
            stats = fetcher.get_stats()
            print(f"  ✓ After fetch: {stats['total_articles']} articles")
    except Exception as e:
        print(f"  ✗ News Fetcher error: {e}")

    # Test Chart Generator
    print("\n[3/5] Testing Chart Generator...")
    try:
        from analysis.chart_generator import ChartGenerator, generate_sample_data
        from analysis.indicators import calculate_all_indicators, format_indicators_for_llm

        generator = ChartGenerator()
        df = generate_sample_data("EURUSD", 200)
        chart_path = generator.generate_chart(df, "EURUSD")

        if chart_path:
            print(f"  ✓ Chart generated: {chart_path}")
        else:
            print("  ✗ Chart generation failed")

        indicators = calculate_all_indicators(df)
        print(f"  ✓ Indicators calculated: {len(indicators)} values")
    except Exception as e:
        print(f"  ✗ Chart/Indicators error: {e}")

    # Test Time Machine
    print("\n[4/5] Testing Time Machine...")
    try:
        from simulation.time_machine import TimeMachine

        tm = TimeMachine("2025-01-01", "2025-01-10", interval_hours=4)
        print(f"  ✓ Time Machine: {tm.start_date} → {tm.end_date}")
        print(f"  ✓ Steps remaining: {tm.remaining_steps()}")

        # Test time filtering
        import pandas as pd
        dates = pd.date_range("2025-01-01", periods=100, freq="4h")
        df = pd.DataFrame({"value": range(100)}, index=dates)

        tm.jump_to("2025-01-05")
        filtered = tm.filter_data_by_time(df)
        print(f"  ✓ Time filtering: {len(df)} rows → {len(filtered)} rows (before 2025-01-05)")
    except Exception as e:
        print(f"  ✗ Time Machine error: {e}")

    # Test Trading Agent
    print("\n[5/5] Testing Trading Agent...")
    try:
        from ensemble.trading_agent import TradingAgent

        agent = TradingAgent()
        print(f"  ✓ Agent initialized with model: {agent.llm.main_model}")

        # Quick decision test
        indicators = {
            "current_price": 1.0850,
            "rsi": 35.5,
            "trend": "bullish",
            "atr": 0.0045,
        }

        decision = agent.make_decision(
            pair="EURUSD",
            current_time="2025-06-15T10:00:00",
            indicators=indicators
        )
        print(f"  ✓ Decision: {decision['action']} (conf: {decision['confidence']:.1%})")
    except Exception as e:
        print(f"  ✗ Trading Agent error: {e}")

    print("\n" + "="*60)
    print("   COMPONENT TEST COMPLETE")
    print("="*60 + "\n")

    return True


def run_simulation(
    start_date: str = None,
    end_date: str = None,
    capital: float = None,
    pairs: list = None,
    max_rounds: int = None
):
    """
    Run the main trading simulation.

    Args:
        start_date: Simulation start date
        end_date: Simulation end date
        capital: Starting capital
        pairs: Currency pairs to trade
        max_rounds: Maximum rounds (for testing)
    """
    from simulation.game_engine import GameEngine
    from simulation.time_machine import TimeMachine
    from ensemble.trading_agent import TradingAgent
    from analysis.chart_generator import ChartGenerator, generate_sample_data
    from analysis.indicators import calculate_all_indicators
    from preprocessing.forex_data import ForexDataFetcher

    print("\n" + "="*60)
    print("   RAG TRADING SYSTEM - SIMULATION")
    print("="*60)

    # Initialize components
    start_date = start_date or SIM_START_DATE
    end_date = end_date or SIM_END_DATE
    capital = capital or SIM_STARTING_CAPITAL
    pairs = pairs or PRIMARY_PAIRS

    engine = GameEngine(
        start_date=start_date,
        end_date=end_date,
        starting_capital=capital,
        pairs=pairs
    )

    agent = TradingAgent()
    chart_gen = ChartGenerator()
    data_fetcher = ForexDataFetcher()

    # Check if we have real data
    data_stats = data_fetcher.get_stats()
    use_real_data = len(data_stats) > 0
    if use_real_data:
        print(f"\nUsing REAL historical data from database")
        for pair, info in data_stats.items():
            print(f"  {pair}: {info['count']} bars")
    else:
        print(f"\nUsing SYNTHETIC data (no historical data in database)")

    print(f"\nSimulation: {start_date} → {end_date}")
    print(f"Capital: ${capital:,.2f}")
    print(f"Pairs: {pairs}")
    print(f"Rounds: ~{engine.time_machine.remaining_steps()}")

    # Main simulation loop
    round_num = 0
    for current_time in engine.time_machine.iterate():
        round_num += 1

        if max_rounds and round_num > max_rounds:
            logger.info(f"Reached max rounds ({max_rounds})")
            break

        logger.info(f"\n--- Round {round_num} | {current_time} ---")

        # Process each pair
        for pair in pairs:
            try:
                # Get price data - prefer real data, fallback to synthetic
                if use_real_data and pair in data_stats:
                    df = data_fetcher.get_prices(pair, timeframe="1D")
                    if df is None or len(df) < 50:
                        df = generate_sample_data(pair, 200, end_date=engine.time_machine.now)
                else:
                    df = generate_sample_data(pair, 200, end_date=engine.time_machine.now)

                # Filter to only past data
                df_past = engine.time_machine.filter_data_by_time(df)

                if len(df_past) < 50:
                    continue

                # Get market state
                market_state = engine.get_market_state(pair, df)

                # Calculate indicators
                indicators = calculate_all_indicators(df_past)

                # Generate chart
                chart_path = chart_gen.generate_chart(df_past, pair)

                # Get chart analysis (if vision model available)
                chart_analysis = None
                if chart_path and agent.llm.vision_model:
                    chart_analysis = agent.analyze_chart(chart_path)

                # Get news context
                news_context = agent.get_relevant_news(
                    pair,
                    engine.time_machine.now_date
                )

                # Get similar past trades
                past_trades = agent.get_similar_trades(
                    pair,
                    f"{indicators.get('trend', 'neutral')} trend with RSI {indicators.get('rsi', 50):.0f}",
                    engine.time_machine.now_date
                )

                # Make decision
                decision = agent.make_decision(
                    pair=pair,
                    current_time=engine.time_machine.now_str,
                    indicators=indicators,
                    chart_analysis=chart_analysis,
                    news_context=news_context,
                    past_trades=past_trades
                )

                # Execute decision
                if decision["action"] != "HOLD":
                    trade = engine.make_decision(
                        pair=pair,
                        action=decision["action"],
                        confidence=decision["confidence"],
                        reasoning=decision["reasoning"],
                        entry_price=decision["entry_price"],
                        stop_loss=decision["stop_loss"],
                        take_profit=decision["take_profit"],
                        indicators=indicators,
                        news_context=json.dumps(news_context[:3]) if news_context else None
                    )

                # Check existing positions
                price_data = {pair: df}
                closed_trades = engine.check_positions(price_data)

                # Extract lessons from closed trades
                for trade in closed_trades:
                    lesson = engine.extract_lesson(trade, agent.llm)
                    logger.info(f"Lesson learned: {lesson}")

            except Exception as e:
                logger.error(f"Error processing {pair}: {e}")

        # Print status every 10 rounds
        if round_num % 10 == 0:
            engine.print_status()

    # Final summary
    print("\n" + "="*60)
    print("   SIMULATION COMPLETE")
    print("="*60)
    engine.print_status()

    summary = engine.get_performance_summary()
    print(f"\nFinal Results:")
    print(json.dumps(summary, indent=2, default=str))

    # Save results
    results_file = RESULTS_DIR / f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="RAG Trading System")
    parser.add_argument("--test", action="store_true", help="Run component test")
    parser.add_argument("--fetch-news", action="store_true", help="Fetch news only")
    parser.add_argument("--quick", action="store_true", help="Quick simulation (10 rounds)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, help="Starting capital")

    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║          RAG TRADING SYSTEM v1.0                            ║
║                                                              ║
║     LLM-Powered Forex Trading with Memory                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    if args.fetch_news:
        fetch_news()
    elif args.test:
        test_components()
    elif args.quick:
        run_simulation(max_rounds=10)
    else:
        run_simulation(
            start_date=args.start,
            end_date=args.end,
            capital=args.capital
        )


if __name__ == "__main__":
    main()
