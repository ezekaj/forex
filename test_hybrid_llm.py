"""
Test script for Hybrid ML+LLM Strategy.

Compares:
1. Baseline: Random Forest only
2. Hybrid: RF + News Sentiment + LLM Review

Goal: Determine if LLM review improves win rate enough to justify costs.
Target: 58%+ win rate to break even (after 1.5 pip transaction costs)

Inspired by AI-Trader architecture, adapted for forex.
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from forex_system.services.data_service import DataService
from forex_system.services.feature_engineering import FeatureEngineer
from forex_system.services.news_service import NewsService
from forex_system.services.llm_service import LLMService
from forex_system.strategies.random_forest import RandomForestStrategy
from forex_system.strategies.hybrid_llm import HybridLLMStrategy
from forex_system.backtesting.engine import BacktestEngine
from forex_system.backtesting.metrics import BacktestMetrics


def print_section(title: str):
    """Print section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print('=' * 80)


def print_comparison(baseline_metrics: dict, hybrid_metrics: dict, hybrid_stats: dict):
    """Print side-by-side comparison."""
    print_section("PERFORMANCE COMPARISON: Baseline vs Hybrid")

    print("\n📊 WIN RATE (Most Important)")
    print(f"{'Metric':<30} {'Baseline':>15} {'Hybrid':>15} {'Improvement':>15}")
    print('-' * 80)

    baseline_wr = baseline_metrics['win_rate']
    hybrid_wr = hybrid_metrics['win_rate']
    wr_improvement = hybrid_wr - baseline_wr

    print(f"{'Win Rate':<30} {baseline_wr:>14.1%} {hybrid_wr:>14.1%} {wr_improvement:>+14.1%}")
    print(f"{'Break-Even Threshold':<30} {'58.0%':>15} {'58.0%':>15} {'(target)':>15}")

    status = "✅ SUCCESS" if hybrid_wr >= 0.58 else "❌ FAILED"
    print(f"\n{status}: Hybrid win rate = {hybrid_wr:.1%}")

    print("\n💰 RETURNS")
    print(f"{'Metric':<30} {'Baseline':>15} {'Hybrid':>15} {'Improvement':>15}")
    print('-' * 80)

    print(f"{'Total Return':<30} {baseline_metrics['total_return']:>14.2%} "
          f"{hybrid_metrics['total_return']:>14.2%} "
          f"{hybrid_metrics['total_return'] - baseline_metrics['total_return']:>+14.2%}")

    print(f"{'Sharpe Ratio':<30} {baseline_metrics['sharpe_ratio']:>15.3f} "
          f"{hybrid_metrics['sharpe_ratio']:>15.3f} "
          f"{hybrid_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio']:>+15.3f}")

    print(f"{'Max Drawdown':<30} {baseline_metrics['max_drawdown']:>14.2%} "
          f"{hybrid_metrics['max_drawdown']:>14.2%} "
          f"{hybrid_metrics['max_drawdown'] - baseline_metrics['max_drawdown']:>+14.2%}")

    print("\n📈 TRADING ACTIVITY")
    print(f"{'Metric':<30} {'Baseline':>15} {'Hybrid':>15} {'Difference':>15}")
    print('-' * 80)

    print(f"{'Total Trades':<30} {baseline_metrics['total_trades']:>15.0f} "
          f"{hybrid_metrics['total_trades']:>15.0f} "
          f"{int(hybrid_metrics['total_trades'] - baseline_metrics['total_trades']):>+15.0f}")

    print(f"{'Winning Trades':<30} {baseline_metrics['winning_trades']:>15.0f} "
          f"{hybrid_metrics['winning_trades']:>15.0f} "
          f"{int(hybrid_metrics['winning_trades'] - baseline_metrics['winning_trades']):>+15.0f}")

    print(f"{'Losing Trades':<30} {baseline_metrics['losing_trades']:>15.0f} "
          f"{hybrid_metrics['losing_trades']:>15.0f} "
          f"{int(hybrid_metrics['losing_trades'] - baseline_metrics['losing_trades']):>+15.0f}")

    print("\n🤖 LLM REVIEW STATISTICS")
    print(f"{'Metric':<30} {'Value':>15}")
    print('-' * 80)

    print(f"{'Total ML Signals':<30} {hybrid_stats['total_signals']:>15.0f}")
    print(f"{'Approved by LLM':<30} {hybrid_stats['approved']:>15.0f} "
          f"({hybrid_stats['approval_rate']:.1%})")
    print(f"{'Rejected by LLM':<30} {hybrid_stats['rejected']:>15.0f} "
          f"({hybrid_stats['rejection_rate']:.1%})")
    print(f"{'Modified by LLM':<30} {hybrid_stats['modified']:>15.0f} "
          f"({hybrid_stats['modification_rate']:.1%})")

    print(f"\n{'Total LLM Cost':<30} ${hybrid_stats['total_cost_usd']:>14.2f}")
    print(f"{'Avg Cost per Signal':<30} ${hybrid_stats['avg_cost_per_signal']:>14.4f}")

    # Calculate cost efficiency
    trades_improved = hybrid_metrics['total_trades']
    if trades_improved > 0:
        profit_per_trade_baseline = baseline_metrics['total_return'] / baseline_metrics['total_trades']
        profit_per_trade_hybrid = hybrid_metrics['total_return'] / hybrid_metrics['total_trades']
        profit_improvement = (profit_per_trade_hybrid - profit_per_trade_baseline) * 10000  # In dollars on $10K account

        roi = (profit_improvement - hybrid_stats['avg_cost_per_signal']) / hybrid_stats['avg_cost_per_signal'] * 100

        print(f"\n💎 COST-BENEFIT ANALYSIS")
        print(f"{'Metric':<30} {'Value':>15}")
        print('-' * 80)
        print(f"{'Profit/Trade (Baseline)':<30} ${profit_per_trade_baseline * 10000:>14.2f}")
        print(f"{'Profit/Trade (Hybrid)':<30} ${profit_per_trade_hybrid * 10000:>14.2f}")
        print(f"{'Improvement per Trade':<30} ${profit_improvement:>14.2f}")
        print(f"{'LLM Cost per Trade':<30} ${hybrid_stats['avg_cost_per_signal']:>14.2f}")
        print(f"{'Net Benefit per Trade':<30} ${profit_improvement - hybrid_stats['avg_cost_per_signal']:>14.2f}")
        print(f"{'ROI on LLM Investment':<30} {roi:>14.1f}%")


def run_test():
    """Run comprehensive test of hybrid system."""

    print_section("HYBRID ML+LLM STRATEGY TEST")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Pair: EURUSD")
    print(f"Timeframe: 1h")
    print(f"Period: 2024 full year")

    # Configuration
    pair = 'EURUSD'
    timeframe = '1h'
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    initial_capital = 10000

    print("\n⚠️  NOTE: This test requires API keys to be set in .env file")
    print("   - JINA_API_KEY for news sentiment")
    print("   - ANTHROPIC_API_KEY or OPENAI_API_KEY for LLM review")
    print("\n   Without API keys, the test will use mock data (for demonstration)")

    # Step 1: Load data
    print_section("STEP 1: Load Historical Data")
    data_service = DataService()
    df = data_service.get_historical_data(pair, timeframe, start_date, end_date)
    print(f"✓ Loaded {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Step 2: Generate features
    print_section("STEP 2: Generate Technical Features")
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.generate_features(df)
    print(f"✓ Generated {len(features_df.columns)} technical indicators")
    print(f"  Sample features: {', '.join(features_df.columns[:10].tolist())}...")

    # Step 3: Generate labels
    print_section("STEP 3: Generate Training Labels")
    rf_strategy_temp = RandomForestStrategy()
    labels = rf_strategy_temp.generate_labels(
        df,
        lookahead_bars=5,
        buy_threshold=0.005,
        sell_threshold=-0.005,
        binary=True  # BUY vs SELL only (no HOLD)
    )

    # Remove NaN labels
    valid_mask = ~labels.isna()
    features_clean = features_df[valid_mask]
    labels_clean = labels[valid_mask]

    buy_count = (labels_clean == 1).sum()
    sell_count = (labels_clean == -1).sum()
    print(f"✓ Generated {len(labels_clean)} labels")
    print(f"  BUY: {buy_count} ({buy_count/len(labels_clean):.1%})")
    print(f"  SELL: {sell_count} ({sell_count/len(labels_clean):.1%})")

    # Step 4: Train baseline (RF only)
    print_section("STEP 4: Train Baseline Strategy (Random Forest)")
    baseline_strategy = RandomForestStrategy(
        n_estimators=200,
        max_depth=20,
        random_state=42
    )
    baseline_metrics_train = baseline_strategy.train(features_clean, labels_clean)
    print(f"✓ Training complete")
    print(f"  Train accuracy: {baseline_metrics_train['train_accuracy']:.3f}")
    print(f"  Val accuracy: {baseline_metrics_train['val_accuracy']:.3f}")
    print(f"  Val win rate: {baseline_metrics_train['val_win_rate']:.3f}")

    # Step 5: Train hybrid (RF + News + LLM)
    print_section("STEP 5: Train Hybrid Strategy (RF + News + LLM)")

    # Check if API keys are available
    news_service = NewsService()
    llm_service = None
    enable_llm = False

    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')

    if anthropic_key or openai_key:
        enable_llm = True
        provider = 'anthropic' if anthropic_key else 'openai'
        llm_service = LLMService(provider=provider)
        print(f"✓ LLM service initialized ({provider})")
    else:
        print("⚠️  No LLM API keys found - LLM review will be DISABLED")
        print("   Hybrid will only add news sentiment features (no LLM review)")

    hybrid_strategy = HybridLLMStrategy(
        base_strategy=RandomForestStrategy(n_estimators=200, max_depth=20, random_state=42),
        news_service=news_service,
        llm_service=llm_service,
        enable_llm_review=enable_llm,
        min_ml_confidence=0.55,
        min_llm_confidence=0.60,
        pair=pair
    )

    print("⏳ Training hybrid strategy (may take a while due to news API calls)...")
    hybrid_metrics_train = hybrid_strategy.train(features_clean, labels_clean)
    print(f"✓ Training complete")
    print(f"  Train accuracy: {hybrid_metrics_train['train_accuracy']:.3f}")
    print(f"  Val accuracy: {hybrid_metrics_train['val_accuracy']:.3f}")
    print(f"  Val win rate: {hybrid_metrics_train['val_win_rate']:.3f}")

    # Step 6: Backtest baseline
    print_section("STEP 6: Backtest Baseline Strategy")
    baseline_backtest = BacktestEngine(
        strategy=baseline_strategy,
        initial_capital=initial_capital,
        spread_pips=1.0,
        slippage_pips=0.5
    )

    print("⏳ Running backtest...")
    baseline_trades, baseline_equity = baseline_backtest.run(features_clean, df[valid_mask])
    baseline_metrics_backtest = BacktestMetrics(baseline_trades, baseline_equity, initial_capital)
    baseline_summary = baseline_metrics_backtest.get_summary()

    print(f"✓ Backtest complete")
    print(f"  Total Return: {baseline_summary['total_return']:.2%}")
    print(f"  Win Rate: {baseline_summary['win_rate']:.1%}")
    print(f"  Total Trades: {baseline_summary['total_trades']}")

    # Step 7: Backtest hybrid
    print_section("STEP 7: Backtest Hybrid Strategy")
    hybrid_backtest = BacktestEngine(
        strategy=hybrid_strategy,
        initial_capital=initial_capital,
        spread_pips=1.0,
        slippage_pips=0.5
    )

    print("⏳ Running backtest (LLM review active - this will take time)...")
    hybrid_trades, hybrid_equity = hybrid_backtest.run(features_clean, df[valid_mask])
    hybrid_metrics_backtest = BacktestMetrics(hybrid_trades, hybrid_equity, initial_capital)
    hybrid_summary = hybrid_metrics_backtest.get_summary()

    print(f"✓ Backtest complete")
    print(f"  Total Return: {hybrid_summary['total_return']:.2%}")
    print(f"  Win Rate: {hybrid_summary['win_rate']:.1%}")
    print(f"  Total Trades: {hybrid_summary['total_trades']}")

    # Step 8: Get LLM review stats
    hybrid_stats = hybrid_strategy.get_review_stats()

    # Step 9: Compare results
    print_comparison(baseline_summary, hybrid_summary, hybrid_stats)

    # Step 10: Final verdict
    print_section("FINAL VERDICT: GO/NO-GO DECISION")

    hybrid_wr = hybrid_summary['win_rate']
    hybrid_return = hybrid_summary['total_return']
    baseline_return = baseline_summary['total_return']

    print(f"\n🎯 TARGET: 58% win rate to break even after transaction costs")
    print(f"📊 RESULT: {hybrid_wr:.1%} win rate achieved")

    if hybrid_wr >= 0.58 and hybrid_return > baseline_return:
        print(f"\n✅ GO: Hybrid system shows promise!")
        print(f"   - Win rate above break-even threshold ({hybrid_wr:.1%} >= 58%)")
        print(f"   - Outperforms baseline ({hybrid_return:.2%} vs {baseline_return:.2%})")
        print(f"   - LLM costs justified by improved performance")
        print(f"\n   NEXT STEPS:")
        print(f"   1. Run paper trading for 30 days")
        print(f"   2. Monitor LLM review decisions closely")
        print(f"   3. Optimize LLM confidence thresholds")
        print(f"   4. Test on 4h timeframe (less noise)")

    elif hybrid_wr >= 0.58 and hybrid_return <= baseline_return:
        print(f"\n⚠️  MAYBE: Win rate good but returns poor")
        print(f"   - Win rate above break-even ({hybrid_wr:.1%} >= 58%)")
        print(f"   - But returns not better than baseline ({hybrid_return:.2%} vs {baseline_return:.2%})")
        print(f"   - LLM is filtering bad trades but may be too conservative")
        print(f"\n   NEXT STEPS:")
        print(f"   1. Lower min_llm_confidence from 0.60 to 0.50")
        print(f"   2. Analyze rejected signals (save_review_log)")
        print(f"   3. Retrain with different features")

    elif hybrid_wr < 0.58 and hybrid_return > baseline_return:
        print(f"\n⚠️  MIXED: Returns improved but win rate below target")
        print(f"   - Win rate below break-even ({hybrid_wr:.1%} < 58%)")
        print(f"   - But returns better than baseline ({hybrid_return:.2%} vs {baseline_return:.2%})")
        print(f"   - May work if risk/reward is good (bigger wins, smaller losses)")
        print(f"\n   NEXT STEPS:")
        print(f"   1. Check profit factor and risk/reward ratio")
        print(f"   2. Optimize position sizing")
        print(f"   3. Test on longer timeframes (4h, daily)")

    else:
        print(f"\n❌ NO-GO: Hybrid system failed to improve performance")
        print(f"   - Win rate below break-even ({hybrid_wr:.1%} < 58%)")
        print(f"   - Returns not better than baseline ({hybrid_return:.2%} vs {baseline_return:.2%})")
        print(f"   - LLM costs not justified")
        print(f"\n   RECOMMENDATION: PIVOT TO STOCKS")
        print(f"   - Forex 1h is too noisy for this approach")
        print(f"   - AI-Trader proved stocks work (+16% on NASDAQ)")
        print(f"   - Apply your infrastructure to S&P 500 stocks")
        print(f"   - Use AI-Trader's architecture as blueprint")

    # Save review log
    if enable_llm:
        log_path = project_root / 'hybrid_review_log.json'
        hybrid_strategy.save_review_log(str(log_path))
        print(f"\n📝 LLM review log saved to: {log_path}")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    try:
        run_test()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
