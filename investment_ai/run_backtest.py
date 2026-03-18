"""
CLI entry point for running the walk-forward backtest.

Usage:
    python -m investment_ai.run_backtest BTC-USD
    python -m investment_ai.run_backtest BTC-USD AAPL NVDA
    python -m investment_ai.run_backtest  # defaults to BTC-USD AAPL NVDA
"""

import sys
from investment_ai.backtest import run_backtest, BacktestConfig


def main():
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["BTC-USD", "AAPL", "NVDA"]

    config = BacktestConfig()
    results = {}

    for symbol in symbols:
        result = run_backtest(symbol, config)
        results[symbol] = result

    if len(results) > 1:
        print("\n" + "=" * 70)
        print("CROSS-ASSET SUMMARY")
        print("=" * 70)
        print(f"{'Symbol':<10} {'Trades':>6} {'WR':>6} {'PF':>6} "
              f"{'Net%':>8} {'B&H%':>8} {'Sharpe':>7} {'DD%':>6} {'Beats':>6}")
        print("-" * 70)
        for sym, r in results.items():
            if r.get("error"):
                print(f"{sym:<10} ERROR: {r['error']}")
                continue
            beats = "YES" if r["beats_buy_hold"] else "no"
            print(
                f"{sym:<10} {r['total_trades']:>6} {r['win_rate']:>5.0%} "
                f"{r['profit_factor']:>6.2f} {r['net_return']:>+7.1f}% "
                f"{r['buy_hold_return']:>+7.1f}% {r['sharpe']:>7.2f} "
                f"{r['max_drawdown']:>5.1f}% {beats:>6}"
            )


if __name__ == "__main__":
    main()
