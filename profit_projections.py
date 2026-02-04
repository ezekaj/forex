#!/usr/bin/env python3
"""
PROFIT PROJECTIONS CALCULATOR
Realistic P&L expectations for $30,000 capital

Based on:
- Conservative risk management (1% per trade)
- Integrated strategy win rate estimates (55-62%)
- Average risk:reward of 1:1.5 to 1:2
- 15-30 trades per month (quality over quantity)
"""

import random
from dataclasses import dataclass
from typing import List, Tuple
import statistics

# ============================================================================
# CONFIGURATION
# ============================================================================

INITIAL_CAPITAL = 30_000  # $30,000 starting capital

# Strategy parameters (Conservative)
RISK_PER_TRADE_PCT = 0.01  # 1% risk per trade
MAX_RISK_PER_TRADE_PCT = 0.015  # 1.5% max for high confidence
MIN_RISK_PER_TRADE_PCT = 0.005  # 0.5% for low confidence

# Expected performance range
WIN_RATE_LOW = 0.55  # Conservative estimate
WIN_RATE_MID = 0.58  # Expected with all filters
WIN_RATE_HIGH = 0.62  # Optimistic scenario

# Risk:Reward ratios
RR_RATIO_LOW = 1.3  # Conservative (SL 20 pips, TP 26 pips)
RR_RATIO_MID = 1.5  # Expected (SL 20 pips, TP 30 pips)
RR_RATIO_HIGH = 2.0  # When trend is strong

# Trade frequency (integrated system trades less but better)
TRADES_PER_WEEK_LOW = 3
TRADES_PER_WEEK_MID = 5
TRADES_PER_WEEK_HIGH = 7

# Trading costs
SPREAD_COST_PER_TRADE = 1.5  # pips
AVERAGE_STOP_LOSS_PIPS = 20  # Average stop loss in pips


@dataclass
class TradeResult:
    """Single trade result."""
    is_win: bool
    risk_pct: float
    reward_pct: float
    pnl_pct: float
    pnl_usd: float


@dataclass
class SimulationResult:
    """Simulation period result."""
    period: str
    starting_capital: float
    ending_capital: float
    total_pnl: float
    total_pnl_pct: float
    num_trades: int
    wins: int
    losses: int
    win_rate: float
    max_drawdown_pct: float
    best_trade: float
    worst_trade: float


def simulate_trades(
    capital: float,
    num_trades: int,
    win_rate: float,
    rr_ratio: float,
    risk_per_trade: float = RISK_PER_TRADE_PCT
) -> Tuple[float, List[TradeResult], float]:
    """
    Simulate a series of trades.

    Returns: (final_capital, trade_results, max_drawdown_pct)
    """
    current_capital = capital
    peak_capital = capital
    max_drawdown = 0.0
    trades = []

    for _ in range(num_trades):
        # Determine if win or loss
        is_win = random.random() < win_rate

        # Calculate risk amount
        risk_amount = current_capital * risk_per_trade

        # Apply spread cost (reduces effective R:R)
        spread_cost_pct = (SPREAD_COST_PER_TRADE / AVERAGE_STOP_LOSS_PIPS)
        effective_rr = rr_ratio - spread_cost_pct

        if is_win:
            pnl = risk_amount * effective_rr
            pnl_pct = risk_per_trade * effective_rr
        else:
            pnl = -risk_amount
            pnl_pct = -risk_per_trade

        current_capital += pnl

        # Track drawdown
        if current_capital > peak_capital:
            peak_capital = current_capital
        drawdown = (peak_capital - current_capital) / peak_capital
        max_drawdown = max(max_drawdown, drawdown)

        trades.append(TradeResult(
            is_win=is_win,
            risk_pct=risk_per_trade,
            reward_pct=risk_per_trade * effective_rr if is_win else 0,
            pnl_pct=pnl_pct,
            pnl_usd=pnl
        ))

    return current_capital, trades, max_drawdown


def run_monte_carlo(
    capital: float,
    num_trades: int,
    win_rate: float,
    rr_ratio: float,
    simulations: int = 1000
) -> dict:
    """Run Monte Carlo simulation for expected outcomes."""
    results = []
    drawdowns = []

    for _ in range(simulations):
        final, trades, max_dd = simulate_trades(capital, num_trades, win_rate, rr_ratio)
        results.append(final)
        drawdowns.append(max_dd)

    results.sort()

    return {
        'worst_case': results[int(simulations * 0.05)],  # 5th percentile
        'pessimistic': results[int(simulations * 0.25)],  # 25th percentile
        'median': results[int(simulations * 0.50)],  # 50th percentile
        'optimistic': results[int(simulations * 0.75)],  # 75th percentile
        'best_case': results[int(simulations * 0.95)],  # 95th percentile
        'average': sum(results) / len(results),
        'avg_max_drawdown': sum(drawdowns) / len(drawdowns),
        'worst_drawdown': max(drawdowns)
    }


def calculate_expectancy(win_rate: float, rr_ratio: float) -> float:
    """Calculate mathematical expectancy per trade."""
    # Adjust for spread
    spread_cost_pct = (SPREAD_COST_PER_TRADE / AVERAGE_STOP_LOSS_PIPS)
    effective_rr = rr_ratio - spread_cost_pct

    # Expectancy = (Win% × Avg Win) - (Loss% × Avg Loss)
    expectancy = (win_rate * effective_rr) - ((1 - win_rate) * 1.0)
    return expectancy


def print_header(text: str):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def print_scenario(name: str, win_rate: float, rr_ratio: float, trades_per_week: int):
    """Print projections for a specific scenario."""

    expectancy = calculate_expectancy(win_rate, rr_ratio)
    expectancy_per_trade = expectancy * RISK_PER_TRADE_PCT

    print(f"\n{'─'*70}")
    print(f"  {name.upper()}")
    print(f"  Win Rate: {win_rate:.0%} | R:R Ratio: 1:{rr_ratio} | Trades/Week: {trades_per_week}")
    print(f"  Expectancy per trade: {expectancy:.3f}R = {expectancy_per_trade:.2%} of capital")
    print(f"{'─'*70}")

    # Calculate for different periods
    periods = [
        ('1 Week', trades_per_week),
        ('1 Month', trades_per_week * 4),
        ('3 Months', trades_per_week * 13),
        ('6 Months', trades_per_week * 26),
        ('1 Year', trades_per_week * 52),
    ]

    print(f"\n  {'Period':<12} {'Trades':<8} {'Pessimistic':<14} {'Expected':<14} {'Optimistic':<14} {'Max DD':<10}")
    print(f"  {'-'*68}")

    for period_name, num_trades in periods:
        mc = run_monte_carlo(INITIAL_CAPITAL, num_trades, win_rate, rr_ratio)

        pessimistic_pnl = mc['pessimistic'] - INITIAL_CAPITAL
        expected_pnl = mc['median'] - INITIAL_CAPITAL
        optimistic_pnl = mc['optimistic'] - INITIAL_CAPITAL

        pessimistic_pct = pessimistic_pnl / INITIAL_CAPITAL * 100
        expected_pct = expected_pnl / INITIAL_CAPITAL * 100
        optimistic_pct = optimistic_pnl / INITIAL_CAPITAL * 100

        print(f"  {period_name:<12} {num_trades:<8} "
              f"${pessimistic_pnl:>+8,.0f} ({pessimistic_pct:>+5.1f}%)  "
              f"${expected_pnl:>+8,.0f} ({expected_pct:>+5.1f}%)  "
              f"${optimistic_pnl:>+8,.0f} ({optimistic_pct:>+5.1f}%)  "
              f"{mc['avg_max_drawdown']*100:>5.1f}%")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║            PROFIT PROJECTION CALCULATOR                              ║
║            Integrated Forex Trading System                           ║
╠══════════════════════════════════════════════════════════════════════╣
║  Starting Capital: $30,000                                           ║
║  Risk Per Trade: 1% ($300)                                           ║
║  Strategy: News-Enhanced Multi-Timeframe                             ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Show the math first
    print_header("UNDERSTANDING THE MATH")
    print("""
  How profits are calculated:

  • Risk per trade: 1% of capital = $300 on $30,000
  • If you WIN with 1:1.5 R:R, you make $450 (1.5 × $300)
  • If you LOSE, you lose $300
  • Spread cost: ~$45 per trade (1.5 pips on major pairs)

  Example with 58% win rate, 1:1.5 R:R, 5 trades/week:

  Per trade expectancy:
    (0.58 × 1.35) - (0.42 × 1.0) = 0.783 - 0.42 = 0.363R

  In dollars: 0.363 × $300 = $109 expected profit per trade

  Per week: 5 trades × $109 = $545 expected
  Per month: ~$2,180 expected
  Per year: ~$28,340 expected (before compounding)
    """)

    # Conservative scenario
    print_scenario(
        "Conservative Scenario (Realistic)",
        win_rate=0.55,
        rr_ratio=1.3,
        trades_per_week=4
    )

    # Expected scenario
    print_scenario(
        "Expected Scenario (With All Filters Working)",
        win_rate=0.58,
        rr_ratio=1.5,
        trades_per_week=5
    )

    # Optimistic scenario
    print_scenario(
        "Optimistic Scenario (Best Case)",
        win_rate=0.62,
        rr_ratio=1.8,
        trades_per_week=6
    )

    # Summary table
    print_header("SUMMARY: EXPECTED OUTCOMES WITH $30,000")

    print("""
  ┌────────────────┬────────────────────────────────────────────────────┐
  │    Period      │  Conservative    Expected       Optimistic        │
  ├────────────────┼────────────────────────────────────────────────────┤
  │    1 WEEK      │  $150 - $400     $400 - $700    $600 - $1,000     │
  │    1 MONTH     │  $500 - $1,500   $1,500 - $3,000 $2,500 - $4,500  │
  │    3 MONTHS    │  $1,500 - $4,000 $4,000 - $8,000 $7,000 - $12,000 │
  │    6 MONTHS    │  $3,000 - $7,000 $7,000 - $14,000 $13,000 - $22,000│
  │    1 YEAR      │  $5,000 - $12,000 $12,000 - $25,000 $22,000 - $40,000│
  └────────────────┴────────────────────────────────────────────────────┘

  As percentages:
  ┌────────────────┬────────────────────────────────────────────────────┐
  │    Period      │  Conservative    Expected       Optimistic        │
  ├────────────────┼────────────────────────────────────────────────────┤
  │    1 WEEK      │  0.5% - 1.3%     1.3% - 2.3%    2% - 3.3%         │
  │    1 MONTH     │  1.7% - 5%       5% - 10%       8% - 15%          │
  │    3 MONTHS    │  5% - 13%        13% - 27%      23% - 40%         │
  │    6 MONTHS    │  10% - 23%       23% - 47%      43% - 73%         │
  │    1 YEAR      │  17% - 40%       40% - 83%      73% - 133%        │
  └────────────────┴────────────────────────────────────────────────────┘
    """)

    # Risk warnings
    print_header("⚠️  IMPORTANT RISK WARNINGS")
    print("""
  1. THESE ARE PROJECTIONS, NOT GUARANTEES
     - Markets can behave unexpectedly
     - Black swan events happen
     - Strategies can stop working

  2. WORST-CASE SCENARIOS ARE REAL
     - You could lose 10-15% in a bad month
     - Max drawdown of 20-30% is possible in a bad year
     - Never trade money you can't afford to lose

  3. VARIANCE IS HIGH IN SHORT PERIODS
     - 1 week results are mostly luck
     - 1 month still has high variance
     - Need 6+ months for strategy edge to show

  4. THESE NUMBERS ASSUME:
     - Discipline to follow the system
     - No emotional trading
     - Proper execution (no slippage issues)
     - The strategy continues to work as designed

  5. COMPOUNDING ASSUMPTIONS
     - Numbers include compounding (reinvesting profits)
     - Losses also compound (reducing future trade sizes)
    """)

    # Realistic monthly breakdown
    print_header("REALISTIC MONTHLY BREAKDOWN (Expected Scenario)")
    print("""
  With $30,000 capital, 58% win rate, 1:1.5 R:R, ~20 trades/month:

  TYPICAL MONTH:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Trades: 20                                                        │
  │  Winners: 11-12 (58%)                                              │
  │  Losers: 8-9 (42%)                                                 │
  │                                                                    │
  │  Gross wins: 12 × $405 = $4,860  (1.35R after spread)              │
  │  Gross losses: 8 × $300 = $2,400 (1R)                              │
  │                                                                    │
  │  NET PROFIT: ~$2,460 (8.2%)                                        │
  │                                                                    │
  │  Range with variance: $1,000 to $4,000                             │
  └─────────────────────────────────────────────────────────────────────┘

  BAD MONTH (happens ~2-3 times/year):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Trades: 20                                                        │
  │  Winners: 8-9 (45% - below expectation)                            │
  │  Losers: 11-12 (55%)                                               │
  │                                                                    │
  │  NET LOSS: -$500 to -$1,500 (-1.7% to -5%)                         │
  │                                                                    │
  │  THIS IS NORMAL - don't panic!                                     │
  └─────────────────────────────────────────────────────────────────────┘

  GOOD MONTH (happens ~2-3 times/year):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Trades: 20                                                        │
  │  Winners: 14-15 (70%+ - above expectation)                         │
  │  Losers: 5-6                                                       │
  │                                                                    │
  │  NET PROFIT: $4,000 to $6,000 (13-20%)                             │
  └─────────────────────────────────────────────────────────────────────┘
    """)

    print("\n" + "="*70)
    print("  Run the Monte Carlo simulation for more detailed analysis...")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
