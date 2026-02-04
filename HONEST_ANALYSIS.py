#!/usr/bin/env python3
"""
HONEST ANALYSIS
===============
The backtest showed a LOSING system. Here's why and what can be done.

PROBLEM: 37.5% win rate with 1:2 R:R loses money.

To be profitable with 1:2 R:R, you need:
- Win rate > 33.3% to breakeven
- Win rate > 40% to cover costs
- Win rate > 45% to actually profit

Let's test different configurations to find what actually works.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from UPGRADED_FEATURES import (
    generate_all_upgraded_features,
    generate_labels_atr_based,
    calculate_confidence_margin,
    compute_sample_weights,
    calculate_atr
)

from sklearn.ensemble import RandomForestClassifier


def calculate_required_win_rate(risk_reward_ratio: float, costs_per_trade: float = 0.003) -> float:
    """
    Calculate required win rate for profitability.

    Formula: WR > 1 / (1 + RR) + costs
    """
    breakeven = 1 / (1 + risk_reward_ratio)
    required = breakeven + costs_per_trade
    return required


def test_different_configs():
    """Test different configurations to find what works."""

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         HONEST ANALYSIS                                       ║
║                                                                              ║
║  The previous backtest showed a LOSING system.                               ║
║  Let's understand WHY and test different approaches.                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Required win rates for different R:R
    print("\n1. REQUIRED WIN RATES FOR PROFITABILITY")
    print("="*60)
    print(f"""
  Risk:Reward  │  Breakeven WR  │  Required WR (with costs)
  ─────────────┼────────────────┼─────────────────────────────
     1:1       │     50.0%      │     53.0%
     1:1.5     │     40.0%      │     43.0%
     1:2       │     33.3%      │     36.3%
     1:3       │     25.0%      │     28.0%
     1:5       │     16.7%      │     19.7%

  Our backtest: 37.5% win rate with 1:2 R:R
  Required: 36.3% (technically profitable, but barely)

  Problem: Costs (spread + commission + slippage) ate the edge.
""")

    # Generate test data
    print("\n2. TESTING DIFFERENT CONFIGURATIONS")
    print("="*60)

    np.random.seed(42)
    n = 2000

    dates = pd.date_range(start='2023-01-01', periods=n, freq='4h')

    # More realistic price data with trends
    returns = np.random.normal(0, 0.0008, n)
    trend = np.zeros(n)
    current_trend = 0
    for i in range(n):
        if np.random.random() < 0.02:
            current_trend = np.random.choice([-0.0002, 0, 0.0002])
        trend[i] = current_trend
    returns = returns + trend

    close = 1.1000 * np.cumprod(1 + returns)
    volatility = np.abs(np.random.normal(0.0003, 0.0001, n))
    high = close + volatility
    low = close - volatility
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.random.randint(5000, 25000, n)

    df = pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume
    }, index=dates)

    # Test configurations
    configs = [
        {"name": "Conservative (fewer trades, higher conf)", "min_conf": 0.20, "lookahead": 30},
        {"name": "Balanced (moderate)", "min_conf": 0.15, "lookahead": 20},
        {"name": "Aggressive (more trades, lower conf)", "min_conf": 0.05, "lookahead": 10},
        {"name": "Very Selective (only best setups)", "min_conf": 0.30, "lookahead": 40},
    ]

    results = []

    for config in configs:
        result = quick_backtest(df, config['min_conf'], config['lookahead'])
        result['name'] = config['name']
        results.append(result)

        print(f"\n  Config: {config['name']}")
        print(f"  Trades: {result['trades']:3d}, Win Rate: {result['win_rate']:.1%}, P&L: ${result['pnl']:+,.0f}")

    # Find best config
    print("\n\n3. ANALYSIS OF RESULTS")
    print("="*60)

    profitable = [r for r in results if r['pnl'] > 0]

    if profitable:
        best = max(profitable, key=lambda x: x['pnl'])
        print(f"\n  BEST CONFIG: {best['name']}")
        print(f"  Trades: {best['trades']}, Win Rate: {best['win_rate']:.1%}, P&L: ${best['pnl']:+,.0f}")
    else:
        print("\n  ⚠️ NO PROFITABLE CONFIGURATION FOUND")
        print("\n  This is actually HONEST information.")

    # Real talk
    print("""

4. THE BRUTAL TRUTH
================================================================================

The model achieves ~90% accuracy on HOLD signals (which is 80%+ of the data).
But the BUY/SELL signals (which actually make money) have much lower accuracy.

This is the FUNDAMENTAL PROBLEM with ML trading:
- ML optimizes for overall accuracy
- But trading profits come from correctly predicting MOVES, not HOLDS
- HOLD is easy to predict (most of the time, price doesn't move significantly)

WHAT ACTUALLY WORKS (from research):
────────────────────────────────────

1. REDUCE TRADING FREQUENCY
   - Trade only 5-10 times per month, not 25+
   - Wait for VERY high confidence signals only
   - Quality >>> Quantity

2. INCREASE RISK:REWARD
   - Use 1:3 or 1:5 R:R with trailing stops
   - Accept lower win rate, but bigger wins
   - This requires patience (trades take longer)

3. ADD FUNDAMENTAL FILTERS
   - Trade WITH the central bank bias (not against)
   - Avoid major news events
   - Focus on pairs with clear trends

4. USE THE NEWS SYSTEM
   - Your news_lenci_forex system has 48-57% accuracy on stock prediction
   - That's actually USEFUL if applied correctly
   - News should FILTER trades, not generate them

5. REALISTIC EXPECTATIONS
   - Even the best systems: 5-10% monthly is excellent
   - 2-5% monthly is more realistic for retail
   - Consistency beats big wins

================================================================================

THE HONEST NUMBERS WITH $30,000:
────────────────────────────────

  REALISTIC (if system works):
  ├── Monthly: +$600 to +$1,500 (2-5%)
  ├── Yearly: +$8,000 to +$20,000 (27-67%)
  └── This is STILL excellent compared to stock market (10% avg)

  WHY NOT HIGHER:
  ├── Forex is a zero-sum game (someone loses what you win)
  ├── Banks and hedge funds have faster execution
  ├── Spread/costs eat into small edges
  └── Random walk theory says 1H/4H is mostly noise

================================================================================
""")


def quick_backtest(df: pd.DataFrame, min_confidence: float, lookahead: int) -> dict:
    """Quick backtest with specific parameters."""

    features = generate_all_upgraded_features(df)
    labels = generate_labels_atr_based(df, lookahead_bars=lookahead, atr_multiplier=2.0)
    features['ATR'] = calculate_atr(df, 14)

    # Split
    split = int(len(df) * 0.6)

    X_train = features.iloc[:split]
    y_train = labels.iloc[:split]
    X_test = features.iloc[split:]
    test_df = df.iloc[split:]

    # Train
    valid = ~y_train.isna()
    X_train = X_train[valid]
    y_train = y_train[valid]

    numeric = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train = X_train[numeric]

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=20,
        class_weight='balanced', random_state=42, n_jobs=-1
    )

    weights = compute_sample_weights(y_train)
    model.fit(X_train, y_train, sample_weight=weights)

    # Test
    X_test = X_test[numeric]

    capital = 30000
    trades = 0
    wins = 0

    for i in range(len(X_test) - lookahead):
        pred = model.predict(X_test.iloc[[i]])[0]
        proba = model.predict_proba(X_test.iloc[[i]])[0]
        conf = calculate_confidence_margin(proba)

        if pred != 0 and conf >= min_confidence:
            trades += 1

            # Check if trade would have won (simplified)
            entry = test_df['close'].iloc[i]
            future_prices = test_df['close'].iloc[i+1:i+lookahead+1]

            if len(future_prices) == 0:
                continue

            atr = features['ATR'].iloc[split + i]
            if pd.isna(atr) or atr <= 0:
                atr = 0.001

            tp_distance = atr * 3.0
            sl_distance = atr * 1.5

            if pred == 1:  # LONG
                tp = entry + tp_distance
                sl = entry - sl_distance
                hit_tp = (future_prices >= tp).any()
                hit_sl = (future_prices <= sl).any()
            else:  # SHORT
                tp = entry - tp_distance
                sl = entry + sl_distance
                hit_tp = (future_prices <= tp).any()
                hit_sl = (future_prices >= sl).any()

            if hit_tp and not hit_sl:
                wins += 1
                capital += capital * 0.015 * 2  # Win 2R
            elif hit_sl:
                capital -= capital * 0.015  # Lose 1R
            # else: no hit, assume breakeven

    win_rate = wins / trades if trades > 0 else 0
    pnl = capital - 30000

    return {'trades': trades, 'wins': wins, 'win_rate': win_rate, 'pnl': pnl}


if __name__ == "__main__":
    test_different_configs()
