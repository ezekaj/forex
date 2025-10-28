"""
Backtest Random Forest strategy with realistic costs.

Tests the trained RF model on historical data with spread, slippage, and commission.
"""
from datetime import datetime, timedelta
from forex_system.services import DataService, FeatureEngineer
from forex_system.strategies import RandomForestStrategy
from forex_system.backtesting import BacktestEngine, BacktestMetrics

print("=" * 80)
print("RANDOM FOREST STRATEGY BACKTEST")
print("=" * 80)

# 1. Load trained model
print("\n1. Loading trained Random Forest model...")
strategy = RandomForestStrategy()
strategy.load('models/random_forest_eurusd_1h.pkl')

# 2. Generate test data (different from training period)
print("\n2. Generating test data (6 months)...")
service = DataService()
start = datetime.utcnow() - timedelta(days=180)  # 6 months
end = datetime.utcnow()
df = service.get_historical_data('EURUSD', '1h', start, end)
print(f"   Generated {len(df)} bars from {df.index[0]} to {df.index[-1]}")

# 3. Generate features
print("\n3. Generating features...")
engineer = FeatureEngineer()
features_df = engineer.generate_features(df)
print(f"   Features: {len(features_df.columns)} columns, {len(features_df)} samples")

# Align data with features (feature engineering drops NaN rows)
df_aligned = df.loc[features_df.index]
print(f"   Aligned data: {len(df_aligned)} bars")

# 4. Run backtest with realistic costs
print("\n4. Running backtest with realistic costs...")
print("   Costs:")
print("     • Spread: 1.0 pips (typical for EURUSD)")
print("     • Slippage: 0.5 pips (typical for retail)")
print("     • Commission: 0.01% per trade")
print("     • Position size: 10% of capital")
print()

engine = BacktestEngine(
    initial_capital=10000.0,
    spread_pips=1.0,
    slippage_pips=0.5,
    commission_pct=0.0001,
    position_size_pct=0.1,
    pip_value=0.0001
)

results = engine.run(
    strategy=strategy,
    data=df_aligned,
    features=features_df,
    verbose=True
)

# 5. Print comprehensive metrics
print("\n5. Performance Metrics:")
metrics_calculator = BacktestMetrics(
    trades=results['trades'],
    equity_curve=results['equity_curve'],
    initial_capital=10000.0
)
metrics_calculator.print_summary(results['metrics'])

# 6. Analysis and recommendations
print("\n" + "=" * 80)
print("ANALYSIS & RECOMMENDATIONS")
print("=" * 80)

metrics = results['metrics']

print("\nStrategy Viability:")
if metrics['total_return_pct'] > 10 and metrics['sharpe_ratio'] > 1.0:
    print("  ✓ Strategy shows promise - positive returns with decent risk adjustment")
elif metrics['total_return_pct'] > 0:
    print("  ⚠ Strategy marginally profitable - needs improvement")
else:
    print("  ✗ Strategy unprofitable - requires significant changes")

print("\nKey Observations:")
print(f"  • Generated {metrics['num_trades']} trades over 6 months")
print(f"  • Average {metrics['num_trades'] / 6:.1f} trades per month")

if metrics['num_trades'] < 10:
    print("  ⚠ Very few trades - model may be too conservative")
elif metrics['num_trades'] > 100:
    print("  ⚠ High trade frequency - may be overtrading")

if metrics['win_rate'] < 45:
    print("  ⚠ Low win rate - model struggles to predict profitable moves")
elif metrics['win_rate'] > 60:
    print("  ✓ Good win rate - model identifies profitable setups")

if metrics['max_drawdown_pct'] > 20:
    print("  ⚠ High drawdown - needs better risk management")
elif metrics['max_drawdown_pct'] < 10:
    print("  ✓ Low drawdown - good risk control")

print("\nNext Steps:")
if metrics['total_return_pct'] > 0:
    print("  1. Try different lookahead periods (3, 5, 10 bars)")
    print("  2. Adjust BUY/SELL thresholds (0.5%, 1.0%)")
    print("  3. Implement stop-loss and take-profit levels")
    print("  4. Test on different currency pairs")
else:
    print("  1. Review feature importance - remove low-value features")
    print("  2. Try XGBoost or ensemble of models")
    print("  3. Increase BUY/SELL thresholds to be more selective")
    print("  4. Consider different classification approach (binary instead of ternary)")

print("=" * 80)
