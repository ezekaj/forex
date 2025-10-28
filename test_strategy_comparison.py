"""
COMMERCIAL VIABILITY TEST - Strategy Comparison

Tests Random Forest vs XGBoost with improved thresholds.
GO/NO-GO GATE: Must achieve 15%+ annual return to proceed with commercial launch.
"""
from datetime import datetime, timedelta
from forex_system.services import DataService, FeatureEngineer
from forex_system.strategies import RandomForestStrategy, XGBoostStrategy
from forex_system.backtesting import BacktestEngine, BacktestMetrics

print("=" * 100)
print("COMMERCIAL VIABILITY TEST - STRATEGY COMPARISON")
print("=" * 100)
print("\nTARGET: 15%+ annual return, max 15% drawdown, Sharpe >1.5")
print("CURRENT RF PERFORMANCE: 0.09% over 6 months (0.18% annually) - NOT VIABLE")
print("\nCRITICAL CHANGE: Increasing BUY/SELL thresholds from 0.3% to 0.5%")
print("REASON: Reduce HOLD bias (93% recall), increase trade quality over quantity")
print("=" * 100)

# 1. Generate 1 year of data
print("\n1. Generating 1 year of EURUSD 1h data...")
service = DataService()
start = datetime.utcnow() - timedelta(days=365)
end = datetime.utcnow()
df = service.get_historical_data('EURUSD', '1h', start, end)
print(f"   Generated {len(df)} bars")

# 2. Generate features
print("\n2. Generating 68 features...")
engineer = FeatureEngineer()
features_df = engineer.generate_features(df)
df_aligned = df.loc[features_df.index]
print(f"   Features: {len(features_df.columns)} columns, {len(features_df)} samples")

#===============================================================================
# TEST 1: RANDOM FOREST WITH 0.5% THRESHOLD
#===============================================================================
print("\n" + "=" * 100)
print("TEST 1: RANDOM FOREST WITH 0.5% THRESHOLD")
print("=" * 100)

print("\n3. Training Random Forest with 0.5% threshold...")
rf_strategy = RandomForestStrategy(n_estimators=200, max_depth=20)
rf_labels = rf_strategy.generate_labels(
    features_df,
    lookahead_bars=5,
    buy_threshold=0.005,  # 0.5% instead of 0.3%
    sell_threshold=-0.005
)

print(f"\nLabel distribution (0.5% threshold):")
print(f"  BUY:  {(rf_labels == 1).sum()} ({(rf_labels == 1).sum() / (~rf_labels.isna()).sum() * 100:.1f}%)")
print(f"  HOLD: {(rf_labels == 0).sum()} ({(rf_labels == 0).sum() / (~rf_labels.isna()).sum() * 100:.1f}%)")
print(f"  SELL: {(rf_labels == -1).sum()} ({(rf_labels == -1).sum() / (~rf_labels.isna()).sum() * 100:.1f}%)")

rf_metrics = rf_strategy.train(features_df, rf_labels, validation_split=0.2)

# Backtest Random Forest
print("\n4. Backtesting Random Forest (6 months)...")
backtest_start = datetime.utcnow() - timedelta(days=180)
backtest_df = service.get_historical_data('EURUSD', '1h', backtest_start, end)
backtest_features = engineer.generate_features(backtest_df)
backtest_df_aligned = backtest_df.loc[backtest_features.index]

rf_engine = BacktestEngine(initial_capital=10000.0, spread_pips=1.0, slippage_pips=0.5)
rf_results = rf_engine.run(rf_strategy, backtest_df_aligned, backtest_features, verbose=False)

rf_metrics_calc = BacktestMetrics(rf_results['trades'], rf_results['equity_curve'], 10000.0)
rf_performance = rf_results['metrics']

print(f"\nRANDOM FOREST RESULTS:")
print(f"  Return: {rf_performance['total_return_pct']:.2f}% over 6 months")
print(f"  Annualized: ~{rf_performance['total_return_pct'] * 2:.2f}%")
print(f"  Trades: {rf_performance['num_trades']}")
print(f"  Win Rate: {rf_performance['win_rate']:.1f}%")
print(f"  Sharpe Ratio: {rf_performance['sharpe_ratio']:.2f}")
print(f"  Max Drawdown: {rf_performance['max_drawdown_pct']:.2f}%")

#===============================================================================
# TEST 2: XGBOOST WITH 0.5% THRESHOLD
#===============================================================================
print("\n" + "=" * 100)
print("TEST 2: XGBOOST WITH 0.5% THRESHOLD")
print("=" * 100)

print("\n5. Training XGBoost with 0.5% threshold...")
xgb_strategy = XGBoostStrategy(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)
xgb_labels = xgb_strategy.generate_labels(
    features_df,
    lookahead_bars=5,
    buy_threshold=0.005,
    sell_threshold=-0.005
)

xgb_metrics = xgb_strategy.train(features_df, xgb_labels, validation_split=0.2)

# Backtest XGBoost
print("\n6. Backtesting XGBoost (6 months)...")
xgb_engine = BacktestEngine(initial_capital=10000.0, spread_pips=1.0, slippage_pips=0.5)
xgb_results = xgb_engine.run(xgb_strategy, backtest_df_aligned, backtest_features, verbose=False)

xgb_metrics_calc = BacktestMetrics(xgb_results['trades'], xgb_results['equity_curve'], 10000.0)
xgb_performance = xgb_results['metrics']

print(f"\nXGBOOST RESULTS:")
print(f"  Return: {xgb_performance['total_return_pct']:.2f}% over 6 months")
print(f"  Annualized: ~{xgb_performance['total_return_pct'] * 2:.2f}%")
print(f"  Trades: {xgb_performance['num_trades']}")
print(f"  Win Rate: {xgb_performance['win_rate']:.1f}%")
print(f"  Sharpe Ratio: {xgb_performance['sharpe_ratio']:.2f}")
print(f"  Max Drawdown: {xgb_performance['max_drawdown_pct']:.2f}%")

#===============================================================================
# COMPARISON AND GO/NO-GO DECISION
#===============================================================================
print("\n" + "=" * 100)
print("COMPARATIVE ANALYSIS")
print("=" * 100)

print(f"\n                      Random Forest    XGBoost       Target")
print(f"-" * 65)
print(f"6-Month Return:       {rf_performance['total_return_pct']:6.2f}%       {xgb_performance['total_return_pct']:6.2f}%       7.5%+")
print(f"Annualized Est:       {rf_performance['total_return_pct']*2:6.2f}%       {xgb_performance['total_return_pct']*2:6.2f}%      15%+")
print(f"Trades Generated:     {rf_performance['num_trades']:6d}         {xgb_performance['num_trades']:6d}        30+")
print(f"Win Rate:             {rf_performance['win_rate']:6.1f}%       {xgb_performance['win_rate']:6.1f}%      50%+")
print(f"Sharpe Ratio:         {rf_performance['sharpe_ratio']:6.2f}        {xgb_performance['sharpe_ratio']:6.2f}       1.5+")
print(f"Max Drawdown:         {rf_performance['max_drawdown_pct']:6.2f}%       {xgb_performance['max_drawdown_pct']:6.2f}%      <15%")
print(f"Profit Factor:        {rf_performance['profit_factor']:6.2f}        {xgb_performance['profit_factor']:6.2f}       1.5+")

# Determine best strategy
rf_annual_est = rf_performance['total_return_pct'] * 2
xgb_annual_est = xgb_performance['total_return_pct'] * 2

best_strategy = "XGBoost" if xgb_annual_est > rf_annual_est else "Random Forest"
best_return = max(rf_annual_est, xgb_annual_est)

print("\n" + "=" * 100)
print("GO/NO-GO DECISION")
print("=" * 100)

print(f"\nBEST PERFORMER: {best_strategy}")
print(f"ESTIMATED ANNUAL RETURN: {best_return:.2f}%")
print(f"TARGET: 15%+ annual return")

if best_return >= 15.0:
    print("\n✅ GO: Strategy meets commercial viability threshold")
    print(f"   Proceed with {best_strategy} as primary strategy")
    print("   Next steps:")
    print("   1. Test on 2-year historical data")
    print("   2. Test on multiple currency pairs")
    print("   3. Integrate real-time data (Alpha Vantage)")
    print("   4. Begin 6-month paper trading with live data")
elif best_return >= 10.0:
    print("\n⚠️  CONDITIONAL GO: Strategy shows promise but needs improvement")
    print(f"   Current: {best_return:.2f}% vs target 15%+")
    print("   Options:")
    print("   1. Try higher threshold (0.7-1.0%)")
    print("   2. Implement ensemble (RF + XGBoost)")
    print("   3. Add more features or different indicators")
    print("   4. Test on trending market periods only")
else:
    print("\n❌ NO-GO: Strategy does not meet minimum commercial viability")
    print(f"   Current: {best_return:.2f}% vs target 15%+")
    print("   Recommendations:")
    print("   1. Completely rethink approach (binary classification instead of ternary)")
    print("   2. Focus on specific market conditions (trending vs ranging)")
    print("   3. Consider different ML architectures (LSTM, Transformer)")
    print("   4. Pivot to educational content only, no signals")

print("\n" + "=" * 100)
print("DETAILED PERFORMANCE METRICS")
print("=" * 100)

print("\n### RANDOM FOREST ###")
rf_metrics_calc.print_summary(rf_performance)

print("\n### XGBOOST ###")
xgb_metrics_calc.print_summary(xgb_performance)

# Save best model
if best_strategy == "XGBoost":
    xgb_strategy.save('models/xgboost_eurusd_1h_05pct.pkl')
    print(f"\nBest model saved: models/xgboost_eurusd_1h_05pct.pkl")
else:
    rf_strategy.save('models/random_forest_eurusd_1h_05pct.pkl')
    print(f"\nBest model saved: models/random_forest_eurusd_1h_05pct.pkl")

print("\n" + "=" * 100)
print("TEST COMPLETE")
print("=" * 100)
