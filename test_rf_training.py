"""
Test Random Forest strategy training.

Generates synthetic data, creates features, and trains RF model.
"""
from datetime import datetime, timedelta
from forex_system.services import DataService, FeatureEngineer
from forex_system.strategies import RandomForestStrategy

print("=" * 80)
print("RANDOM FOREST STRATEGY TRAINING TEST")
print("=" * 80)

# 1. Generate 1 year of EURUSD 1h data
print("\n1. Generating 1 year of EURUSD 1h data...")
service = DataService()
start = datetime.utcnow() - timedelta(days=365)
end = datetime.utcnow()
df = service.get_historical_data('EURUSD', '1h', start, end)
print(f"   Generated {len(df)} bars from {df.index[0]} to {df.index[-1]}")
print(f"   Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")

# 2. Generate 69 features
print("\n2. Generating 69 features...")
engineer = FeatureEngineer()
features_df = engineer.generate_features(df)
print(f"   Features generated: {len(features_df.columns)} columns")
print(f"   Samples after dropping NaN: {len(features_df)}")

# 3. Generate BUY/HOLD/SELL labels
print("\n3. Generating BUY/HOLD/SELL labels...")
strategy = RandomForestStrategy()
labels = strategy.generate_labels(
    features_df,
    lookahead_bars=5,
    buy_threshold=0.003,  # 0.3% = ~30 pips for EURUSD
    sell_threshold=-0.003
)
print(f"   Total labels: {len(labels)}")
print(f"   Valid labels (non-NaN): {(~labels.isna()).sum()}")
print(f"   Label distribution:")
print(f"     BUY:  {(labels == 1).sum()} ({(labels == 1).sum() / (~labels.isna()).sum() * 100:.1f}%)")
print(f"     HOLD: {(labels == 0).sum()} ({(labels == 0).sum() / (~labels.isna()).sum() * 100:.1f}%)")
print(f"     SELL: {(labels == -1).sum()} ({(labels == -1).sum() / (~labels.isna()).sum() * 100:.1f}%)")

# 4. Train Random Forest model
print("\n4. Training Random Forest model...")
print("-" * 80)
metrics = strategy.train(features_df, labels, validation_split=0.2)
print("-" * 80)

# 5. Show training summary
print("\n5. Training Summary:")
print(f"   Train accuracy: {metrics['train_accuracy']:.4f}")
print(f"   Val accuracy:   {metrics['val_accuracy']:.4f}")
print(f"   Train samples:  {metrics['train_samples']}")
print(f"   Val samples:    {metrics['val_samples']}")
print(f"   Features used:  {metrics['n_features']}")

# 6. Show top 10 most important features
print("\n6. Top 10 Most Important Features:")
strategy.print_top_features(n=10)

# 7. Save trained model
print("\n7. Saving trained model...")
strategy.save('models/random_forest_eurusd_1h.pkl')

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"\nModel Performance:")
print(f"  • Validation Accuracy: {metrics['val_accuracy']:.2%}")
print(f"  • Expected for forex: 52-58% is realistic")
print(f"  • Profitability depends on risk management, not just accuracy")
print(f"\nNext Steps:")
print(f"  1. Create backtesting framework to validate strategy")
print(f"  2. Test with different parameters (lookahead, thresholds)")
print(f"  3. Implement risk management (position sizing, stop loss)")
print("=" * 80)
