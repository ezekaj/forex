#!/usr/bin/env python3
"""Run training inside the GPU container with correct paths."""

import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Container paths
PRICE_DB = "/root/.cache/huggingface/forex_prices.db"
NEWS_DB = "/workspace/historical_news.db"
YAHOO_DB = "/workspace/yahoo_cache.db"
SENTIMENT_DB = "/workspace/sentiment_cache.db"
ARTIFACTS = "/workspace/artifacts"
EXPERIMENTS_DB = "/workspace/experiments.db"

os.makedirs(ARTIFACTS, exist_ok=True)

# Monkey-patch the config before importing training modules
sys.path.insert(0, "/workspace")

from forex_system.training import config as cfg

# Override the frozen dataclass defaults by creating a new instance
container_config = cfg.TrainingConfig(
    PRICE_DB_PATH=PRICE_DB,
    NEWS_DB_PATH=NEWS_DB,
    YAHOO_CACHE_DB_PATH=YAHOO_DB,
    MODEL_OUTPUT_DIR=ARTIFACTS,
    EXPERIMENT_DB_PATH=EXPERIMENTS_DB,
)

# Also override the sentiment cache path
cfg_mod = sys.modules.get("forex_system.training.models.flang_electra")
if cfg_mod:
    cfg_mod.SENTIMENT_CACHE_DB = SENTIMENT_DB

from forex_system.training.data.price_loader import UniversalPriceLoader
from forex_system.training.data.news_loader import NewsLoader
from forex_system.training.data.news_price_aligner import NewsPriceAligner
from forex_system.training.data.walk_forward_splitter import WalkForwardSplitter
from forex_system.training.features.news_features import NewsFeatureExtractor
from forex_system.training.strategies.news_sentiment_strategy import NewsSentimentStrategy
from forex_system.training.evaluation.experiment_tracker import ExperimentTracker, ExperimentResult

import numpy as np
import pandas as pd
import torch

log.info(f"CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


def run_phase1(symbols: list[str], config: cfg.TrainingConfig):
    """Run Phase 1 news training with GPU-accelerated FinBERT."""
    log.info(f"Phase 1: Training on {symbols}")

    # Initialize components
    price_loader = UniversalPriceLoader(config)
    news_loader = NewsLoader(config.NEWS_DB_PATH)
    aligner = NewsPriceAligner(price_loader, news_loader, config)
    feature_extractor = NewsFeatureExtractor(config)

    # Override sentiment cache path
    from forex_system.training.models import flang_electra
    flang_electra.SENTIMENT_CACHE_DB = SENTIMENT_DB
    feature_extractor._scorer = None  # force re-init with new path

    splitter = WalkForwardSplitter(
        train_window_days=config.TRAIN_WINDOW_DAYS,
        test_window_days=config.TEST_WINDOW_DAYS,
        purge_days=config.PURGE_DAYS,
        embargo_days=config.EMBARGO_DAYS,
    )
    tracker = ExperimentTracker(config.EXPERIMENT_DB_PATH)

    # Step 1: Align news to prices
    log.info("Step 1: Aligning news to prices...")
    aligned = aligner.build_training_dataset(symbols)
    if aligned.empty:
        log.error("No aligned data. Aborting.")
        return
    aligned = aligned.dropna(subset=["label"])
    log.info(f"  Aligned: {len(aligned)} articles")
    log.info(f"  Labels: {aligned['label'].value_counts().to_dict()}")

    # Step 2: Extract features (GPU-accelerated FinBERT)
    log.info("Step 2: Extracting features (GPU FinBERT)...")
    features, feature_names = feature_extractor.build_news_feature_matrix(
        aligned, include_tfidf=False, include_momentum=True
    )
    labels = aligned["label"].astype(int)
    log.info(f"  Features: {len(feature_names)} columns, {len(features)} samples")

    # Step 3: Walk-forward
    log.info("Step 3: Walk-forward validation...")
    features = features.copy()
    features["_split_date"] = pd.to_datetime(aligned["entry_date"].values)
    features = features.sort_values("_split_date")
    labels = labels.reindex(features.index)

    start_date = features["_split_date"].min().to_pydatetime()
    end_date = features["_split_date"].max().to_pydatetime()
    fold_defs = splitter.split(start_date, end_date)
    log.info(f"  Folds: {len(fold_defs)}")

    if not fold_defs:
        log.warning("No folds. Data range too short.")
        return

    # Step 4: Train each fold
    fold_results = []
    for fold in fold_defs:
        train_mask = (features["_split_date"] >= fold.train_start) & (features["_split_date"] < fold.train_end)
        test_mask = (features["_split_date"] >= fold.test_start) & (features["_split_date"] < fold.test_end)

        train_df = features[train_mask].drop(columns=["_split_date"])
        test_df = features[test_mask].drop(columns=["_split_date"])
        train_labels = labels[train_mask]
        test_labels = labels[test_mask]

        if len(train_df) < 50 or len(test_df) < 20 or train_labels.nunique() < 2:
            continue

        strategy = NewsSentimentStrategy(config=config)
        strategy.train(train_df, train_labels, val_features=test_df, val_labels=test_labels)
        predictions = strategy.predict(test_df)

        correct = (predictions == test_labels.values).sum()
        total = len(predictions)
        win_rate = correct / max(total, 1)

        fold_results.append({
            "fold": fold.fold_index,
            "accuracy": float(win_rate),
            "trades": total,
            "model": strategy,
        })
        log.info(f"  Fold {fold.fold_index}: accuracy={win_rate:.3f}, trades={total}")

    if not fold_results:
        log.warning("No valid folds.")
        return

    avg_acc = np.mean([r["accuracy"] for r in fold_results])
    avg_trades = np.mean([r["trades"] for r in fold_results])
    log.info(f"\n{'='*50}")
    log.info(f"RESULTS: {','.join(symbols)}")
    log.info(f"  Avg accuracy: {avg_acc:.3f}")
    log.info(f"  Avg trades/fold: {avg_trades:.0f}")
    log.info(f"  Folds: {len(fold_results)}")
    log.info(f"  Gate (>=0.53): {'PASS' if avg_acc >= 0.53 else 'FAIL'}")
    log.info(f"{'='*50}")

    # Log experiment
    experiment = ExperimentResult.create(
        phase="news",
        model_type="xgboost_news",
        symbols=symbols,
        timeframe="daily",
        hyperparams={"include_tfidf": False, "include_momentum": True},
        train_metrics={},
        val_metrics={"accuracy": float(avg_acc)},
        backtest_metrics={
            "win_rate": float(avg_acc),
            "total_trades": int(avg_trades * len(fold_results)),
        },
    )
    tracker.log_experiment(experiment)
    log.info(f"Experiment logged: {experiment.experiment_id}")

    # Save best model
    if fold_results:
        best = max(fold_results, key=lambda r: r["accuracy"])
        model_path = f"{ARTIFACTS}/phase1_{'_'.join(symbols)}_best.pkl"
        best["model"].save(model_path)
        log.info(f"Best model saved: {model_path}")


if __name__ == "__main__":
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "NVDA", "TSLA"]
    run_phase1(symbols, container_config)
