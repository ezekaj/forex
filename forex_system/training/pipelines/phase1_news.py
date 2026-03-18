"""Phase 1 Pipeline: News-only training with walk-forward validation."""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from forex_system.training.config import TrainingConfig
from forex_system.training.data.price_loader import UniversalPriceLoader
from forex_system.training.data.news_loader import NewsLoader
from forex_system.training.data.news_price_aligner import NewsPriceAligner
from forex_system.training.data.walk_forward_splitter import WalkForwardSplitter
from forex_system.training.features.news_features import NewsFeatureExtractor
from forex_system.training.strategies.news_sentiment_strategy import NewsSentimentStrategy
from forex_system.training.evaluation.experiment_tracker import ExperimentTracker, ExperimentResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class Phase1Pipeline:
    """
    Phase 1: Train news-only sentiment model.

    Pipeline: load news → align to prices → extract features → label →
              walk-forward train → evaluate → log experiment
    """

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.price_loader = UniversalPriceLoader(self.config)
        self.news_loader = NewsLoader(self.config.NEWS_DB_PATH)
        self.aligner = NewsPriceAligner(self.price_loader, self.news_loader, self.config)
        self.feature_extractor = NewsFeatureExtractor(self.config)
        self.splitter = WalkForwardSplitter(
            train_window_days=self.config.TRAIN_WINDOW_DAYS,
            test_window_days=self.config.TEST_WINDOW_DAYS,
            purge_days=self.config.PURGE_DAYS,
            embargo_days=self.config.EMBARGO_DAYS,
        )
        self.tracker = ExperimentTracker(self.config.EXPERIMENT_DB_PATH)

    def run(
        self,
        symbols: list[str] = None,
        hyperparams: Optional[dict] = None,
        include_tfidf: bool = False,
        include_momentum: bool = True,
        forward_window: int = 3,
    ) -> ExperimentResult:
        """
        Run the full Phase 1 pipeline.

        Returns ExperimentResult with metrics from walk-forward validation.
        """
        symbols = symbols or list(self.config.MAJOR_PAIRS[:3])
        hyperparams = hyperparams or {}

        log.info(f"Phase 1: Starting news-only training for {symbols}")

        # Step 1: Align news to prices
        log.info("Step 1: Aligning news to prices...")
        aligned = self.aligner.build_training_dataset(symbols)
        if aligned.empty:
            log.error("No aligned data found. Aborting.")
            return self._empty_result(symbols, hyperparams)

        # Filter to articles with valid labels
        aligned = aligned.dropna(subset=["label"])
        log.info(f"  Aligned articles: {len(aligned)}")
        log.info(f"  Label distribution: {aligned['label'].value_counts().to_dict()}")

        # Step 2: Extract features
        log.info("Step 2: Extracting news features...")
        features, feature_names = self.feature_extractor.build_news_feature_matrix(
            aligned,
            include_tfidf=include_tfidf,
            include_momentum=include_momentum,
        )
        labels = aligned["label"].astype(int)
        log.info(f"  Features: {len(feature_names)} columns, {len(features)} samples")

        # Step 3: Walk-forward split
        log.info("Step 3: Walk-forward validation...")

        # Add date column for splitting (keep integer index for label alignment)
        features = features.copy()
        features["_split_date"] = pd.to_datetime(aligned["entry_date"].values)
        features = features.sort_values("_split_date")
        labels = labels.reindex(features.index)  # align by integer index

        # Manual walk-forward split using the date column
        from forex_system.training.data.walk_forward_splitter import WalkForwardSplitter
        start_date = features["_split_date"].min().to_pydatetime()
        end_date = features["_split_date"].max().to_pydatetime()
        fold_defs = self.splitter.split(start_date, end_date)
        log.info(f"  Folds: {len(fold_defs)}")

        if not fold_defs:
            log.warning("No valid walk-forward folds. Data range too short?")
            return self._empty_result(symbols, hyperparams)

        # Step 4: Train and evaluate each fold
        fold_results = []
        for fold in fold_defs:
            train_mask = (features["_split_date"] >= fold.train_start) & (features["_split_date"] < fold.train_end)
            test_mask = (features["_split_date"] >= fold.test_start) & (features["_split_date"] < fold.test_end)

            train_df = features[train_mask].drop(columns=["_split_date"])
            test_df = features[test_mask].drop(columns=["_split_date"])
            train_labels = labels[train_mask]
            test_labels = labels[test_mask]

            # Skip folds with too few samples or single-class labels
            if len(train_df) < 20 or len(test_df) < 10:
                continue
            if train_labels.nunique() < 2:
                continue

            result = self._run_single_fold(
                train_df, train_labels, test_df, test_labels, fold, hyperparams
            )
            fold_results.append(result)
            log.info(
                f"  Fold {fold.fold_index}: "
                f"accuracy={result['accuracy']:.3f}, "
                f"trades={result['total_trades']}, "
                f"win_rate={result.get('win_rate', 0):.3f}"
            )

        if not fold_results:
            log.warning("No valid fold results.")
            return self._empty_result(symbols, hyperparams)

        # Step 5: Aggregate and log
        agg = self._aggregate_results(fold_results)
        log.info(f"\n{'='*50}")
        log.info(f"Phase 1 Results:")
        log.info(f"  Avg accuracy:    {agg['avg_accuracy']:.3f}")
        log.info(f"  Avg win rate:    {agg['avg_win_rate']:.3f}")
        log.info(f"  Avg trades/fold: {agg['avg_trades']:.0f}")
        log.info(f"  Total folds:     {agg['total_folds']}")
        log.info(f"{'='*50}")

        strategy_params = {
            "n_estimators": hyperparams.get("n_estimators", self.config.XGBOOST_N_ESTIMATORS),
            "max_depth": hyperparams.get("max_depth", self.config.XGBOOST_MAX_DEPTH),
            "learning_rate": hyperparams.get("learning_rate", self.config.XGBOOST_LEARNING_RATE),
            "include_tfidf": include_tfidf,
            "include_momentum": include_momentum,
            "forward_window": forward_window,
        }

        experiment = ExperimentResult.create(
            phase="news",
            model_type="xgboost_news",
            symbols=symbols,
            timeframe="daily",
            hyperparams=strategy_params,
            train_metrics={"avg_train_accuracy": agg.get("avg_train_accuracy", 0)},
            val_metrics={"accuracy": agg["avg_accuracy"], "win_rate": agg["avg_win_rate"]},
            backtest_metrics={
                "win_rate": agg["avg_win_rate"],
                "total_trades": int(agg["avg_trades"] * agg["total_folds"]),
                "sharpe_ratio": agg.get("avg_sharpe", 0),
                "profit_factor": agg.get("avg_profit_factor", 0),
                "max_drawdown": agg.get("avg_max_dd", 0),
            },
            data_hash=ExperimentTracker.compute_data_hash(features),
        )

        self.tracker.log_experiment(experiment)
        gate_passed = self.tracker.passes_gate(experiment)
        log.info(f"Gate passed: {gate_passed}")

        # Save best model if gate passed
        if gate_passed and fold_results:
            best_fold = max(fold_results, key=lambda r: r["accuracy"])
            if "model" in best_fold:
                model_path = str(self.config.model_output_path / "phase1_news_best.pkl")
                best_fold["model"].save(model_path)
                experiment.model_path = model_path
                log.info(f"Model saved: {model_path}")

        return experiment

    def _run_single_fold(
        self,
        train_features: pd.DataFrame,
        train_labels: pd.Series,
        test_features: pd.DataFrame,
        test_labels: pd.Series,
        fold,
        hyperparams: dict,
    ) -> dict:
        """Train and evaluate on a single walk-forward fold."""
        strategy = NewsSentimentStrategy(
            n_estimators=hyperparams.get("n_estimators"),
            max_depth=hyperparams.get("max_depth"),
            learning_rate=hyperparams.get("learning_rate"),
            config=self.config,
        )

        # Train
        metrics = strategy.train(
            train_features, train_labels,
            val_features=test_features, val_labels=test_labels,
        )

        # Predict on test
        predictions = strategy.predict(test_features)
        probas = strategy.predict_proba(test_features)

        # Compute test metrics
        test_labels_binary = ((test_labels + 1) / 2).astype(int)
        pred_binary = ((predictions + 1) / 2).astype(int)

        accuracy = float(np.mean(test_labels_binary == pred_binary))
        total_trades = len(predictions)

        # Win rate (correct predictions)
        correct = (predictions == test_labels.values).sum()
        win_rate = correct / max(total_trades, 1)

        # Simple profit factor approximation
        wins = (predictions == test_labels.values).sum()
        losses = total_trades - wins
        profit_factor = wins / max(losses, 1)

        return {
            "fold_index": fold.fold_index,
            "accuracy": accuracy,
            "win_rate": float(win_rate),
            "total_trades": total_trades,
            "train_accuracy": metrics.get("train_accuracy", 0),
            "profit_factor": float(profit_factor),
            "model": strategy,
            "feature_importance": strategy.get_feature_importance(),
        }

    def _aggregate_results(self, fold_results: list[dict]) -> dict:
        """Average metrics across all walk-forward folds."""
        return {
            "avg_accuracy": np.mean([r["accuracy"] for r in fold_results]),
            "avg_win_rate": np.mean([r["win_rate"] for r in fold_results]),
            "avg_trades": np.mean([r["total_trades"] for r in fold_results]),
            "avg_train_accuracy": np.mean([r["train_accuracy"] for r in fold_results]),
            "avg_profit_factor": np.mean([r["profit_factor"] for r in fold_results]),
            "avg_sharpe": 0,  # TODO: compute from equity curve
            "avg_max_dd": 0,  # TODO: compute from equity curve
            "total_folds": len(fold_results),
        }

    def _empty_result(self, symbols: list[str], hyperparams: dict) -> ExperimentResult:
        return ExperimentResult.create(
            phase="news",
            model_type="xgboost_news",
            symbols=symbols,
            timeframe="daily",
            hyperparams=hyperparams,
            train_metrics={},
            val_metrics={"accuracy": 0, "win_rate": 0},
            backtest_metrics={"win_rate": 0, "total_trades": 0},
            notes="No valid data or folds",
        )
