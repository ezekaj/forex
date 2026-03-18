"""CLI entry point for the training pipeline."""

import argparse
import sys

from forex_system.training.config import TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Forex ML Training Pipeline")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--include-tfidf", action="store_true")
    parser.add_argument("--no-momentum", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    config = TrainingConfig()

    if args.compare:
        from forex_system.training.evaluation.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(config.EXPERIMENT_DB_PATH)
        phase_map = {1: "news", 2: "price", 3: "merge"}
        history = tracker.get_history(phase=phase_map[args.phase], limit=args.top)
        if not history:
            print("No experiments found.")
            return
        for exp in history:
            bt = exp.backtest_metrics
            print(
                f"  {exp.experiment_id} | {exp.model_type:20s} | "
                f"WR={bt.get('win_rate', 0):.3f} | "
                f"Sharpe={bt.get('sharpe_ratio', 0):.2f} | "
                f"Trades={bt.get('total_trades', 0)}"
            )
        return

    if args.phase == 1:
        from forex_system.training.pipelines.phase1_news import Phase1Pipeline
        pipeline = Phase1Pipeline(config)
        result = pipeline.run(
            symbols=args.symbols,
            include_tfidf=args.include_tfidf,
            include_momentum=not args.no_momentum,
        )
        print(f"\nExperiment ID: {result.experiment_id}")
        print(f"Win Rate: {result.backtest_metrics.get('win_rate', 0):.3f}")

    elif args.phase == 2:
        print("Phase 2 not yet implemented.")

    elif args.phase == 3:
        print("Phase 3 not yet implemented.")


if __name__ == "__main__":
    main()
