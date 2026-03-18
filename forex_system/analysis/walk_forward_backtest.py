"""
Walk-Forward Backtester — honest evaluation of the ensemble strategy.

Runs the full pipeline:
    1. Load price data
    2. Compute all features (signals, regime, technicals)
    3. Walk-forward: for each fold, train XGBoost on past, test on future
    4. Simulate trades with realistic costs
    5. Compare to buy-and-hold
    6. Report honest metrics

Key principles:
    - NO look-ahead bias: train only on past data, test on future
    - Transaction costs included (0.1% per side)
    - Enter at NEXT day's close (can't act instantly on today's signal)
    - Position sizing via Kelly criterion
    - Buy-and-hold is always the benchmark
    - Separate reporting for bull/bear/range periods

Usage:
    python -m forex_system.analysis.walk_forward_backtest BTC-USD
    python -m forex_system.analysis.walk_forward_backtest AAPL NVDA BTC-USD
"""

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from forex_system.strategies.ensemble_meta import EnsembleMeta, EnsembleConfig, FEATURE_COLS
from forex_system.training.config import TrainingConfig
from forex_system.training.data.price_loader import UniversalPriceLoader
from forex_system.training.data.walk_forward_splitter import WalkForwardSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-22s] %(levelname)s  %(message)s",
)
log = logging.getLogger("walk_forward")


@dataclass
class Trade:
    entry_date: datetime
    exit_date: datetime
    direction: int          # +1 long, -1 short
    entry_price: float
    exit_price: float
    position_size: float    # fraction of capital
    probability: float      # XGBoost confidence
    pnl_gross: float        # before costs
    pnl_net: float          # after costs
    cost: float             # total transaction cost
    regime: str
    fold: int
    holding_days: int

    @property
    def is_winner(self) -> bool:
        return self.pnl_net > 0


@dataclass
class FoldResult:
    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_samples: int
    test_days: int
    trades: list[Trade]
    buy_hold_return: float
    model_accuracy: float

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.is_winner) / len(self.trades)

    @property
    def total_return(self) -> float:
        if not self.trades:
            return 0.0
        equity = 1.0
        for t in self.trades:
            equity *= (1 + t.pnl_net * t.position_size)
        return (equity - 1) * 100


@dataclass
class BacktestResult:
    symbol: str
    total_days: int
    folds: list[FoldResult]
    config: EnsembleConfig

    @property
    def all_trades(self) -> list[Trade]:
        return [t for f in self.folds for t in f.trades]

    @property
    def n_trades(self) -> int:
        return len(self.all_trades)

    @property
    def win_rate(self) -> float:
        trades = self.all_trades
        if not trades:
            return 0.0
        return sum(1 for t in trades if t.is_winner) / len(trades)

    @property
    def total_return(self) -> float:
        """Compounded return across all folds (out-of-sample only)."""
        equity = 1.0
        for t in self.all_trades:
            equity *= (1 + t.pnl_net * t.position_size)
        return (equity - 1) * 100

    @property
    def buy_hold_return(self) -> float:
        """Buy-and-hold return over the same test periods."""
        if not self.folds:
            return 0.0
        compound = 1.0
        for f in self.folds:
            compound *= (1 + f.buy_hold_return / 100)
        return (compound - 1) * 100

    @property
    def profit_factor(self) -> float:
        trades = self.all_trades
        gross_wins = sum(t.pnl_net for t in trades if t.pnl_net > 0)
        gross_losses = abs(sum(t.pnl_net for t in trades if t.pnl_net <= 0))
        return gross_wins / (gross_losses + 1e-10)

    @property
    def sharpe_ratio(self) -> float:
        trades = self.all_trades
        if len(trades) < 5:
            return 0.0
        returns = [t.pnl_net * t.position_size for t in trades]
        avg_hold = np.mean([t.holding_days for t in trades])
        if avg_hold < 1:
            avg_hold = 1
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret < 1e-10:
            return 0.0
        # Annualize: assume avg_hold days per trade
        trades_per_year = 252 / avg_hold
        return (mean_ret / std_ret) * np.sqrt(trades_per_year)

    @property
    def max_drawdown(self) -> float:
        trades = self.all_trades
        if not trades:
            return 0.0
        equity = [1.0]
        for t in trades:
            equity.append(equity[-1] * (1 + t.pnl_net * t.position_size))
        eq = np.array(equity)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        return abs(dd.min()) * 100

    @property
    def avg_trade_pnl(self) -> float:
        trades = self.all_trades
        if not trades:
            return 0.0
        return np.mean([t.pnl_net * 100 for t in trades])


class WalkForwardBacktester:
    """Runs walk-forward backtest with realistic trade simulation."""

    def __init__(
        self,
        ensemble_config: EnsembleConfig = None,
        train_days: int = 252,
        test_days: int = 63,
        purge_days: int = 5,
        embargo_days: int = 2,
    ):
        self.ensemble = EnsembleMeta(ensemble_config or EnsembleConfig())
        self.config = self.ensemble.config
        self.splitter = WalkForwardSplitter(
            train_window_days=train_days,
            test_window_days=test_days,
            purge_days=purge_days,
            embargo_days=embargo_days,
        )

    def run(self, symbol: str, df_price: pd.DataFrame = None) -> BacktestResult:
        """
        Run full walk-forward backtest for one symbol.

        Args:
            symbol: Asset symbol
            df_price: Optional pre-loaded OHLCV DataFrame. If None, loads from data sources.

        Returns:
            BacktestResult with all trades and metrics.
        """
        if df_price is None:
            loader = UniversalPriceLoader(TrainingConfig())
            df_price = loader.load_ohlcv(symbol, "1d")

        if df_price.empty or len(df_price) < 300:
            log.error(f"{symbol}: insufficient data ({len(df_price)} bars, need 300+)")
            return BacktestResult(symbol=symbol, total_days=0, folds=[], config=self.config)

        log.info(f"{'=' * 70}")
        log.info(f"BACKTEST: {symbol}")
        log.info(f"  Data: {df_price.index[0]:%Y-%m-%d} to {df_price.index[-1]:%Y-%m-%d} "
                 f"({len(df_price)} bars)")
        log.info(f"  Config: horizon={self.config.horizon_days}d, "
                 f"cost={self.config.cost_per_trade*100:.2f}%, "
                 f"min_conf={self.config.min_confidence:.0%}")
        log.info(f"{'=' * 70}")

        # Prepare features on FULL dataset (indicators need warmup)
        df_full = self.ensemble.prepare_features(df_price)
        labels_full = self.ensemble.generate_labels(df_full)

        # Get feature matrix
        features_full = self.ensemble.get_feature_matrix(df_full)

        # Walk-forward splits
        splits = self.splitter.split_dataframe(df_full)

        if not splits:
            log.error(f"{symbol}: no valid walk-forward folds")
            return BacktestResult(symbol=symbol, total_days=len(df_price), folds=[], config=self.config)

        log.info(f"  Walk-forward: {len(splits)} folds")
        fold_results = []

        for train_df, test_df, fold in splits:
            fold_result = self._run_fold(
                symbol, fold, train_df, test_df,
                features_full, labels_full, df_full,
            )
            if fold_result:
                fold_results.append(fold_result)

        result = BacktestResult(
            symbol=symbol,
            total_days=len(df_price),
            folds=fold_results,
            config=self.config,
        )

        _print_backtest_summary(result)
        return result

    def _run_fold(
        self,
        symbol: str,
        fold,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        features_full: pd.DataFrame,
        labels_full: pd.Series,
        df_full: pd.DataFrame,
    ) -> FoldResult | None:
        """Run one walk-forward fold: train on past, trade on future."""

        # Extract train features/labels
        train_features = features_full.loc[train_df.index].copy()
        train_labels = labels_full.loc[train_df.index].copy()

        # Train model
        model = self.ensemble.train(train_features, train_labels)
        if model is None:
            return None

        # Extract test features
        test_features = features_full.loc[test_df.index].copy()
        test_labels = labels_full.loc[test_df.index].copy()

        # Predict on test period
        directions, probabilities = self.ensemble.predict(model, test_features)

        # Model accuracy on test set (where we have labels)
        valid_labels = test_labels.dropna()
        if len(valid_labels) > 0:
            pred_for_eval = directions[test_features.index.isin(valid_labels.index)]
            # Map: direction +1 → label 1, direction -1 → label 0
            pred_labels = (pred_for_eval > 0).astype(int)
            actual_labels = valid_labels.values[:len(pred_labels)]
            accuracy = (pred_labels == actual_labels).mean()
        else:
            accuracy = 0.0

        # Simulate trades
        trades = self._simulate_trades(
            test_df, df_full, directions, probabilities, test_features, fold.fold_index
        )

        # Buy-and-hold return for the test period
        if not test_df.empty and len(test_df) > 1:
            bh_return = (test_df["close"].iloc[-1] / test_df["close"].iloc[0] - 1) * 100
        else:
            bh_return = 0.0

        fold_result = FoldResult(
            fold_index=fold.fold_index,
            train_start=fold.train_start.strftime("%Y-%m-%d"),
            train_end=fold.train_end.strftime("%Y-%m-%d"),
            test_start=fold.test_start.strftime("%Y-%m-%d"),
            test_end=fold.test_end.strftime("%Y-%m-%d"),
            train_samples=len(train_df),
            test_days=len(test_df),
            trades=trades,
            buy_hold_return=bh_return,
            model_accuracy=accuracy,
        )

        wr = f"{fold_result.win_rate:.0%}" if trades else "N/A"
        log.info(
            f"  Fold {fold.fold_index}: "
            f"test [{fold.test_start:%Y-%m-%d}→{fold.test_end:%Y-%m-%d}] "
            f"| {len(trades)} trades, WR={wr}, "
            f"return={fold_result.total_return:+.1f}%, "
            f"B&H={bh_return:+.1f}%, acc={accuracy:.1%}"
        )

        return fold_result

    def _simulate_trades(
        self,
        test_df: pd.DataFrame,
        df_full: pd.DataFrame,
        directions: np.ndarray,
        probabilities: np.ndarray,
        test_features: pd.DataFrame,
        fold_index: int,
    ) -> list[Trade]:
        """
        Simulate trades on the test period with realistic execution.

        Rules:
            - Signal on day T → enter at day T+1 close
            - Exit when: (a) opposite signal, (b) strategy exit signal, (c) max holding period
            - Transaction cost on both entry and exit
            - Position size from Kelly criterion
        """
        cfg = self.config
        trades = []
        position = None  # None when flat
        max_hold = cfg.horizon_days * 3  # max 3x the prediction horizon

        test_dates = test_df.index.tolist()

        for i in range(len(test_dates) - 1):
            date = test_dates[i]
            next_date = test_dates[i + 1] if i + 1 < len(test_dates) else None

            if next_date is None:
                break

            direction = int(directions[i])
            probability = float(probabilities[i])

            # Get regime for position sizing
            regime = df_full.loc[date, "regime"] if date in df_full.index else "ranging"
            is_volatile = df_full.loc[date, "volatile"] if date in df_full.index else False
            regime_code = df_full.loc[date, "regime_code"] if date in df_full.index else 0

            # Position sizing
            size = self.ensemble.compute_position_size(probability, regime_code, bool(is_volatile))

            # Exit logic (if we have a position)
            if position is not None:
                holding = (date - position["entry_date"]).days
                should_exit = False
                reason = ""

                # Max holding period
                if holding >= max_hold:
                    should_exit = True
                    reason = "max_hold"
                # Opposite signal with conviction
                elif direction != 0 and direction != position["direction"] and size > 0:
                    should_exit = True
                    reason = "reversal"
                # Strategy exit signals
                elif position["direction"] == 1 and date in df_full.index:
                    rsi_exit = df_full.loc[date, "rsi_exit"] if "rsi_exit" in df_full.columns else False
                    if rsi_exit and position.get("entry_signal") == "rsi":
                        should_exit = True
                        reason = "rsi_exit"
                elif position["direction"] == -1 and date in df_full.index:
                    rsi_exit = df_full.loc[date, "rsi_exit"] if "rsi_exit" in df_full.columns else False
                    if rsi_exit and position.get("entry_signal") == "rsi":
                        should_exit = True
                        reason = "rsi_exit"

                if should_exit:
                    exit_price = float(test_df.loc[next_date, "close"])
                    entry_price = position["entry_price"]
                    pos_dir = position["direction"]

                    # Gross P&L
                    if pos_dir == 1:
                        pnl_gross = (exit_price / entry_price - 1)
                    else:
                        pnl_gross = (entry_price / exit_price - 1)

                    # Costs: entry + exit
                    cost = cfg.cost_per_trade * 2
                    pnl_net = pnl_gross - cost

                    trades.append(Trade(
                        entry_date=position["entry_date"],
                        exit_date=next_date,
                        direction=pos_dir,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        position_size=position["size"],
                        probability=position["probability"],
                        pnl_gross=pnl_gross,
                        pnl_net=pnl_net,
                        cost=cost,
                        regime=position["regime"],
                        fold=fold_index,
                        holding_days=(next_date - position["entry_date"]).days,
                    ))
                    position = None

            # Entry logic (if flat and have a signal)
            if position is None and size > 0 and direction != 0:
                entry_price = float(test_df.loc[next_date, "close"])

                # Determine which strategy drove the signal
                if date in df_full.index:
                    rsi_sig = df_full.loc[date, "rsi_signal"] if "rsi_signal" in df_full.columns else 0
                    trend_sig = df_full.loc[date, "trend_signal"] if "trend_signal" in df_full.columns else 0
                    entry_signal = "rsi" if abs(rsi_sig) > 0 else ("trend" if abs(trend_sig) > 0 else "mom")
                else:
                    entry_signal = "unknown"

                position = {
                    "entry_date": next_date,
                    "entry_price": entry_price,
                    "direction": direction,
                    "size": size,
                    "probability": probability,
                    "regime": regime,
                    "entry_signal": entry_signal,
                }

        # Close any remaining position at end of test period
        if position is not None and len(test_dates) > 0:
            last_date = test_dates[-1]
            exit_price = float(test_df.loc[last_date, "close"])
            entry_price = position["entry_price"]
            pos_dir = position["direction"]

            if pos_dir == 1:
                pnl_gross = (exit_price / entry_price - 1)
            else:
                pnl_gross = (entry_price / exit_price - 1)

            cost = cfg.cost_per_trade * 2
            pnl_net = pnl_gross - cost

            trades.append(Trade(
                entry_date=position["entry_date"],
                exit_date=last_date,
                direction=pos_dir,
                entry_price=entry_price,
                exit_price=exit_price,
                position_size=position["size"],
                probability=position["probability"],
                pnl_gross=pnl_gross,
                pnl_net=pnl_net,
                cost=cost,
                regime=position["regime"],
                fold=fold_index,
                holding_days=(last_date - position["entry_date"]).days,
            ))

        return trades


def _print_backtest_summary(result: BacktestResult):
    """Print comprehensive backtest results."""
    log.info("")
    log.info("=" * 70)
    log.info(f"BACKTEST RESULTS: {result.symbol}")
    log.info("=" * 70)

    if not result.all_trades:
        log.info("  No trades generated.")
        return

    trades = result.all_trades

    log.info(f"  Total folds:       {len(result.folds)}")
    log.info(f"  Total trades:      {result.n_trades}")
    log.info(f"  Win rate:          {result.win_rate:.1%}")
    log.info(f"  Profit factor:     {result.profit_factor:.2f}")
    log.info(f"  Avg trade P&L:     {result.avg_trade_pnl:+.3f}%")
    log.info(f"  Sharpe ratio:      {result.sharpe_ratio:.2f}")
    log.info(f"  Max drawdown:      {result.max_drawdown:.1f}%")
    log.info("")
    log.info(f"  Strategy return:   {result.total_return:+.1f}%")
    log.info(f"  Buy & hold return: {result.buy_hold_return:+.1f}%")
    beats = "YES" if result.total_return > result.buy_hold_return else "NO"
    log.info(f"  Beats buy & hold:  {beats}")

    # Long vs Short breakdown
    longs = [t for t in trades if t.direction == 1]
    shorts = [t for t in trades if t.direction == -1]
    log.info("")
    log.info(f"  Long trades:  {len(longs)}, "
             f"WR={sum(1 for t in longs if t.is_winner)/max(len(longs),1):.0%}")
    log.info(f"  Short trades: {len(shorts)}, "
             f"WR={sum(1 for t in shorts if t.is_winner)/max(len(shorts),1):.0%}")

    # Regime breakdown
    log.info("")
    log.info("  By regime:")
    for regime in ["trending", "ranging", "volatile"]:
        regime_trades = [t for t in trades if t.regime == regime]
        if regime_trades:
            wr = sum(1 for t in regime_trades if t.is_winner) / len(regime_trades)
            avg_pnl = np.mean([t.pnl_net * 100 for t in regime_trades])
            log.info(f"    {regime:>10}: {len(regime_trades)} trades, "
                     f"WR={wr:.0%}, avg_pnl={avg_pnl:+.3f}%")

    # Per-fold breakdown
    log.info("")
    log.info("  Per-fold results:")
    log.info(f"  {'Fold':>4} {'Test Period':<25} {'Trades':>6} {'WR':>6} "
             f"{'Return':>8} {'B&H':>8} {'Acc':>6} {'Beats':>6}")
    log.info("  " + "-" * 78)

    for f in result.folds:
        wr = f"{f.win_rate:.0%}" if f.trades else "N/A"
        beats = "YES" if f.total_return > f.buy_hold_return else "no"
        log.info(
            f"  {f.fold_index:>4} {f.test_start} → {f.test_end}  "
            f"{f.n_trades:>6} {wr:>6} "
            f"{f.total_return:>+7.1f}% {f.buy_hold_return:>+7.1f}% "
            f"{f.model_accuracy:>5.0%} {beats:>6}"
        )

    # Transaction cost impact
    total_cost = sum(t.cost * t.position_size for t in trades) * 100
    total_gross = sum(t.pnl_gross * t.position_size for t in trades) * 100
    log.info("")
    log.info(f"  Gross return:      {total_gross:+.1f}%")
    log.info(f"  Total costs:       -{total_cost:.1f}%")
    log.info(f"  Net return:        {result.total_return:+.1f}%")

    log.info("=" * 70)


def run_backtest(
    symbols: list[str] = None,
    train_days: int = 252,
    test_days: int = 63,
    horizon: int = 3,
    cost: float = 0.001,
    min_confidence: float = 0.55,
) -> dict[str, BacktestResult]:
    """Run walk-forward backtest for multiple symbols."""
    symbols = symbols or ["BTC-USD"]

    config = EnsembleConfig(
        horizon_days=horizon,
        cost_per_trade=cost,
        min_confidence=min_confidence,
    )

    backtester = WalkForwardBacktester(
        ensemble_config=config,
        train_days=train_days,
        test_days=test_days,
    )

    loader = UniversalPriceLoader(TrainingConfig())
    results = {}

    for symbol in symbols:
        log.info(f"\nLoading {symbol}...")
        df = loader.load_ohlcv(symbol, "1d")
        result = backtester.run(symbol, df)
        results[symbol] = result

    # Cross-asset summary
    if len(results) > 1:
        log.info("\n" + "=" * 70)
        log.info("CROSS-ASSET SUMMARY")
        log.info("=" * 70)
        log.info(f"{'Symbol':<10} {'Trades':>6} {'WR':>6} {'PF':>6} "
                 f"{'Return':>8} {'B&H':>8} {'Sharpe':>7} {'MaxDD':>7} {'Beats':>6}")
        log.info("-" * 70)

        for sym, r in results.items():
            beats = "YES" if r.total_return > r.buy_hold_return else "no"
            log.info(
                f"{sym:<10} {r.n_trades:>6} {r.win_rate:>5.0%} {r.profit_factor:>6.2f} "
                f"{r.total_return:>+7.1f}% {r.buy_hold_return:>+7.1f}% "
                f"{r.sharpe_ratio:>7.2f} {r.max_drawdown:>6.1f}% {beats:>6}"
            )

    return results


if __name__ == "__main__":
    symbols = sys.argv[1:] if len(sys.argv) > 1 else None
    run_backtest(symbols)
