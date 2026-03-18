"""
Walk-Forward Backtester — the only test that matters.

Runs the full pipeline end-to-end with NO cherry-picking:
  1. Load data → features → labels
  2. Expanding-window walk-forward (purge ≥ max_holding)
  3. Train ensemble per fold on PAST data only
  4. Simulate trades bar-by-bar with 4-layer exit engine
  5. Per-asset transaction costs
  6. Compare to buy-and-hold on EVERY fold
  7. Final holdout test (last 6 months, never seen during development)

If this doesn't show profit after costs, the system doesn't work. Period.
"""

import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from investment_ai.features import generate_features, get_feature_columns, normalize_features
from investment_ai.labels import generate_triple_barrier_labels, get_purge_gap
from investment_ai.exits import Position, check_exit, compute_trade_pnl
from investment_ai.sizing import (
    classify_conviction, compute_position_size, passes_cost_check,
    get_cost_per_side, ConvictionTier,
)
from investment_ai.model import EnsembleModel

from forex_system.training.config import TrainingConfig, ASSET_REGISTRY, get_asset
from forex_system.training.data.price_loader import UniversalPriceLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)-20s] %(message)s")
log = logging.getLogger("backtest")


# ── Configuration ──

@dataclass(frozen=True)
class BacktestConfig:
    # Triple barrier
    pt_sl_ratio: float = 2.0       # 2:1 reward-to-risk
    max_holding: int = 10          # bars
    atr_multiplier: float = 2.0    # stop = 2×ATR

    # Walk-forward
    min_train_bars: int = 252      # 1 year minimum training
    test_bars: int = 63            # 3 months test
    step_bars: int = 63            # slide by 3 months
    holdout_bars: int = 180        # 6 months final holdout

    # Model
    max_feature_ratio: int = 25    # samples/features ratio

    # Trading
    max_positions: int = 3
    max_hold_bars: int = 30        # 3× max_holding

    # Exit engine R:R
    stop_atr_mult: float = 2.0
    tp_atr_mult: float = 4.0      # 2:1 R:R
    trail_activation_atr: float = 1.5
    trail_distance_atr: float = 1.5


@dataclass
class TradeRecord:
    symbol: str
    fold: int
    direction: int
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    position_size: float
    pnl_gross: float
    pnl_net: float
    cost: float
    r_multiple: float
    exit_reason: str
    bars_held: int
    conviction: int
    regime: str
    model_prob: float


@dataclass
class FoldResult:
    fold_idx: int
    train_bars: int
    test_start: str
    test_end: str
    n_trades: int
    wins: int
    losses: int
    win_rate: float
    net_return: float       # compounded
    buy_hold_return: float
    beats_bh: bool
    avg_r: float
    train_acc: float
    calib_acc: float
    n_features: int
    trades: list[TradeRecord] = field(default_factory=list)


def run_backtest(
    symbol: str,
    config: BacktestConfig = None,
    df: pd.DataFrame = None,
) -> dict:
    """
    Run full walk-forward backtest for one symbol.
    Returns dict with all results.
    """
    config = config or BacktestConfig()
    if df is None:
        loader = UniversalPriceLoader(TrainingConfig())
        df = loader.load_ohlcv(symbol, "1d")

    if len(df) < config.min_train_bars + config.test_bars + 50:
        log.error(f"{symbol}: only {len(df)} bars, need ≥{config.min_train_bars + config.test_bars + 50}")
        return {"symbol": symbol, "error": "insufficient_data"}

    try:
        asset = get_asset(symbol)
        asset_class = asset.asset_class
    except KeyError:
        asset_class = "stock"

    cost_per_side = get_cost_per_side(symbol)
    purge_gap = get_purge_gap(config.max_holding)

    log.info(f"{'='*70}")
    log.info(f"BACKTEST: {symbol} ({asset_class})")
    log.info(f"  Data: {len(df)} bars, {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}")
    log.info(f"  Cost: {cost_per_side*100:.2f}%/side, Purge: {purge_gap}d")
    log.info(f"{'='*70}")

    t0 = time.perf_counter()

    # ── Generate features + labels on full data ──
    df_feat = generate_features(df, asset_class=asset_class)
    feature_cols = get_feature_columns(df_feat)
    tb = generate_triple_barrier_labels(
        df, pt_sl_ratio=config.pt_sl_ratio,
        max_holding=config.max_holding, atr_multiplier=config.atr_multiplier,
    )
    df_feat["tb_label"] = tb["label"]
    df_feat["tb_holding"] = tb["holding_period"]
    df_feat["tb_barrier"] = tb["barrier_hit"]

    # Compute ATR for exit engine (use raw ATR, not normalized)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df_feat["_atr14"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    # Hurst for regime (already in features if UPGRADED_FEATURES worked)
    if "hurst_exponent" not in df_feat.columns:
        df_feat["hurst_exponent"] = 0.5
    if "regime" not in df_feat.columns and "hurst_regime" not in df_feat.columns:
        df_feat["_regime"] = "ranging"
    else:
        hurst_col = "hurst_regime" if "hurst_regime" in df_feat.columns else "hurst_exponent"
        df_feat["_regime"] = df_feat[hurst_col].apply(
            lambda h: "trending" if h > 0.55 else ("ranging" if h > 0.45 else "mean_reverting")
        ) if hurst_col == "hurst_exponent" else df_feat[hurst_col].map(
            {1: "trending", 0: "ranging", -1: "mean_reverting"}
        ).fillna("ranging")

    # Signal consensus (if available)
    if "signal_consensus" not in df_feat.columns:
        df_feat["signal_consensus"] = 0.0

    # ── Determine fold boundaries ──
    n = len(df_feat)
    holdout_start = n - config.holdout_bars
    dev_end = holdout_start

    # Warmup: need min_train_bars of valid features (skip first 260 for indicator warmup)
    warmup = 260
    first_test_start = warmup + config.min_train_bars + purge_gap

    folds = []
    test_start = first_test_start
    fold_idx = 0
    while test_start + config.test_bars <= dev_end:
        folds.append({
            "train_start": warmup,  # expanding: always from warmup
            "train_end": test_start - purge_gap,
            "test_start": test_start,
            "test_end": min(test_start + config.test_bars, dev_end),
            "fold_idx": fold_idx,
        })
        test_start += config.step_bars
        fold_idx += 1

    log.info(f"  Walk-forward: {len(folds)} development folds + 1 holdout")
    log.info(f"  Holdout: bars {holdout_start}-{n} ({df_feat.index[holdout_start]:%Y-%m-%d} → {df_feat.index[-1]:%Y-%m-%d})")

    # ── Run each fold ──
    all_fold_results = []
    all_trades = []

    for fold in folds:
        fold_result, trades = _run_one_fold(
            df_feat, feature_cols, tb, fold, config, symbol, asset_class, cost_per_side,
        )
        if fold_result:
            all_fold_results.append(fold_result)
            all_trades.extend(trades)

    # ── Run holdout ──
    holdout_fold = {
        "train_start": warmup,
        "train_end": holdout_start - purge_gap,
        "test_start": holdout_start,
        "test_end": n,
        "fold_idx": -1,  # -1 = holdout
    }
    holdout_result, holdout_trades = _run_one_fold(
        df_feat, feature_cols, tb, holdout_fold, config, symbol, asset_class, cost_per_side,
    )

    elapsed = time.perf_counter() - t0

    # ── Aggregate results ──
    result = _aggregate_results(
        symbol, all_fold_results, all_trades, holdout_result, holdout_trades, config, elapsed,
    )

    _print_results(result)
    return result


def _run_one_fold(df_feat, feature_cols, tb, fold, config, symbol, asset_class, cost_per_side):
    """Train on past, simulate trades on test period."""
    train_slice = df_feat.iloc[fold["train_start"]:fold["train_end"]]
    test_slice = df_feat.iloc[fold["test_start"]:fold["test_end"]]
    fold_idx = fold["fold_idx"]

    # Prepare training data
    train_valid = train_slice.dropna(subset=["tb_label"])
    if len(train_valid) < 100:
        return None, []

    X_train = train_valid[feature_cols]
    y_train = train_valid["tb_label"]
    labels_df = tb.iloc[fold["train_start"]:fold["train_end"]]

    # Train model
    model = EnsembleModel(max_feature_ratio=config.max_feature_ratio)
    metrics = model.train(X_train, y_train, labels_df=labels_df)

    if metrics.get("error"):
        return None, []

    # Predict on test period
    X_test = test_slice[feature_cols]
    # Handle NaN features in test (forward fill then drop remaining)
    X_test_clean = X_test.ffill().dropna(axis=0)
    if len(X_test_clean) < 10:
        return None, []

    # Only predict on rows we have features for
    test_indices = X_test_clean.index
    directions, probabilities, n_agree = model.predict(X_test_clean)

    # ── Bar-by-bar trade simulation ──
    trades = []
    open_positions = []
    equity = 1.0

    for i in range(len(test_indices) - 1):
        idx = test_indices[i]
        next_idx = test_indices[min(i + 1, len(test_indices) - 1)]
        row = df_feat.loc[idx]
        next_row = df_feat.loc[next_idx]

        direction = int(directions[i])
        prob = float(probabilities[i])
        n_agr = int(n_agree[i])

        cur_high = float(row["high"])
        cur_low = float(row["low"])
        cur_close = float(row["close"])
        cur_atr = float(row["_atr14"]) if not np.isnan(row["_atr14"]) else cur_close * 0.02
        regime = str(row.get("_regime", "ranging"))
        hurst = float(row.get("hurst_exponent", 0.5))
        sig_consensus = float(row.get("signal_consensus", 0))

        # ── Check exits on open positions ──
        closed_this_bar = []
        for pos in open_positions:
            exit_signals = check_exit(
                pos, cur_high, cur_low, cur_close,
                max_hold_bars=config.max_hold_bars,
                opposite_confidence=0.0,  # simplified: no signal reversal exit
            )
            for es in exit_signals:
                if es.pct_to_close >= 1.0:
                    # Full exit
                    pnl = compute_trade_pnl(pos, es, cost_per_side)
                    trades.append(TradeRecord(
                        symbol=symbol, fold=fold_idx, direction=pos.direction,
                        entry_date=pos.entry_date, exit_date=idx,
                        entry_price=pos.entry_price, exit_price=pnl["exit_price"],
                        position_size=pos.size, pnl_gross=pnl["pnl_gross"],
                        pnl_net=pnl["pnl_net"], cost=pnl["cost"],
                        r_multiple=pnl["r_multiple"], exit_reason=pnl["exit_reason"],
                        bars_held=pnl["bars_held"], conviction=pos.conviction,
                        regime=pos.regime, model_prob=pos.probability if hasattr(pos, 'probability') else prob,
                    ))
                    equity *= (1 + pnl["pnl_net"] * pos.size)
                    closed_this_bar.append(pos)
                # Partial exits just reduce position size (handled inside check_exit)

        for pos in closed_this_bar:
            if pos in open_positions:
                open_positions.remove(pos)

        # ── Entry decision ──
        if len(open_positions) >= config.max_positions:
            continue

        # Conviction
        conviction = classify_conviction(prob, sig_consensus, regime, hurst, n_agr)
        if conviction == ConvictionTier.WEAK:
            continue

        # Position size
        size = compute_position_size(prob, conviction, regime, asset_class, direction)
        if size <= 0:
            continue

        # Cost check
        atr_pct = cur_atr / cur_close if cur_close > 0 else 0.02
        if not passes_cost_check(prob, atr_pct, symbol):
            continue

        # Enter at next bar's open (approximate with close)
        entry_price = float(next_row["close"])
        if entry_price <= 0:
            continue

        pos = Position(
            symbol=symbol, direction=direction, entry_price=entry_price,
            entry_atr=cur_atr, entry_date=next_idx, size=size,
            conviction=int(conviction), fold_index=fold_idx,
        )
        # Store probability for logging
        pos.probability = prob
        pos.regime = regime
        open_positions.append(pos)

    # Close any remaining positions at end of test
    if open_positions:
        last_idx = test_indices[-1]
        last_close = float(df_feat.loc[last_idx, "close"])
        for pos in open_positions:
            from investment_ai.exits import ExitSignal
            es = ExitSignal("end_of_fold", last_close, 1.0)
            pnl = compute_trade_pnl(pos, es, cost_per_side)
            trades.append(TradeRecord(
                symbol=symbol, fold=fold_idx, direction=pos.direction,
                entry_date=pos.entry_date, exit_date=last_idx,
                entry_price=pos.entry_price, exit_price=last_close,
                position_size=pos.size, pnl_gross=pnl["pnl_gross"],
                pnl_net=pnl["pnl_net"], cost=pnl["cost"],
                r_multiple=pnl["r_multiple"], exit_reason="end_of_fold",
                bars_held=pnl["bars_held"], conviction=pos.conviction,
                regime=pos.regime, model_prob=pos.probability if hasattr(pos, 'probability') else 0.5,
            ))

    # Buy-and-hold return for test period
    test_prices = df_feat.loc[test_indices, "close"]
    if len(test_prices) > 1:
        bh_return = (float(test_prices.iloc[-1]) / float(test_prices.iloc[0]) - 1) * 100
    else:
        bh_return = 0.0

    # Strategy return
    strat_equity = 1.0
    for t in trades:
        strat_equity *= (1 + t.pnl_net * t.position_size)
    strat_return = (strat_equity - 1) * 100

    wins = sum(1 for t in trades if t.pnl_net > 0)
    losses = len(trades) - wins
    wr = wins / len(trades) if trades else 0
    avg_r = np.mean([t.r_multiple for t in trades]) if trades else 0

    fold_label = f"Fold {fold_idx}" if fold_idx >= 0 else "HOLDOUT"
    test_start_date = df_feat.index[fold["test_start"]].strftime("%Y-%m-%d")
    test_end_date = df_feat.index[min(fold["test_end"] - 1, len(df_feat) - 1)].strftime("%Y-%m-%d")

    beats = strat_return > bh_return
    log.info(
        f"  {fold_label}: [{test_start_date}→{test_end_date}] "
        f"{len(trades)} trades, WR={wr:.0%}, strat={strat_return:+.1f}%, "
        f"B&H={bh_return:+.1f}%, {'BEATS' if beats else 'loses'}"
    )

    fr = FoldResult(
        fold_idx=fold_idx, train_bars=fold["train_end"] - fold["train_start"],
        test_start=test_start_date, test_end=test_end_date,
        n_trades=len(trades), wins=wins, losses=losses, win_rate=wr,
        net_return=strat_return, buy_hold_return=bh_return, beats_bh=beats,
        avg_r=avg_r,
        train_acc=metrics.get("avg_train_acc", 0),
        calib_acc=metrics.get("avg_calib_acc", 0),
        n_features=metrics.get("n_features_selected", 0),
        trades=trades,
    )
    return fr, trades


def _aggregate_results(symbol, folds, trades, holdout, holdout_trades, config, elapsed):
    """Compute aggregate statistics."""
    total_trades = len(trades)
    wins = sum(1 for t in trades if t.pnl_net > 0)
    losses = total_trades - wins

    # Compounded return across all folds
    equity = 1.0
    for t in trades:
        equity *= (1 + t.pnl_net * t.position_size)
    net_return = (equity - 1) * 100

    # Buy-and-hold across folds
    bh_compound = 1.0
    for f in folds:
        bh_compound *= (1 + f.buy_hold_return / 100)
    bh_return = (bh_compound - 1) * 100

    # Profit factor
    gross_wins = sum(t.pnl_net for t in trades if t.pnl_net > 0)
    gross_losses = abs(sum(t.pnl_net for t in trades if t.pnl_net <= 0))
    pf = gross_wins / (gross_losses + 1e-10)

    # Sharpe (from trade returns)
    if total_trades > 5:
        trade_rets = [t.pnl_net * t.position_size for t in trades]
        avg_hold = np.mean([t.bars_held for t in trades]) or 1
        trades_per_year = 252 / avg_hold
        sharpe = (np.mean(trade_rets) / (np.std(trade_rets) + 1e-10)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Max drawdown
    eq_curve = [1.0]
    for t in trades:
        eq_curve.append(eq_curve[-1] * (1 + t.pnl_net * t.position_size))
    eq = np.array(eq_curve)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-10)
    max_dd = abs(dd.min()) * 100

    # Exit breakdown
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    # Costs
    total_costs = sum(t.cost * t.position_size for t in trades) * 100
    total_gross = sum(t.pnl_gross * t.position_size for t in trades) * 100

    return {
        "symbol": symbol,
        "elapsed_s": elapsed,
        "n_folds": len(folds),
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / max(total_trades, 1),
        "profit_factor": pf,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "net_return": net_return,
        "gross_return": total_gross,
        "total_costs": total_costs,
        "buy_hold_return": bh_return,
        "beats_buy_hold": net_return > bh_return,
        "avg_r": np.mean([t.r_multiple for t in trades]) if trades else 0,
        "exit_breakdown": exit_reasons,
        "folds": folds,
        "holdout": holdout,
        "holdout_trades": holdout_trades,
        "config": config,
    }


def _print_results(r):
    log.info("")
    log.info("=" * 70)
    log.info(f"RESULTS: {r['symbol']}")
    log.info("=" * 70)

    if r.get("error"):
        log.info(f"  ERROR: {r['error']}")
        return

    log.info(f"  Folds: {r['n_folds']} dev + 1 holdout")
    log.info(f"  Trades: {r['total_trades']} ({r['wins']}W / {r['losses']}L)")
    log.info(f"  Win rate: {r['win_rate']:.1%}")
    log.info(f"  Profit factor: {r['profit_factor']:.2f}")
    log.info(f"  Avg R: {r['avg_r']:+.2f}")
    log.info(f"  Sharpe: {r['sharpe']:.2f}")
    log.info(f"  Max drawdown: {r['max_drawdown']:.1f}%")
    log.info(f"")
    log.info(f"  Gross return:  {r['gross_return']:+.1f}%")
    log.info(f"  Total costs:   -{r['total_costs']:.1f}%")
    log.info(f"  NET RETURN:    {r['net_return']:+.1f}%")
    log.info(f"  Buy & hold:    {r['buy_hold_return']:+.1f}%")
    log.info(f"  BEATS B&H:     {'YES' if r['beats_buy_hold'] else 'NO'}")

    # Exit breakdown
    log.info(f"")
    log.info(f"  Exit breakdown:")
    for reason, count in sorted(r["exit_breakdown"].items(), key=lambda x: -x[1]):
        log.info(f"    {reason:20s}: {count:>4} ({count/max(r['total_trades'],1)*100:.0f}%)")

    # Holdout
    if r.get("holdout"):
        h = r["holdout"]
        log.info(f"")
        log.info(f"  HOLDOUT: [{h.test_start}→{h.test_end}]")
        log.info(f"    {h.n_trades} trades, WR={h.win_rate:.0%}, "
                 f"return={h.net_return:+.1f}%, B&H={h.buy_hold_return:+.1f}%")

    log.info(f"")
    log.info(f"  Time: {r['elapsed_s']:.1f}s")
    log.info("=" * 70)
