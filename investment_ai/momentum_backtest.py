"""
Time-Series Momentum Backtest — NO ML, just proven rules.

Based on Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum"
published in Journal of Financial Economics.

The strategy:
  1. For each asset, compute momentum signal from multiple lookbacks
  2. Signal is CONTINUOUS (-1 to +1), not binary
  3. Position size = signal_strength × target_risk / asset_volatility
  4. Exit: 4-layer ATR engine (stop + trailing + time + reversal)
  5. Rebalance daily (check signal, adjust position if needed)

Why this should work when ML didn't:
  - No parameters to overfit (lookback windows are fixed from research)
  - Signal is based on PRICE TREND, not learned patterns
  - Inverse-vol sizing ensures equal risk per position
  - The exit engine protects capital regardless of entry quality

Usage:
    python3 -m investment_ai.momentum_backtest BTC-USD AAPL NVDA
"""

import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from investment_ai.exits import Position, check_exit, compute_trade_pnl, ExitSignal
from investment_ai.sizing import get_cost_per_side

from forex_system.training.config import TrainingConfig, get_asset
from forex_system.training.data.price_loader import UniversalPriceLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)-20s] %(message)s")
log = logging.getLogger("momentum")


# ── Momentum Signal ──

def compute_momentum_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time-series momentum signal from multiple lookback windows.

    Moskowitz et al. (2012): use 12-month (252d) lookback.
    We use weighted combination of 4 lookbacks for smoother signal:
      21d (1 month)  — weight 0.4 (most responsive)
      63d (3 months) — weight 0.3
      126d (6 months)— weight 0.2
      252d (12 months)— weight 0.1 (most stable)

    Signal is normalized to [-1, +1].
    """
    out = df.copy()
    close = out["close"]

    # Raw momentum (percentage return over lookback)
    for lb in [21, 63, 126, 252]:
        out[f"mom_{lb}"] = close.pct_change(lb)

    # Normalize each momentum to z-score (rolling 252d window)
    for lb in [21, 63, 126, 252]:
        raw = out[f"mom_{lb}"]
        roll_mean = raw.rolling(252, min_periods=63).mean()
        roll_std = raw.rolling(252, min_periods=63).std()
        out[f"mom_{lb}_z"] = (raw - roll_mean) / (roll_std + 1e-10)

    # Weighted signal
    out["signal"] = (
        out["mom_21_z"] * 0.4
        + out["mom_63_z"] * 0.3
        + out["mom_126_z"] * 0.2
        + out["mom_252_z"] * 0.1
    )

    # Clip to [-1, 1]
    out["signal"] = out["signal"].clip(-1, 1)

    # ATR for exit engine and position sizing
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - close.shift(1)).abs(),
        (out["low"] - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    out["atr14"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    # Realized volatility (21-day, annualized) for position sizing
    daily_ret = close.pct_change()
    out["realized_vol"] = daily_ret.rolling(21, min_periods=10).std() * np.sqrt(252)

    # SMA200 for trend filter
    out["sma200"] = close.rolling(200, min_periods=200).mean()

    return out


def compute_position_size(
    signal: float,
    realized_vol: float,
    target_annual_risk: float = 0.10,
    max_position: float = 0.05,
) -> float:
    """
    Inverse-volatility position sizing.

    Size = |signal| × (target_risk / realized_vol)
    This ensures each position contributes ~equal risk regardless of asset volatility.

    BTC (vol=60%) gets small positions. AAPL (vol=25%) gets larger positions.
    """
    if realized_vol < 0.01 or abs(signal) < 0.1:
        return 0.0

    raw_size = abs(signal) * (target_annual_risk / realized_vol)
    return min(raw_size, max_position)


# ── Backtest ──

@dataclass
class MomentumTrade:
    symbol: str
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
    signal_strength: float


def run_momentum_backtest(
    symbol: str,
    target_risk: float = 0.10,
    max_position: float = 0.05,
    max_hold: int = 30,
    use_trend_filter: bool = True,
    df: pd.DataFrame = None,
) -> dict:
    """
    Run time-series momentum backtest on one asset.

    No walk-forward needed (no training). But we skip first 300 bars for warmup
    and test on the remaining ~1500 bars.
    """
    if df is None:
        loader = UniversalPriceLoader(TrainingConfig())
        df = loader.load_ohlcv(symbol, "1d")

    if len(df) < 350:
        return {"symbol": symbol, "error": "insufficient_data"}

    try:
        asset = get_asset(symbol)
        asset_class = asset.asset_class
    except KeyError:
        asset_class = "stock"

    cost_per_side = get_cost_per_side(symbol)

    log.info(f"{'='*70}")
    log.info(f"MOMENTUM BACKTEST: {symbol} ({asset_class})")
    log.info(f"  Data: {len(df)} bars, {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}")
    log.info(f"  Cost: {cost_per_side*100:.3f}%/side")
    log.info(f"  Target risk: {target_risk*100:.0f}% annual, Max pos: {max_position*100:.0f}%")
    log.info(f"  Trend filter: {'ON' if use_trend_filter else 'OFF'}")
    log.info(f"{'='*70}")

    t0 = time.perf_counter()

    # Compute signals
    df_sig = compute_momentum_signal(df)

    # Skip warmup (need 300 bars for 252d momentum + 63d z-score normalization)
    warmup = 300
    test_data = df_sig.iloc[warmup:]

    # Simulate bar-by-bar
    trades = []
    position = None
    indices = test_data.index.tolist()

    for i in range(len(indices) - 1):
        idx = indices[i]
        next_idx = indices[i + 1]
        row = test_data.loc[idx]

        cur_high = float(row["high"])
        cur_low = float(row["low"])
        cur_close = float(row["close"])
        cur_atr = float(row["atr14"]) if not np.isnan(row["atr14"]) else cur_close * 0.02
        signal = float(row["signal"]) if not np.isnan(row["signal"]) else 0.0
        vol = float(row["realized_vol"]) if not np.isnan(row["realized_vol"]) else 0.3
        sma200 = float(row["sma200"]) if not np.isnan(row["sma200"]) else cur_close

        # ── Check exit for open position ──
        if position is not None:
            exit_signals = check_exit(
                position, cur_high, cur_low, cur_close,
                max_hold_bars=max_hold,
            )

            # Also check for signal reversal (momentum flipped)
            if position is not None and abs(signal) > 0.3:
                if np.sign(signal) != position.direction:
                    exit_signals.append(ExitSignal("signal_reversal", cur_close, 1.0))

            for es in exit_signals:
                if es.pct_to_close >= 1.0:
                    pnl = compute_trade_pnl(position, es, cost_per_side)
                    trades.append(MomentumTrade(
                        symbol=symbol, direction=position.direction,
                        entry_date=position.entry_date, exit_date=idx,
                        entry_price=position.entry_price, exit_price=pnl["exit_price"],
                        position_size=position.size,
                        pnl_gross=pnl["pnl_gross"], pnl_net=pnl["pnl_net"],
                        cost=pnl["cost"], r_multiple=pnl["r_multiple"],
                        exit_reason=pnl["exit_reason"], bars_held=pnl["bars_held"],
                        signal_strength=position.size,
                    ))
                    position = None
                    break

        # ── Entry decision ──
        if position is not None:
            continue  # already in a trade

        # Need meaningful signal
        if abs(signal) < 0.2:
            continue

        direction = 1 if signal > 0 else -1

        # Trend filter: only long above SMA200, only short below
        if use_trend_filter:
            if direction == 1 and cur_close < sma200:
                continue
            if direction == -1 and cur_close > sma200:
                continue

        # Position sizing
        size = compute_position_size(signal, vol, target_risk, max_position)
        if size < 0.005:
            continue

        # Cost check: need expected move > 3× round-trip cost
        atr_pct = cur_atr / cur_close
        expected_move = abs(signal) * atr_pct * 3  # signal × ATR as proxy for expected move
        if expected_move < cost_per_side * 2 * 3:
            continue

        # Enter at next bar close
        entry_price = float(test_data.loc[next_idx, "close"])
        if entry_price <= 0:
            continue

        position = Position(
            symbol=symbol, direction=direction, entry_price=entry_price,
            entry_atr=cur_atr, entry_date=next_idx, size=size,
            conviction=2, fold_index=0,
        )

    # Close remaining position
    if position is not None:
        last_close = float(test_data.iloc[-1]["close"])
        es = ExitSignal("end_of_test", last_close, 1.0)
        pnl = compute_trade_pnl(position, es, cost_per_side)
        trades.append(MomentumTrade(
            symbol=symbol, direction=position.direction,
            entry_date=position.entry_date, exit_date=indices[-1],
            entry_price=position.entry_price, exit_price=last_close,
            position_size=position.size,
            pnl_gross=pnl["pnl_gross"], pnl_net=pnl["pnl_net"],
            cost=pnl["cost"], r_multiple=pnl["r_multiple"],
            exit_reason="end_of_test", bars_held=pnl["bars_held"],
            signal_strength=position.size,
        ))

    elapsed = time.perf_counter() - t0

    # ── Results ──
    result = _compute_results(symbol, trades, test_data, elapsed)
    _print_results(result)
    return result


def _compute_results(symbol, trades, test_data, elapsed):
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl_net > 0)
    losses = n - wins

    # Compounded return
    equity = 1.0
    for t in trades:
        equity *= (1 + t.pnl_net * t.position_size)
    net_return = (equity - 1) * 100

    # Buy-and-hold
    bh = (float(test_data.iloc[-1]["close"]) / float(test_data.iloc[0]["close"]) - 1) * 100

    # Profit factor
    g_wins = sum(t.pnl_net for t in trades if t.pnl_net > 0)
    g_losses = abs(sum(t.pnl_net for t in trades if t.pnl_net <= 0))
    pf = g_wins / (g_losses + 1e-10)

    # Sharpe
    if n > 5:
        rets = [t.pnl_net * t.position_size for t in trades]
        avg_hold = np.mean([t.bars_held for t in trades]) or 1
        tpy = 252 / avg_hold
        sharpe = (np.mean(rets) / (np.std(rets) + 1e-10)) * np.sqrt(tpy)
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
    exits = {}
    for t in trades:
        exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1

    # Long/short
    longs = [t for t in trades if t.direction == 1]
    shorts = [t for t in trades if t.direction == -1]

    # Costs
    total_cost = sum(t.cost * t.position_size for t in trades) * 100
    total_gross = sum(t.pnl_gross * t.position_size for t in trades) * 100

    # Per-year breakdown
    years = {}
    for t in trades:
        yr = t.entry_date.year
        if yr not in years:
            years[yr] = {"trades": 0, "wins": 0, "pnl": 0.0}
        years[yr]["trades"] += 1
        if t.pnl_net > 0:
            years[yr]["wins"] += 1
        years[yr]["pnl"] += t.pnl_net * t.position_size * 100

    return {
        "symbol": symbol,
        "total_trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / max(n, 1),
        "profit_factor": pf,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "net_return": net_return,
        "gross_return": total_gross,
        "total_costs": total_cost,
        "buy_hold": bh,
        "beats_bh": net_return > bh,
        "avg_r": np.mean([t.r_multiple for t in trades]) if trades else 0,
        "exit_breakdown": exits,
        "n_long": len(longs),
        "n_short": len(shorts),
        "long_wr": sum(1 for t in longs if t.pnl_net > 0) / max(len(longs), 1),
        "short_wr": sum(1 for t in shorts if t.pnl_net > 0) / max(len(shorts), 1),
        "per_year": years,
        "elapsed": elapsed,
        "test_period": f"{test_data.index[0]:%Y-%m-%d} → {test_data.index[-1]:%Y-%m-%d}",
    }


def _print_results(r):
    log.info("")
    log.info("=" * 70)
    log.info(f"RESULTS: {r['symbol']} (Time-Series Momentum)")
    log.info("=" * 70)

    log.info(f"  Period: {r['test_period']}")
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
    log.info(f"  Buy & hold:    {r['buy_hold']:+.1f}%")
    log.info(f"  BEATS B&H:     {'YES' if r['beats_bh'] else 'NO'}")

    log.info(f"")
    log.info(f"  Long:  {r['n_long']} trades, WR={r['long_wr']:.0%}")
    log.info(f"  Short: {r['n_short']} trades, WR={r['short_wr']:.0%}")

    log.info(f"")
    log.info(f"  Exit breakdown:")
    for reason, count in sorted(r["exit_breakdown"].items(), key=lambda x: -x[1]):
        pct = count / max(r["total_trades"], 1) * 100
        log.info(f"    {reason:20s}: {count:>4} ({pct:.0f}%)")

    log.info(f"")
    log.info(f"  Per-year:")
    for yr in sorted(r["per_year"]):
        y = r["per_year"][yr]
        wr = y["wins"] / max(y["trades"], 1) * 100
        log.info(f"    {yr}: {y['trades']} trades, WR={wr:.0f}%, P&L={y['pnl']:+.1f}%")

    log.info(f"")
    log.info(f"  Time: {r['elapsed']:.1f}s")
    log.info("=" * 70)


if __name__ == "__main__":
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["BTC-USD", "AAPL", "NVDA"]

    results = {}
    for sym in symbols:
        results[sym] = run_momentum_backtest(sym)

    if len(results) > 1:
        print("\n" + "=" * 70)
        print("CROSS-ASSET SUMMARY (Time-Series Momentum)")
        print("=" * 70)
        print(f"{'Symbol':<10} {'Trades':>6} {'WR':>6} {'PF':>6} "
              f"{'Net%':>8} {'B&H%':>8} {'Sharpe':>7} {'DD%':>6} {'Beats':>6}")
        print("-" * 70)
        for sym, r in results.items():
            if r.get("error"):
                print(f"{sym:<10} ERROR: {r['error']}")
                continue
            beats = "YES" if r["beats_bh"] else "no"
            print(
                f"{sym:<10} {r['total_trades']:>6} {r['win_rate']:>5.0%} "
                f"{r['profit_factor']:>6.2f} {r['net_return']:>+7.1f}% "
                f"{r['buy_hold']:>+7.1f}% {r['sharpe']:>7.2f} "
                f"{r['max_drawdown']:>5.1f}% {beats:>6}"
            )
