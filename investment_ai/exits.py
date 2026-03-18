"""
4-Layer Exit Engine — ATR stop/target + trailing stop + time exit + signal reversal.

Ported from LEVEL_UP_STRATEGY's R:R logic:
  - TP1 at 2×ATR: close 50%, move stop to breakeven
  - TP2 at 3×ATR: close 25% more
  - Runner: trail remaining 25% with 1.5×ATR trailing stop

Layer 1: Hard ATR limits (stop-loss at 2×ATR, take-profit at 4×ATR)
Layer 2: Trailing stop (activates at 1.5×ATR profit, trails at 1.5×ATR)
Layer 3: Time-based exit (max 3× prediction horizon)
Layer 4: Signal-based exit (opposite signal with >70% confidence)
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class ExitSignal:
    reason: str      # "stop_loss", "take_profit", "trailing_stop", "time_exit", "signal_reversal"
    price: float     # exit price
    pct_to_close: float = 1.0  # fraction of position to close (1.0 = full, 0.5 = half)


@dataclass
class Position:
    symbol: str
    direction: int              # +1 long, -1 short
    entry_price: float
    entry_atr: float            # ATR(14) at entry time
    entry_date: datetime
    size: float                 # fraction of capital
    conviction: int             # 0-4 (WEAK→KILLER)
    fold_index: int

    # Dynamic state (mutated during trade)
    remaining_size: float = 0.0  # starts equal to size, reduced by partial TPs
    bars_held: int = 0
    trailing_active: bool = False
    trailing_stop: float = 0.0
    breakeven_stop: bool = False  # True after first partial TP
    highest_close: float = 0.0   # for trailing (long)
    lowest_close: float = float("inf")  # for trailing (short)

    def __post_init__(self):
        self.remaining_size = self.size
        self.highest_close = self.entry_price
        self.lowest_close = self.entry_price

    @property
    def stop_loss(self) -> float:
        if self.breakeven_stop:
            return self.entry_price  # moved to breakeven after first TP
        if self.direction == 1:
            return self.entry_price - 2 * self.entry_atr
        return self.entry_price + 2 * self.entry_atr

    @property
    def take_profit_1(self) -> float:
        """First TP at 2×ATR (close 50%)."""
        return self.entry_price + self.direction * 2 * self.entry_atr

    @property
    def take_profit_2(self) -> float:
        """Second TP at 3×ATR (close 25%)."""
        return self.entry_price + self.direction * 3 * self.entry_atr

    @property
    def take_profit_full(self) -> float:
        """Full target at 4×ATR (2:1 R:R)."""
        return self.entry_price + self.direction * 4 * self.entry_atr


def check_exit(
    position: Position,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    max_hold_bars: int = 30,
    opposite_confidence: float = 0.0,
) -> list[ExitSignal]:
    """
    Check all 4 exit layers for a position against the current bar.

    Args:
        position: The open position.
        bar_high: Current bar's high price.
        bar_low: Current bar's low price.
        bar_close: Current bar's close price.
        max_hold_bars: Maximum bars to hold (Layer 3).
        opposite_confidence: Model confidence in opposite direction (Layer 4).

    Returns:
        List of ExitSignals (may be empty if no exit triggered, or contain
        partial exits for TP scaling).
    """
    position.bars_held += 1
    exits = []
    d = position.direction

    # Update tracking
    if d == 1:
        position.highest_close = max(position.highest_close, bar_close)
    else:
        position.lowest_close = min(position.lowest_close, bar_close)

    # ── Layer 1: Hard ATR limits ──

    # Stop-loss
    sl = position.stop_loss
    if d == 1 and bar_low <= sl:
        exits.append(ExitSignal("stop_loss", sl, pct_to_close=1.0))
        return exits  # Full exit, no further checks
    if d == -1 and bar_high >= sl:
        exits.append(ExitSignal("stop_loss", sl, pct_to_close=1.0))
        return exits

    # Take-profit scaling (from LEVEL_UP_STRATEGY)
    # TP1 at 2×ATR: close 50%, move stop to breakeven
    tp1 = position.take_profit_1
    if not position.breakeven_stop:
        if (d == 1 and bar_high >= tp1) or (d == -1 and bar_low <= tp1):
            exits.append(ExitSignal("take_profit_1", tp1, pct_to_close=0.50))
            position.breakeven_stop = True
            position.remaining_size *= 0.50

    # TP2 at 3×ATR: close half of remaining (25% of original)
    tp2 = position.take_profit_2
    if position.breakeven_stop and position.remaining_size > position.size * 0.3:
        if (d == 1 and bar_high >= tp2) or (d == -1 and bar_low <= tp2):
            exits.append(ExitSignal("take_profit_2", tp2, pct_to_close=0.50))
            position.remaining_size *= 0.50

    # Full TP at 4×ATR: close everything
    tp_full = position.take_profit_full
    if (d == 1 and bar_high >= tp_full) or (d == -1 and bar_low <= tp_full):
        exits.append(ExitSignal("take_profit", tp_full, pct_to_close=1.0))
        return exits

    # ── Layer 2: Trailing stop ──

    unrealized = d * (bar_close - position.entry_price)
    if unrealized >= 1.5 * position.entry_atr:
        position.trailing_active = True

    if position.trailing_active:
        if d == 1:
            new_trail = position.highest_close - 1.5 * position.entry_atr
            position.trailing_stop = max(position.trailing_stop, new_trail)
            if bar_low <= position.trailing_stop:
                exits.append(ExitSignal("trailing_stop", position.trailing_stop, pct_to_close=1.0))
                return exits
        else:
            new_trail = position.lowest_close + 1.5 * position.entry_atr
            if position.trailing_stop == 0:
                position.trailing_stop = new_trail
            else:
                position.trailing_stop = min(position.trailing_stop, new_trail)
            if bar_high >= position.trailing_stop:
                exits.append(ExitSignal("trailing_stop", position.trailing_stop, pct_to_close=1.0))
                return exits

    # ── Layer 3: Time-based exit ──

    if position.bars_held >= max_hold_bars:
        exits.append(ExitSignal("time_exit", bar_close, pct_to_close=1.0))
        return exits

    # ── Layer 4: Signal reversal ──

    if opposite_confidence > 0.70:
        exits.append(ExitSignal("signal_reversal", bar_close, pct_to_close=1.0))
        return exits

    return exits


def compute_trade_pnl(
    position: Position,
    exit_signal: ExitSignal,
    cost_per_side: float,
) -> dict:
    """
    Compute P&L for a (partial or full) exit.

    Returns dict with: pnl_gross, pnl_net, cost, size_closed, r_multiple
    """
    d = position.direction
    entry = position.entry_price
    exit_price = exit_signal.price
    size_closed = position.remaining_size * exit_signal.pct_to_close

    # Gross P&L (before costs)
    if d == 1:
        pnl_gross = (exit_price / entry - 1)
    else:
        pnl_gross = (entry / exit_price - 1)

    # Costs (entry + exit)
    cost = cost_per_side * 2

    pnl_net = pnl_gross - cost

    # R-multiple: how many R (risk units) did we make?
    risk_per_r = 2 * position.entry_atr / entry  # stop-loss distance as fraction
    r_multiple = pnl_gross / risk_per_r if risk_per_r > 0 else 0

    return {
        "pnl_gross": pnl_gross,
        "pnl_net": pnl_net,
        "cost": cost,
        "size_closed": size_closed,
        "r_multiple": r_multiple,
        "exit_reason": exit_signal.reason,
        "exit_price": exit_price,
        "bars_held": position.bars_held,
    }
