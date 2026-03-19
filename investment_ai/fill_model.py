"""
Realistic fill models extracted from NautilusTrader.

VolumeSensitiveFillModel: limits fills to 25% of bar volume at best price,
remainder fills 1 tick worse. Makes backtests more realistic.

Source: nautilus_trader/backtest/models/fill.pyx lines 776-859
"""

import numpy as np


def volume_sensitive_fill(
    order_shares: float,
    order_side: str,       # "BUY" or "SELL"
    bar_volume: float,
    bar_close: float,
    bar_high: float,
    bar_low: float,
    fill_fraction: float = 0.25,  # max 25% of bar volume at best price
    tick_size: float = 0.01,
) -> dict:
    """
    Simulate a realistic fill considering volume constraints.

    For a BUY order:
      - Up to fill_fraction × bar_volume fills at bar_close
      - Remainder fills at bar_close + tick_size (worse price)

    For a SELL order:
      - Up to fill_fraction × bar_volume fills at bar_close
      - Remainder fills at bar_close - tick_size

    Returns:
        dict with avg_fill_price, slippage, filled_shares
    """
    if bar_volume <= 0 or order_shares <= 0:
        return {"avg_fill_price": bar_close, "slippage": 0, "filled_shares": order_shares}

    max_at_best = bar_volume * fill_fraction

    if order_shares <= max_at_best:
        # Full fill at close price
        return {
            "avg_fill_price": bar_close,
            "slippage": 0.0,
            "filled_shares": order_shares,
        }
    else:
        # Partial at close, remainder at worse price
        shares_at_best = max_at_best
        shares_at_worse = order_shares - shares_at_best

        if order_side == "BUY":
            worse_price = min(bar_close + tick_size, bar_high)
        else:
            worse_price = max(bar_close - tick_size, bar_low)

        avg_price = (shares_at_best * bar_close + shares_at_worse * worse_price) / order_shares
        slippage = abs(avg_price - bar_close) / bar_close

        return {
            "avg_fill_price": round(avg_price, 4),
            "slippage": round(slippage, 6),
            "filled_shares": order_shares,
        }


def compute_backtest_fill(
    order_shares: float,
    order_side: str,
    bar: dict,  # {open, high, low, close, volume}
    cost_per_side: float = 0.0005,
) -> dict:
    """
    Complete backtest fill computation with volume sensitivity + transaction costs.

    Returns:
        dict with fill_price, cost, slippage, total_cost_pct
    """
    fill = volume_sensitive_fill(
        order_shares=order_shares,
        order_side=order_side,
        bar_volume=bar.get("volume", 1e9),
        bar_close=bar["close"],
        bar_high=bar["high"],
        bar_low=bar["low"],
    )

    fill_price = fill["avg_fill_price"]
    order_value = order_shares * fill_price
    commission = order_value * cost_per_side
    slippage_cost = fill["slippage"] * order_value

    return {
        "fill_price": fill_price,
        "slippage": fill["slippage"],
        "commission": commission,
        "total_cost": commission + slippage_cost,
        "total_cost_pct": (commission + slippage_cost) / order_value if order_value > 0 else 0,
    }
