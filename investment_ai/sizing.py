"""
Position Sizing — Quarter-Kelly + Conviction Tiers + Cost Check.

Quarter-Kelly provides the mathematical optimal fraction of capital to risk,
reduced by 4× for safety. Conviction tiers (from LEVEL_UP_STRATEGY) scale
this up or down based on signal quality.

Cost check: don't trade if expected alpha < 3× transaction cost.
"""

from enum import IntEnum

import numpy as np


class ConvictionTier(IntEnum):
    WEAK = 0         # Skip trade
    MODERATE = 1     # 0.5× base size
    STRONG = 2       # 1.0× base size
    VERY_STRONG = 3  # 1.5× base size
    KILLER = 4       # 2.0× base size (capped at 5%)


# Per-asset class transaction costs (per side)
COST_MODEL = {
    "mega_cap_stock": 0.0003,
    "large_cap_stock": 0.0005,
    "mid_cap_stock": 0.0008,
    "small_cap_stock": 0.0015,
    "btc_eth": 0.0005,
    "alt_large_crypto": 0.0010,
    "alt_small_crypto": 0.0015,
    "commodity": 0.0010,
    "index": 0.0005,
}

# Map specific symbols to cost tiers
SYMBOL_COST_TIER = {
    # Mega-cap stocks
    "AAPL": "mega_cap_stock", "MSFT": "mega_cap_stock", "AMZN": "mega_cap_stock",
    "GOOGL": "mega_cap_stock", "META": "mega_cap_stock",
    # Large-cap stocks
    "NVDA": "large_cap_stock", "TSLA": "large_cap_stock", "AVGO": "large_cap_stock",
    "JPM": "large_cap_stock", "V": "large_cap_stock", "MA": "large_cap_stock",
    "JNJ": "large_cap_stock", "UNH": "large_cap_stock", "LLY": "large_cap_stock",
    "WMT": "large_cap_stock", "PG": "large_cap_stock", "KO": "large_cap_stock",
    "XOM": "large_cap_stock", "CVX": "large_cap_stock", "NFLX": "large_cap_stock",
    "ORCL": "large_cap_stock", "CRM": "large_cap_stock", "AMD": "large_cap_stock",
    "INTC": "large_cap_stock", "BAC": "large_cap_stock", "GS": "large_cap_stock",
    "MS": "large_cap_stock", "DIS": "large_cap_stock", "BA": "large_cap_stock",
    "CAT": "large_cap_stock", "GE": "large_cap_stock", "BRK-B": "large_cap_stock",
    "COST": "large_cap_stock", "ABBV": "large_cap_stock",
    # Mid-cap stocks
    "PLTR": "mid_cap_stock", "COIN": "mid_cap_stock", "HOOD": "mid_cap_stock",
    "UBER": "mid_cap_stock", "SHOP": "mid_cap_stock", "SQ": "mid_cap_stock",
    # Small-cap stocks
    "NIO": "small_cap_stock", "SOFI": "small_cap_stock", "RIVN": "small_cap_stock",
    "F": "small_cap_stock", "GM": "small_cap_stock", "SNAP": "small_cap_stock",
    "NKE": "small_cap_stock", "TGT": "small_cap_stock", "MCD": "small_cap_stock",
    "PFE": "small_cap_stock",
    # Crypto
    "BTC-USD": "btc_eth", "ETH-USD": "btc_eth",
    "SOL-USD": "alt_large_crypto", "XRP-USD": "alt_large_crypto",
    "ADA-USD": "alt_large_crypto", "DOGE-USD": "alt_large_crypto",
    "AVAX-USD": "alt_large_crypto", "LINK-USD": "alt_large_crypto",
    "DOT-USD": "alt_large_crypto", "MATIC-USD": "alt_large_crypto",
    "UNI-USD": "alt_small_crypto", "ATOM-USD": "alt_small_crypto",
    "FIL-USD": "alt_small_crypto", "APT-USD": "alt_small_crypto",
    "ARB-USD": "alt_small_crypto", "OP-USD": "alt_small_crypto",
    "NEAR-USD": "alt_small_crypto", "SUI-USD": "alt_small_crypto",
    "PEPE-USD": "alt_small_crypto", "RENDER-USD": "alt_small_crypto",
    # Commodities
    "GC=F": "commodity", "CL=F": "commodity",
    "XAUUSD": "commodity", "XAGUSD": "commodity",
    "BRENTCMDUSD": "commodity", "LIGHTCMDUSD": "commodity",
    # Indices
    "^FTSE": "index", "^N225": "index",
    "USA500IDXUSD": "index", "USATECHIDXUSD": "index", "USA30IDXUSD": "index",
}


def get_cost_per_side(symbol: str) -> float:
    """Get transaction cost per side for a given symbol."""
    tier = SYMBOL_COST_TIER.get(symbol, "mid_cap_stock")
    return COST_MODEL.get(tier, 0.001)


def classify_conviction(
    model_prob: float,
    signal_consensus: float,
    regime: str,
    hurst: float,
    n_agreeing_models: int,
) -> ConvictionTier:
    """
    Classify trade conviction based on model + strategy + regime alignment.

    Args:
        model_prob: Calibrated probability of +1 (profit target hit).
        signal_consensus: Sum of strategy signals (-3 to +3).
        regime: 'trending', 'ranging', 'volatile'.
        hurst: Hurst exponent (0-1).
        n_agreeing_models: How many of 3 ensemble models agree on direction.

    Returns:
        ConvictionTier (WEAK → KILLER).
    """
    # Base confidence from probability (0 = coin flip, 1 = certain)
    confidence = abs(model_prob - 0.5) * 2

    # Model agreement bonus
    if n_agreeing_models >= 3:
        confidence *= 1.3
    elif n_agreeing_models == 2:
        confidence *= 1.0
    else:
        confidence *= 0.5

    # Strategy signal confirmation
    direction = 1 if model_prob > 0.5 else -1
    if signal_consensus != 0:
        if np.sign(signal_consensus) == direction:
            confidence *= 1.2  # ML + rules agree
        else:
            confidence *= 0.6  # ML + rules disagree

    # Regime alignment
    if regime == "trending" and hurst > 0.55:
        confidence *= 1.1
    elif regime == "ranging" and hurst < 0.45:
        confidence *= 1.1
    elif regime == "volatile":
        confidence *= 0.5

    # Map to tiers
    if confidence < 0.10:
        return ConvictionTier.WEAK
    elif confidence < 0.25:
        return ConvictionTier.MODERATE
    elif confidence < 0.45:
        return ConvictionTier.STRONG
    elif confidence < 0.65:
        return ConvictionTier.VERY_STRONG
    else:
        return ConvictionTier.KILLER


def compute_position_size(
    model_prob: float,
    conviction: ConvictionTier,
    regime: str,
    asset_class: str,
    direction: int,
    avg_win_loss_ratio: float = 1.5,
    max_position: float = 0.05,
) -> float:
    """
    Compute position size as fraction of capital using quarter-Kelly.

    Args:
        model_prob: Calibrated win probability.
        conviction: ConvictionTier from classify_conviction.
        regime: Market regime.
        asset_class: 'stock', 'crypto', 'commodity', 'index'.
        direction: +1 long, -1 short.
        avg_win_loss_ratio: Historical avg_win / avg_loss (per regime).
        max_position: Maximum position size as fraction of capital.

    Returns:
        Position size (0.0 to max_position).
    """
    if conviction == ConvictionTier.WEAK:
        return 0.0

    # Effective win probability
    p = max(model_prob, 1 - model_prob)
    q = 1 - p
    b = avg_win_loss_ratio

    # Kelly criterion
    kelly = (p * b - q) / b
    if kelly <= 0:
        return 0.0

    # Quarter-Kelly for safety
    base_size = kelly / 4

    # Conviction multiplier
    conviction_mult = {
        ConvictionTier.MODERATE: 0.5,
        ConvictionTier.STRONG: 1.0,
        ConvictionTier.VERY_STRONG: 1.5,
        ConvictionTier.KILLER: 2.0,
    }
    base_size *= conviction_mult.get(conviction, 1.0)

    # Volatile regime → halve
    if regime == "volatile":
        base_size *= 0.5

    # Short stock penalty (stocks drift up naturally)
    if asset_class == "stock" and direction == -1:
        base_size *= 0.7

    # Cap
    return min(base_size, max_position)


def passes_cost_check(
    model_prob: float,
    atr_pct: float,
    symbol: str,
    cost_multiplier: float = 3.0,
) -> bool:
    """
    Check if expected alpha exceeds transaction costs.

    Don't trade if the expected move (from model probability × ATR)
    is less than cost_multiplier × round-trip costs.

    Args:
        model_prob: Calibrated probability.
        atr_pct: Current ATR as percentage of price.
        symbol: Asset symbol for cost lookup.
        cost_multiplier: Required alpha/cost ratio (3.0 = need 3× cost).

    Returns:
        True if trade is worth taking after costs.
    """
    cost_per_side = get_cost_per_side(symbol)
    round_trip_cost = cost_per_side * 2

    # Expected move: probability edge × ATR
    edge = abs(model_prob - 0.5) * 2  # 0 = no edge, 1 = certain
    expected_move = edge * atr_pct

    return expected_move >= round_trip_cost * cost_multiplier
