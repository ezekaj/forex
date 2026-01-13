"""
Pattern detection for market analysis.
Identifies common chart patterns and price movements.
"""

from typing import List
import numpy as np


def detect_market_patterns(prices: List[float]) -> List[str]:
    """
    Detect market patterns from price data.
    Returns list of identified patterns.
    """
    if len(prices) < 10:
        return []

    patterns = []

    # Convert to numpy for efficient calculation
    prices_arr = np.array(prices)

    # 1. Trend Detection
    if _is_uptrend(prices_arr):
        patterns.append("Uptrend")
    elif _is_downtrend(prices_arr):
        patterns.append("Downtrend")
    else:
        patterns.append("Sideways")

    # 2. Higher Highs / Lower Lows
    if _has_higher_highs(prices_arr):
        patterns.append("Higher Highs")
    if _has_lower_lows(prices_arr):
        patterns.append("Lower Lows")

    # 3. Support and Resistance
    support, resistance = _find_support_resistance(prices_arr)
    if support:
        patterns.append(f"Support @ {support:.2f}")
    if resistance:
        patterns.append(f"Resistance @ {resistance:.2f}")

    # 4. Breakout Detection
    if _is_breakout(prices_arr):
        patterns.append("Breakout")

    # 5. Consolidation
    if _is_consolidating(prices_arr):
        patterns.append("Consolidating")

    # 6. Volatility patterns
    volatility = _calculate_volatility(prices_arr)
    if volatility > 0.02:
        patterns.append("High Volatility")
    elif volatility < 0.005:
        patterns.append("Low Volatility")

    return patterns


def _is_uptrend(prices: np.ndarray, lookback: int = 20) -> bool:
    """Check if price is in uptrend"""
    if len(prices) < lookback:
        return False

    recent = prices[-lookback:]

    # Linear regression to determine trend
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0]

    return slope > 0


def _is_downtrend(prices: np.ndarray, lookback: int = 20) -> bool:
    """Check if price is in downtrend"""
    if len(prices) < lookback:
        return False

    recent = prices[-lookback:]
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0]

    return slope < 0


def _has_higher_highs(prices: np.ndarray, lookback: int = 20) -> bool:
    """Check for higher highs pattern (bullish)"""
    if len(prices) < lookback:
        return False

    recent = prices[-lookback:]

    # Find local highs
    highs = []
    for i in range(2, len(recent) - 2):
        if recent[i] > recent[i-1] and recent[i] > recent[i+1]:
            highs.append(recent[i])

    # Check if highs are increasing
    if len(highs) >= 2:
        return highs[-1] > highs[0]

    return False


def _has_lower_lows(prices: np.ndarray, lookback: int = 20) -> bool:
    """Check for lower lows pattern (bearish)"""
    if len(prices) < lookback:
        return False

    recent = prices[-lookback:]

    # Find local lows
    lows = []
    for i in range(2, len(recent) - 2):
        if recent[i] < recent[i-1] and recent[i] < recent[i+1]:
            lows.append(recent[i])

    # Check if lows are decreasing
    if len(lows) >= 2:
        return lows[-1] < lows[0]

    return False


def _find_support_resistance(prices: np.ndarray, lookback: int = 50) -> tuple:
    """Find support and resistance levels"""
    if len(prices) < lookback:
        return None, None

    recent = prices[-lookback:]

    # Support: price level that acts as a floor
    # Find lowest price in recent history and check if price bounced off it multiple times
    min_price = recent.min()
    tolerance = (recent.max() - recent.min()) * 0.01  # 1% tolerance

    touches = sum(1 for p in recent if abs(p - min_price) < tolerance)
    support = min_price if touches >= 3 else None

    # Resistance: price level that acts as a ceiling
    max_price = recent.max()
    touches = sum(1 for p in recent if abs(p - max_price) < tolerance)
    resistance = max_price if touches >= 3 else None

    return support, resistance


def _is_breakout(prices: np.ndarray, lookback: int = 20) -> bool:
    """Detect if price is breaking out of a range"""
    if len(prices) < lookback + 5:
        return False

    # Compare recent price to historical range
    historical = prices[-(lookback + 5):-5]
    recent = prices[-5:]

    hist_high = historical.max()
    hist_low = historical.min()
    hist_range = hist_high - hist_low

    # Breakout if recent prices exceed historical range
    recent_max = recent.max()
    recent_min = recent.min()

    breakout_threshold = hist_range * 0.02  # 2% beyond range

    return (recent_max > hist_high + breakout_threshold) or \
           (recent_min < hist_low - breakout_threshold)


def _is_consolidating(prices: np.ndarray, lookback: int = 20) -> bool:
    """Check if price is consolidating (low volatility, tight range)"""
    if len(prices) < lookback:
        return False

    recent = prices[-lookback:]

    # Calculate price range relative to average price
    price_range = recent.max() - recent.min()
    avg_price = recent.mean()
    range_percent = (price_range / avg_price) * 100

    # Consolidation if range is less than 2% of price
    return range_percent < 2.0


def _calculate_volatility(prices: np.ndarray, lookback: int = 20) -> float:
    """Calculate historical volatility (standard deviation of returns)"""
    if len(prices) < lookback + 1:
        return 0.0

    recent = prices[-lookback:]

    # Calculate returns
    returns = np.diff(recent) / recent[:-1]

    # Standard deviation of returns
    volatility = returns.std()

    return volatility


def detect_candlestick_patterns(candles: List) -> List[str]:
    """
    Detect candlestick patterns.
    Requires access to Candle objects with open, high, low, close.
    """
    if len(candles) < 3:
        return []

    patterns = []

    # Get last 3 candles
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]

    # Doji (indecision)
    if abs(c3.close - c3.open) < (c3.high - c3.low) * 0.1:
        patterns.append("Doji")

    # Hammer (bullish reversal)
    body = abs(c3.close - c3.open)
    lower_shadow = min(c3.open, c3.close) - c3.low
    upper_shadow = c3.high - max(c3.open, c3.close)

    if lower_shadow > body * 2 and upper_shadow < body:
        patterns.append("Hammer")

    # Shooting Star (bearish reversal)
    if upper_shadow > body * 2 and lower_shadow < body:
        patterns.append("Shooting Star")

    # Engulfing patterns
    if c2.close < c2.open and c3.close > c3.open:  # Previous red, current green
        if c3.close > c2.open and c3.open < c2.close:
            patterns.append("Bullish Engulfing")

    if c2.close > c2.open and c3.close < c3.open:  # Previous green, current red
        if c3.close < c2.open and c3.open > c2.close:
            patterns.append("Bearish Engulfing")

    return patterns
