"""
Operational safety patterns extracted from NautilusTrader.

- SlidingWindowThrottler: rate limiting for API calls
- CircuitBreaker: 3-state (ACTIVE/REDUCING/HALTED) trading control
- BalanceTracker: track buying power consumed by pending orders
- FillReconciler: handle missed events and state recovery
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class TradingState(Enum):
    ACTIVE = "ACTIVE"        # normal trading
    REDUCING = "REDUCING"    # scale down positions, no new entries
    HALTED = "HALTED"        # no trading at all, emergency


class SlidingWindowThrottler:
    """
    Rate limiter using sliding window.
    Prevents API rate limit errors (429) from Alpaca or other brokers.

    Usage:
        throttler = SlidingWindowThrottler(max_requests=200, window_seconds=60)
        if throttler.can_proceed():
            # make API call
            throttler.record()
        else:
            time.sleep(throttler.wait_time())
    """

    def __init__(self, max_requests: int = 200, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window = window_seconds
        self.timestamps: deque = deque()

    def _clean(self):
        now = time.monotonic()
        while self.timestamps and (now - self.timestamps[0]) > self.window:
            self.timestamps.popleft()

    def can_proceed(self) -> bool:
        self._clean()
        return len(self.timestamps) < self.max_requests

    def record(self):
        self.timestamps.append(time.monotonic())

    def wait_time(self) -> float:
        if self.can_proceed():
            return 0.0
        oldest = self.timestamps[0]
        return max(0, self.window - (time.monotonic() - oldest))

    def throttle(self):
        """Block until a slot is available."""
        while not self.can_proceed():
            time.sleep(0.1)
        self.record()


@dataclass
class CircuitBreaker:
    """
    Three-state circuit breaker for portfolio-level risk management.

    ACTIVE → REDUCING: when drawdown exceeds reduce_threshold
    REDUCING → HALTED: when drawdown exceeds halt_threshold
    HALTED → ACTIVE: when drawdown recovers below recovery_threshold

    In REDUCING state: only close positions, no new entries.
    In HALTED state: close all positions immediately.
    """

    reduce_threshold: float = 0.08    # -8% drawdown → start reducing
    halt_threshold: float = 0.12      # -12% drawdown → halt everything
    recovery_threshold: float = 0.06  # recover to -6% → resume

    state: TradingState = TradingState.ACTIVE
    peak_value: float = 0.0
    current_drawdown: float = 0.0

    def update(self, portfolio_value: float) -> TradingState:
        """Update state based on current portfolio value."""
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        else:
            self.current_drawdown = 0.0

        prev_state = self.state

        if self.state == TradingState.ACTIVE:
            if self.current_drawdown >= self.halt_threshold:
                self.state = TradingState.HALTED
            elif self.current_drawdown >= self.reduce_threshold:
                self.state = TradingState.REDUCING

        elif self.state == TradingState.REDUCING:
            if self.current_drawdown >= self.halt_threshold:
                self.state = TradingState.HALTED
            elif self.current_drawdown <= self.recovery_threshold:
                self.state = TradingState.ACTIVE

        elif self.state == TradingState.HALTED:
            if self.current_drawdown <= self.recovery_threshold:
                self.state = TradingState.ACTIVE

        if self.state != prev_state:
            print(f"[CIRCUIT BREAKER] {prev_state.value} → {self.state.value} "
                  f"(drawdown: {self.current_drawdown:.1%}, peak: ${self.peak_value:,.0f})")

        return self.state

    def can_open_new(self) -> bool:
        return self.state == TradingState.ACTIVE

    def should_reduce(self) -> bool:
        return self.state in (TradingState.REDUCING, TradingState.HALTED)

    def should_close_all(self) -> bool:
        return self.state == TradingState.HALTED


@dataclass
class BalanceTracker:
    """
    Track buying power consumed by pending orders.

    Alpaca doesn't always surface locked balance clearly.
    This tracks our own estimate of available buying power.
    """

    total_equity: float = 0.0
    cash: float = 0.0
    pending_buy_value: float = 0.0

    @property
    def available_buying_power(self) -> float:
        return max(0, self.cash - self.pending_buy_value)

    def lock(self, order_value: float):
        """Lock buying power for a pending buy order."""
        self.pending_buy_value += order_value

    def release(self, order_value: float):
        """Release buying power when order fills or cancels."""
        self.pending_buy_value = max(0, self.pending_buy_value - order_value)

    def can_afford(self, order_value: float) -> bool:
        return self.available_buying_power >= order_value

    def sync(self, cash: float, equity: float):
        """Sync with actual broker balances."""
        self.cash = cash
        self.total_equity = equity
        # Don't reset pending — those are our tracking


class FillReconciler:
    """
    Handle missed events and state recovery after restart.

    On startup:
      1. Query broker for actual positions
      2. Compare to our last known state (SQLite)
      3. If mismatch → log warning, update to broker truth
      4. Any pending orders from before restart → cancel and re-evaluate
    """

    def __init__(self, db_holdings: dict, broker_positions: dict):
        self.db_holdings = db_holdings      # {symbol: shares} from our DB
        self.broker_positions = broker_positions  # {symbol: shares} from broker API
        self.mismatches = []

    def reconcile(self) -> dict:
        """
        Compare DB state vs broker state.
        Returns the reconciled holdings (broker is truth).
        """
        all_symbols = set(self.db_holdings.keys()) | set(self.broker_positions.keys())
        reconciled = {}

        for sym in all_symbols:
            db_shares = self.db_holdings.get(sym, 0)
            broker_shares = self.broker_positions.get(sym, 0)

            if abs(db_shares - broker_shares) > 0.001:
                self.mismatches.append({
                    "symbol": sym,
                    "db_shares": db_shares,
                    "broker_shares": broker_shares,
                    "diff": broker_shares - db_shares,
                })

            # Broker is always truth
            if broker_shares > 0.001:
                reconciled[sym] = broker_shares

        if self.mismatches:
            print(f"[RECONCILIATION] {len(self.mismatches)} mismatches found:")
            for m in self.mismatches:
                print(f"  {m['symbol']}: DB={m['db_shares']:.4f}, Broker={m['broker_shares']:.4f}, Diff={m['diff']:+.4f}")

        return reconciled

    def has_mismatches(self) -> bool:
        return len(self.mismatches) > 0
