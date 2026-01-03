"""
Backtesting engine for realistic trading simulation.

Handles order execution, position tracking, and cost modeling.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from ..strategies.base import BaseStrategy, Signal
from ..risk import RiskProfile


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    direction: int  # 1 for long, -1 for short
    size: float
    profit_loss: float
    profit_loss_pct: float
    costs: float


@dataclass
class Position:
    """Represents an open position."""
    entry_time: datetime
    entry_price: float
    direction: int  # 1 for long, -1 for short
    size: float


class BacktestEngine:
    """
    Backtesting engine for strategy validation.

    Features:
    - Realistic cost modeling (spread, slippage, commission)
    - Position tracking with proper accounting
    - Time series integrity (no look-ahead bias)
    - Comprehensive performance metrics
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        spread_pips: float = 1.0,
        slippage_pips: float = 0.5,
        commission_pct: float = 0.0001,  # 0.01%
        position_size_pct: float = 0.1,  # 10% of capital per trade
        pip_value: float = 0.0001  # For EURUSD
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital
            spread_pips: Bid-ask spread in pips
            slippage_pips: Execution slippage in pips
            commission_pct: Commission as percentage of trade value
            position_size_pct: Position size as percentage of capital
            pip_value: Value of one pip (0.0001 for most pairs)
        """
        self.initial_capital = initial_capital
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.commission_pct = commission_pct
        self.position_size_pct = position_size_pct
        self.pip_value = pip_value

        # State tracking
        self.capital = initial_capital
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.timestamps: List[datetime] = []

    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        features: pd.DataFrame,
        verbose: bool = True,
        risk_profile: Optional[RiskProfile] = None
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data.

        Args:
            strategy: Trained trading strategy
            data: OHLCV data with 'close' column
            features: Feature DataFrame aligned with data
            verbose: Print progress updates

        Returns:
            Dictionary with backtest results and metrics
        """
        if not strategy.is_trained:
            raise ValueError("Strategy must be trained before backtesting")

        if len(data) != len(features):
            raise ValueError("Data and features must have same length")

        # Reset state
        self.capital = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.timestamps = []
        self.risk_profile = risk_profile
        self.halted_reason = None
        self.peak_equity = self.initial_capital
        self.day_start_equity = self.initial_capital
        self.week_start_equity = self.initial_capital
        self.current_day_key = None
        self.current_week_key = None

        if verbose:
            print(f"Starting backtest with ${self.initial_capital:,.2f}")
            print(f"Spread: {self.spread_pips} pips, Slippage: {self.slippage_pips} pips")
            print(f"Position size: {self.position_size_pct * 100}% of capital")
            print(f"Total bars: {len(data)}")
            print("-" * 80)

        # Generate predictions for all data at once
        predictions = strategy.predict(features)

        # Simulate trading bar by bar
        for i in range(len(data)):
            if self.halted_reason:
                if verbose:
                    print(f"Risk halt triggered: {self.halted_reason}")
                break

            timestamp = data.index[i] if hasattr(data.index[i], 'to_pydatetime') else i
            close_price = data.iloc[i]['close']
            signal = predictions[i]

            # Process signal
            self._process_signal(signal, close_price, timestamp)

            # Track equity
            current_equity = self._calculate_equity(close_price)
            self.equity_curve.append(current_equity)
            self.timestamps.append(timestamp)

            # Risk monitoring and circuit breakers
            self._update_risk_limits(timestamp, current_equity)

            if verbose and (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(data)} bars, Equity: ${current_equity:,.2f}")

        # Close any open position at end
        if self.position:
            final_price = data.iloc[-1]['close']
            final_time = data.index[-1] if hasattr(data.index[-1], 'to_pydatetime') else len(data) - 1
            self._close_position(final_price, final_time)

        if verbose:
            print("-" * 80)
            print(f"Backtest complete. Trades executed: {len(self.trades)}")
            print(f"Final equity: ${self.equity_curve[-1]:,.2f}")
            print(f"Return: {(self.equity_curve[-1] / self.initial_capital - 1) * 100:.2f}%")

        # Calculate metrics
        from .metrics import BacktestMetrics
        metrics = BacktestMetrics(
            trades=self.trades,
            equity_curve=self.equity_curve,
            initial_capital=self.initial_capital
        )

        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'timestamps': self.timestamps,
            'metrics': metrics.calculate_all(),
            'strategy': strategy.name,
            'final_capital': self.equity_curve[-1]
        }

    def _process_signal(self, signal: int, price: float, timestamp) -> None:
        """Process trading signal and manage positions."""
        if self.halted_reason:
            return

        if self.position is None:
            # No position - open new position if signal is BUY or SELL
            if signal == Signal.BUY.value:
                self._open_position(price, timestamp, direction=1)
            elif signal == Signal.SELL.value:
                self._open_position(price, timestamp, direction=-1)
        else:
            # Have position - close if opposite signal or HOLD with exit logic
            should_close = False

            if self.position.direction == 1 and signal == Signal.SELL.value:
                should_close = True  # Long position, SELL signal
            elif self.position.direction == -1 and signal == Signal.BUY.value:
                should_close = True  # Short position, BUY signal

            if should_close:
                self._close_position(price, timestamp)

    def _open_position(self, price: float, timestamp, direction: int) -> None:
        """Open a new position."""
        if self.halted_reason:
            return

        # Calculate position size
        position_pct = self.position_size_pct
        if self.risk_profile:
            position_pct = min(
                position_pct,
                self.risk_profile.max_position_pct,
                max(self.risk_profile.risk_per_trade_pct, 0.0001)
            )

        if position_pct <= 0:
            return

        position_value = self.capital * position_pct
        size = position_value / price

        # Apply spread and slippage
        entry_price = self._apply_costs(price, direction, is_entry=True)

        # Commission
        commission = position_value * self.commission_pct
        self.capital -= commission

        self.position = Position(
            entry_time=timestamp,
            entry_price=entry_price,
            direction=direction,
            size=size
        )

    def _close_position(self, price: float, timestamp) -> None:
        """Close the open position."""
        if not self.position:
            return

        # Apply spread and slippage
        exit_price = self._apply_costs(price, self.position.direction, is_entry=False)

        # Calculate P&L
        price_diff = exit_price - self.position.entry_price
        profit_loss = price_diff * self.position.direction * self.position.size
        profit_loss_pct = (price_diff / self.position.entry_price) * self.position.direction

        # Commission
        position_value = self.position.size * price
        commission = position_value * self.commission_pct
        costs = commission

        # Update capital
        self.capital += profit_loss - commission

        # Record trade
        trade = Trade(
            entry_time=self.position.entry_time,
            exit_time=timestamp,
            entry_price=self.position.entry_price,
            exit_price=exit_price,
            direction=self.position.direction,
            size=self.position.size,
            profit_loss=profit_loss,
            profit_loss_pct=profit_loss_pct,
            costs=costs
        )
        self.trades.append(trade)

        self.position = None

    def _apply_costs(self, price: float, direction: int, is_entry: bool) -> float:
        """Apply spread and slippage costs."""
        spread_cost = self.spread_pips * self.pip_value
        slippage_cost = self.slippage_pips * self.pip_value

        total_cost = spread_cost + slippage_cost

        if is_entry:
            # Buying: pay spread + slippage
            if direction == 1:  # Long
                return price + total_cost
            else:  # Short
                return price - total_cost
        else:
            # Selling: pay spread + slippage
            if direction == 1:  # Closing long
                return price - total_cost
            else:  # Closing short
                return price + total_cost

    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity including open position."""
        if not self.position:
            return self.capital

        # Calculate unrealized P&L
        current_value = current_price - self.position.entry_price
        unrealized_pl = current_value * self.position.direction * self.position.size

        return self.capital + unrealized_pl

    def _update_risk_limits(self, timestamp, equity: float) -> None:
        """Enforce drawdown and period loss limits."""
        if not self.risk_profile:
            return

        # Track peak for drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity

        drawdown_pct = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if drawdown_pct >= self.risk_profile.max_drawdown_pct:
            self.halted_reason = (
                f"Max drawdown reached: {drawdown_pct*100:.2f}% >= {self.risk_profile.max_drawdown_pct*100:.2f}%"
            )
            return

        # Period loss checks only if timestamps are datetime
        if isinstance(timestamp, datetime):
            day_key = timestamp.date()
            week_key = (timestamp.isocalendar().year, timestamp.isocalendar().week)

            if self.current_day_key != day_key:
                self.current_day_key = day_key
                self.day_start_equity = equity

            if self.current_week_key != week_key:
                self.current_week_key = week_key
                self.week_start_equity = equity

            daily_loss_pct = (self.day_start_equity - equity) / self.day_start_equity if self.day_start_equity > 0 else 0.0
            weekly_loss_pct = (self.week_start_equity - equity) / self.week_start_equity if self.week_start_equity > 0 else 0.0

            if daily_loss_pct >= self.risk_profile.max_daily_loss_pct:
                self.halted_reason = (
                    f"Daily loss limit hit: {daily_loss_pct*100:.2f}% >= {self.risk_profile.max_daily_loss_pct*100:.2f}%"
                )
                return

            if weekly_loss_pct >= self.risk_profile.max_weekly_loss_pct:
                self.halted_reason = (
                    f"Weekly loss limit hit: {weekly_loss_pct*100:.2f}% >= {self.risk_profile.max_weekly_loss_pct*100:.2f}%"
                )
