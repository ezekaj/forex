#!/usr/bin/env python3
"""
ADVANCED RISK MANAGER
Implements professional-grade risk management for forex trading.

Key Features:
1. Dynamic position sizing based on volatility
2. Portfolio heat monitoring
3. Correlation-aware exposure limits
4. Drawdown-based circuit breakers
5. Session and time-based filters
6. Proper Kelly Criterion implementation
"""
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class Position:
    """Represents an open trading position."""
    pair: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    risk_amount: float = 0.0


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    # Per-trade limits
    max_risk_per_trade_pct: float = 0.01  # 1% max per trade
    min_risk_per_trade_pct: float = 0.005  # 0.5% min per trade

    # Portfolio limits
    max_portfolio_heat_pct: float = 0.05  # 5% max total risk
    max_positions: int = 3
    max_positions_per_pair: int = 1
    max_correlated_positions: int = 2

    # Drawdown limits
    max_daily_loss_pct: float = 0.03  # 3% daily loss limit
    max_weekly_loss_pct: float = 0.06  # 6% weekly loss limit
    max_drawdown_pct: float = 0.10  # 10% max drawdown

    # Time filters
    trade_start_hour: int = 8  # UTC
    trade_end_hour: int = 20  # UTC
    avoid_friday_after: int = 18  # UTC

    # Spread filters
    max_spread_pips: float = 3.0


@dataclass
class RiskState:
    """Current risk state."""
    capital: float
    equity: float
    open_positions: List[Position] = field(default_factory=list)
    day_start_equity: float = 0.0
    week_start_equity: float = 0.0
    peak_equity: float = 0.0
    current_drawdown_pct: float = 0.0
    daily_pnl_pct: float = 0.0
    weekly_pnl_pct: float = 0.0
    is_halted: bool = False
    halt_reason: Optional[str] = None


class RiskManager:
    """
    Professional-grade risk manager for forex trading.
    """

    # Correlation matrix for major pairs (approximate values)
    CORRELATIONS = {
        ('EURUSD', 'GBPUSD'): 0.85,
        ('EURUSD', 'AUDUSD'): 0.75,
        ('EURUSD', 'NZDUSD'): 0.70,
        ('EURUSD', 'USDCHF'): -0.95,
        ('EURUSD', 'USDCAD'): -0.55,
        ('GBPUSD', 'AUDUSD'): 0.70,
        ('USDJPY', 'EURJPY'): 0.85,
        ('USDJPY', 'GBPJPY'): 0.80,
        ('AUDUSD', 'NZDUSD'): 0.90,
        ('AUDUSD', 'USDCAD'): -0.65,
    }

    def __init__(
        self,
        initial_capital: float,
        limits: Optional[RiskLimits] = None
    ):
        """
        Initialize risk manager.

        Args:
            initial_capital: Starting capital
            limits: Risk limits configuration
        """
        self.limits = limits or RiskLimits()

        self.state = RiskState(
            capital=initial_capital,
            equity=initial_capital,
            day_start_equity=initial_capital,
            week_start_equity=initial_capital,
            peak_equity=initial_capital
        )

    # =========================================================================
    # POSITION SIZING
    # =========================================================================

    def calculate_position_size(
        self,
        pair: str,
        direction: str,
        entry_price: float,
        stop_loss_price: float,
        confidence: float,
        atr: float,
        pip_value: float = 0.0001
    ) -> Tuple[float, float]:
        """
        Calculate optimal position size using modified Kelly Criterion.

        Args:
            pair: Currency pair
            direction: 'LONG' or 'SHORT'
            entry_price: Proposed entry price
            stop_loss_price: Stop loss price
            confidence: Signal confidence (0-1)
            atr: Current ATR for the pair
            pip_value: Value of 1 pip

        Returns:
            (position_size, risk_amount)
        """
        if self.state.is_halted:
            return 0.0, 0.0

        # Calculate risk per pip
        if direction == 'LONG':
            stop_distance = entry_price - stop_loss_price
        else:
            stop_distance = stop_loss_price - entry_price

        if stop_distance <= 0:
            return 0.0, 0.0

        stop_distance_pips = stop_distance / pip_value

        # Base risk percentage (confidence-adjusted)
        base_risk = self.limits.min_risk_per_trade_pct + \
            (confidence - 0.5) * 2 * \
            (self.limits.max_risk_per_trade_pct - self.limits.min_risk_per_trade_pct)

        base_risk = max(
            self.limits.min_risk_per_trade_pct,
            min(self.limits.max_risk_per_trade_pct, base_risk)
        )

        # Volatility adjustment (reduce size in high volatility)
        typical_atr = 0.0010  # 10 pips for majors
        vol_ratio = atr / typical_atr if typical_atr > 0 else 1.0

        if vol_ratio > 1.5:
            # High volatility - reduce size
            vol_adjustment = 1.0 / vol_ratio
        elif vol_ratio < 0.7:
            # Low volatility - slightly increase size
            vol_adjustment = min(1.3, 1.0 / max(vol_ratio, 0.5))
        else:
            vol_adjustment = 1.0

        adjusted_risk = base_risk * vol_adjustment

        # Portfolio heat check
        current_heat = self._calculate_portfolio_heat()
        remaining_heat = self.limits.max_portfolio_heat_pct - current_heat

        if remaining_heat <= 0:
            return 0.0, 0.0

        # Cap to remaining heat
        adjusted_risk = min(adjusted_risk, remaining_heat)

        # Calculate risk amount
        risk_amount = self.state.equity * adjusted_risk

        # Calculate position size
        # Risk = Position Size * Stop Distance (in value terms)
        # For forex: Position Size = Risk / (Stop Pips * Pip Value per Lot)
        pip_value_per_lot = 10.0  # $10 per pip for standard lot on majors
        position_size = risk_amount / (stop_distance_pips * pip_value_per_lot)

        # Apply minimum size
        min_size = 0.01  # Minimum 0.01 lots (micro lot)
        position_size = max(position_size, min_size)

        return position_size, risk_amount

    def _calculate_portfolio_heat(self) -> float:
        """Calculate current portfolio heat (total risk exposure)."""
        if not self.state.open_positions or self.state.equity <= 0:
            return 0.0

        total_risk = sum(pos.risk_amount for pos in self.state.open_positions)
        return total_risk / self.state.equity

    # =========================================================================
    # TRADE VALIDATION
    # =========================================================================

    def can_open_trade(
        self,
        pair: str,
        direction: str,
        spread_pips: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a new trade can be opened.

        Args:
            pair: Currency pair
            direction: 'LONG' or 'SHORT'
            spread_pips: Current spread

        Returns:
            (can_trade, rejection_reason)
        """
        # Check halt status
        if self.state.is_halted:
            return False, f"Trading halted: {self.state.halt_reason}"

        # Check drawdown limits
        if not self._check_drawdown_limits():
            return False, "Drawdown limit reached"

        # Check position limits
        if len(self.state.open_positions) >= self.limits.max_positions:
            return False, f"Max positions ({self.limits.max_positions}) reached"

        # Check positions per pair
        pair_positions = sum(1 for p in self.state.open_positions if p.pair == pair)
        if pair_positions >= self.limits.max_positions_per_pair:
            return False, f"Max positions for {pair} reached"

        # Check correlation exposure
        if not self._check_correlation_limits(pair, direction):
            return False, "Correlated exposure limit reached"

        # Check spread
        if spread_pips > self.limits.max_spread_pips:
            return False, f"Spread too wide: {spread_pips:.1f} pips"

        # Check trading hours
        if not self._is_trading_hours():
            return False, "Outside trading hours"

        # Check portfolio heat
        current_heat = self._calculate_portfolio_heat()
        if current_heat >= self.limits.max_portfolio_heat_pct:
            return False, "Portfolio heat limit reached"

        return True, None

    def _check_drawdown_limits(self) -> bool:
        """Check if drawdown limits are breached."""
        # Calculate current drawdown
        if self.state.peak_equity > 0:
            self.state.current_drawdown_pct = (
                self.state.peak_equity - self.state.equity
            ) / self.state.peak_equity

        if self.state.current_drawdown_pct >= self.limits.max_drawdown_pct:
            self._halt_trading(f"Max drawdown reached: {self.state.current_drawdown_pct:.1%}")
            return False

        # Check daily loss
        if self.state.day_start_equity > 0:
            self.state.daily_pnl_pct = (
                self.state.equity - self.state.day_start_equity
            ) / self.state.day_start_equity

            if self.state.daily_pnl_pct <= -self.limits.max_daily_loss_pct:
                self._halt_trading(f"Daily loss limit: {self.state.daily_pnl_pct:.1%}")
                return False

        # Check weekly loss
        if self.state.week_start_equity > 0:
            self.state.weekly_pnl_pct = (
                self.state.equity - self.state.week_start_equity
            ) / self.state.week_start_equity

            if self.state.weekly_pnl_pct <= -self.limits.max_weekly_loss_pct:
                self._halt_trading(f"Weekly loss limit: {self.state.weekly_pnl_pct:.1%}")
                return False

        return True

    def _check_correlation_limits(self, pair: str, direction: str) -> bool:
        """Check if adding this position would exceed correlation limits."""
        correlated_count = 0

        for pos in self.state.open_positions:
            if pos.pair == pair:
                continue

            # Check correlation
            corr = self._get_correlation(pair, pos.pair)

            if abs(corr) >= 0.7:  # Significantly correlated
                # Same direction + positive correlation = exposure
                # Opposite direction + negative correlation = exposure
                same_exposure = (
                    (direction == pos.direction and corr > 0) or
                    (direction != pos.direction and corr < 0)
                )

                if same_exposure:
                    correlated_count += 1

        return correlated_count < self.limits.max_correlated_positions

    def _get_correlation(self, pair1: str, pair2: str) -> float:
        """Get correlation between two pairs."""
        # Try direct lookup
        corr = self.CORRELATIONS.get((pair1, pair2))
        if corr is not None:
            return corr

        # Try reverse
        corr = self.CORRELATIONS.get((pair2, pair1))
        if corr is not None:
            return corr

        # Unknown - assume moderate positive correlation
        return 0.3

    def _is_trading_hours(self) -> bool:
        """Check if current time is within trading hours."""
        now = datetime.utcnow()
        current_hour = now.hour

        # Check basic trading hours
        if current_hour < self.limits.trade_start_hour:
            return False
        if current_hour >= self.limits.trade_end_hour:
            return False

        # Check Friday cutoff
        if now.weekday() == 4:  # Friday
            if current_hour >= self.limits.avoid_friday_after:
                return False

        return True

    def _halt_trading(self, reason: str):
        """Halt trading with reason."""
        self.state.is_halted = True
        self.state.halt_reason = reason

    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================

    def register_position(self, position: Position):
        """Register a new open position."""
        self.state.open_positions.append(position)

    def close_position(self, pair: str, realized_pnl: float):
        """Close a position and update state."""
        # Remove position
        self.state.open_positions = [
            p for p in self.state.open_positions if p.pair != pair
        ]

        # Update equity
        self.state.equity += realized_pnl

        # Update peak
        if self.state.equity > self.state.peak_equity:
            self.state.peak_equity = self.state.equity

    def update_equity(self, new_equity: float):
        """Update current equity value."""
        self.state.equity = new_equity

        if new_equity > self.state.peak_equity:
            self.state.peak_equity = new_equity

        # Re-check drawdown limits
        self._check_drawdown_limits()

    def reset_daily(self):
        """Reset daily tracking (call at start of new day)."""
        self.state.day_start_equity = self.state.equity
        self.state.daily_pnl_pct = 0.0

    def reset_weekly(self):
        """Reset weekly tracking (call at start of new week)."""
        self.state.week_start_equity = self.state.equity
        self.state.weekly_pnl_pct = 0.0

    def resume_trading(self):
        """Resume trading after halt (manual override)."""
        self.state.is_halted = False
        self.state.halt_reason = None

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_risk_report(self) -> Dict:
        """Get current risk status report."""
        return {
            'capital': self.state.capital,
            'equity': self.state.equity,
            'pnl': self.state.equity - self.state.capital,
            'pnl_pct': (self.state.equity - self.state.capital) / self.state.capital,
            'open_positions': len(self.state.open_positions),
            'portfolio_heat': self._calculate_portfolio_heat(),
            'current_drawdown': self.state.current_drawdown_pct,
            'daily_pnl': self.state.daily_pnl_pct,
            'weekly_pnl': self.state.weekly_pnl_pct,
            'is_halted': self.state.is_halted,
            'halt_reason': self.state.halt_reason,
            'is_trading_hours': self._is_trading_hours()
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_conservative_risk_manager(capital: float) -> RiskManager:
    """Create a conservative risk manager."""
    limits = RiskLimits(
        max_risk_per_trade_pct=0.01,
        min_risk_per_trade_pct=0.005,
        max_portfolio_heat_pct=0.03,
        max_positions=2,
        max_daily_loss_pct=0.02,
        max_weekly_loss_pct=0.04,
        max_drawdown_pct=0.08
    )
    return RiskManager(capital, limits)


def create_moderate_risk_manager(capital: float) -> RiskManager:
    """Create a moderate risk manager."""
    limits = RiskLimits(
        max_risk_per_trade_pct=0.015,
        min_risk_per_trade_pct=0.005,
        max_portfolio_heat_pct=0.05,
        max_positions=3,
        max_daily_loss_pct=0.03,
        max_weekly_loss_pct=0.06,
        max_drawdown_pct=0.10
    )
    return RiskManager(capital, limits)


def create_aggressive_risk_manager(capital: float) -> RiskManager:
    """Create an aggressive risk manager (NOT recommended)."""
    limits = RiskLimits(
        max_risk_per_trade_pct=0.02,
        min_risk_per_trade_pct=0.01,
        max_portfolio_heat_pct=0.08,
        max_positions=4,
        max_daily_loss_pct=0.05,
        max_weekly_loss_pct=0.10,
        max_drawdown_pct=0.15
    )
    return RiskManager(capital, limits)
