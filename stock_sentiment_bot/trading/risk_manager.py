"""
Risk Manager - Circuit breakers and risk controls.
This is the CRITICAL component that prevents bust.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Dict, List

from config.constants import (
    MAX_DAILY_LOSS_PERCENT,
    MAX_DAILY_TRADES,
    WARNING_DRAWDOWN,
    HALT_DRAWDOWN,
    REDUCE_AFTER_LOSSES,
    HALT_AFTER_LOSSES,
    MAX_POSITIONS,
    MAX_SINGLE_POSITION_PERCENT,
    MAX_SECTOR_EXPOSURE_PERCENT,
    DRAWDOWN_RISK_REDUCTION,
    BASE_RISK_PERCENT,
    MAX_RISK_PERCENT,
    MIN_RISK_PERCENT,
)


logger = logging.getLogger("sentiment_bot.risk_manager")


@dataclass
class RiskState:
    """Current risk state tracking."""
    # Account
    starting_equity: float = 0.0
    current_equity: float = 0.0
    high_water_mark: float = 0.0

    # Daily tracking
    daily_starting_equity: float = 0.0
    daily_pnl: float = 0.0
    daily_trades: int = 0

    # Streak tracking
    consecutive_wins: int = 0
    consecutive_losses: int = 0

    # Circuit breaker states
    daily_loss_halt: bool = False
    trade_limit_halt: bool = False
    drawdown_halt: bool = False
    consecutive_loss_halt: bool = False

    # Date tracking for daily reset
    last_trading_date: Optional[date] = None

    @property
    def is_halted(self) -> bool:
        """Check if any circuit breaker is active."""
        return any([
            self.daily_loss_halt,
            self.trade_limit_halt,
            self.drawdown_halt,
            self.consecutive_loss_halt
        ])

    @property
    def halt_reasons(self) -> List[str]:
        """Get list of active halt reasons."""
        reasons = []
        if self.daily_loss_halt:
            reasons.append(f"Daily loss limit ({MAX_DAILY_LOSS_PERCENT*100:.0f}%)")
        if self.trade_limit_halt:
            reasons.append(f"Daily trade limit ({MAX_DAILY_TRADES})")
        if self.drawdown_halt:
            reasons.append(f"Drawdown halt ({HALT_DRAWDOWN*100:.0f}%)")
        if self.consecutive_loss_halt:
            reasons.append(f"Consecutive losses ({HALT_AFTER_LOSSES})")
        return reasons

    @property
    def drawdown(self) -> float:
        """Current drawdown from high water mark."""
        if self.high_water_mark <= 0:
            return 0.0
        return (self.high_water_mark - self.current_equity) / self.high_water_mark

    @property
    def daily_pnl_percent(self) -> float:
        """Today's P&L as percentage."""
        if self.daily_starting_equity <= 0:
            return 0.0
        return (self.current_equity - self.daily_starting_equity) / self.daily_starting_equity


class RiskManager:
    """
    Risk management with hard circuit breakers.

    Circuit Breakers (CANNOT be overridden):
    1. Daily loss limit: Halt at -10% daily
    2. Trade limit: Max 5 trades per day
    3. Drawdown halt: Full stop at -25% from peak
    4. Consecutive losses: Halt after 7 consecutive losses

    Position Limits:
    1. Max 3 concurrent positions
    2. Max 35% in single position
    3. Max 50% in single sector
    """

    def __init__(self, initial_equity: float):
        self.state = RiskState(
            starting_equity=initial_equity,
            current_equity=initial_equity,
            high_water_mark=initial_equity,
            daily_starting_equity=initial_equity,
            last_trading_date=date.today()
        )
        logger.info(f"RiskManager initialized with equity: ${initial_equity:,.2f}")

    def update_equity(self, new_equity: float):
        """Update current equity and check circuit breakers."""
        self.state.current_equity = new_equity

        # Update high water mark
        if new_equity > self.state.high_water_mark:
            self.state.high_water_mark = new_equity
            logger.info(f"New high water mark: ${new_equity:,.2f}")

        # Check all circuit breakers
        self._check_circuit_breakers()

    def record_trade_result(self, pnl: float, is_win: bool):
        """Record trade result and update streaks."""
        # Reset for new day if needed
        self._check_new_day()

        # Update daily stats
        self.state.daily_pnl += pnl
        self.state.daily_trades += 1

        # Update streaks
        if is_win:
            self.state.consecutive_wins += 1
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0

        # Check circuit breakers
        self._check_circuit_breakers()

        logger.info(
            f"Trade recorded: PnL=${pnl:+,.2f}, "
            f"Daily={self.state.daily_trades}, "
            f"Streak={'W' if is_win else 'L'}{max(self.state.consecutive_wins, self.state.consecutive_losses)}"
        )

    def can_trade(self) -> tuple[bool, str]:
        """
        Check if trading is allowed.
        Returns (allowed, reason).
        """
        self._check_new_day()
        self._check_circuit_breakers()

        if self.state.is_halted:
            return False, f"HALTED: {', '.join(self.state.halt_reasons)}"

        return True, "OK"

    def get_risk_multiplier(self) -> float:
        """
        Get risk multiplier based on current state.
        Returns value between 0.0 and 1.0.
        """
        multiplier = 1.0

        # Reduce after consecutive losses
        if self.state.consecutive_losses >= REDUCE_AFTER_LOSSES:
            multiplier *= 0.5
            logger.info(f"Risk reduced 50% due to {self.state.consecutive_losses} consecutive losses")

        # Reduce based on drawdown
        drawdown = self.state.drawdown
        for (low, high), mult in DRAWDOWN_RISK_REDUCTION.items():
            if low <= drawdown < high:
                multiplier *= mult
                if mult < 1.0:
                    logger.info(f"Risk reduced to {mult*100:.0f}% due to {drawdown*100:.1f}% drawdown")
                break

        return multiplier

    def calculate_position_risk(self, base_risk: float = BASE_RISK_PERCENT) -> float:
        """
        Calculate actual risk percent for next trade.
        Applies all risk multipliers and enforces bounds.
        """
        # Apply risk multiplier
        risk = base_risk * self.get_risk_multiplier()

        # Enforce bounds
        risk = max(MIN_RISK_PERCENT, min(MAX_RISK_PERCENT, risk))

        return risk

    def check_position_limits(
        self,
        ticker: str,
        proposed_value: float,
        sector: Optional[str] = None,
        current_positions: Optional[Dict[str, float]] = None
    ) -> tuple[bool, str]:
        """
        Check if proposed position violates limits.
        Returns (allowed, reason).
        """
        current_positions = current_positions or {}
        equity = self.state.current_equity

        # Check position count
        if len(current_positions) >= MAX_POSITIONS and ticker not in current_positions:
            return False, f"Max positions ({MAX_POSITIONS}) reached"

        # Check single position size
        max_position_value = equity * MAX_SINGLE_POSITION_PERCENT
        current_in_ticker = current_positions.get(ticker, 0)
        if current_in_ticker + proposed_value > max_position_value:
            return False, f"Position size exceeds {MAX_SINGLE_POSITION_PERCENT*100:.0f}% limit"

        # Check sector exposure (if sector provided)
        if sector:
            sector_exposure = sum(
                v for t, v in current_positions.items()
                if self._get_sector(t) == sector
            )
            max_sector_value = equity * MAX_SECTOR_EXPOSURE_PERCENT
            if sector_exposure + proposed_value > max_sector_value:
                return False, f"Sector exposure exceeds {MAX_SECTOR_EXPOSURE_PERCENT*100:.0f}% limit"

        return True, "OK"

    def reset_daily(self):
        """Reset daily tracking (call at start of trading day)."""
        self.state.daily_starting_equity = self.state.current_equity
        self.state.daily_pnl = 0.0
        self.state.daily_trades = 0
        self.state.daily_loss_halt = False
        self.state.trade_limit_halt = False
        self.state.last_trading_date = date.today()
        logger.info(f"Daily reset - Starting equity: ${self.state.current_equity:,.2f}")

    def get_status(self) -> Dict:
        """Get current risk status summary."""
        return {
            "equity": self.state.current_equity,
            "high_water_mark": self.state.high_water_mark,
            "drawdown_pct": self.state.drawdown * 100,
            "daily_pnl_pct": self.state.daily_pnl_percent * 100,
            "daily_trades": self.state.daily_trades,
            "consecutive_losses": self.state.consecutive_losses,
            "consecutive_wins": self.state.consecutive_wins,
            "is_halted": self.state.is_halted,
            "halt_reasons": self.state.halt_reasons,
            "risk_multiplier": self.get_risk_multiplier(),
            "effective_risk_pct": self.calculate_position_risk() * 100,
        }

    def _check_circuit_breakers(self):
        """Check and update all circuit breaker states."""
        # Daily loss limit
        if self.state.daily_pnl_percent <= -MAX_DAILY_LOSS_PERCENT:
            if not self.state.daily_loss_halt:
                self.state.daily_loss_halt = True
                logger.warning(f"CIRCUIT BREAKER: Daily loss limit hit ({self.state.daily_pnl_percent*100:.1f}%)")

        # Trade limit
        if self.state.daily_trades >= MAX_DAILY_TRADES:
            if not self.state.trade_limit_halt:
                self.state.trade_limit_halt = True
                logger.warning(f"CIRCUIT BREAKER: Daily trade limit hit ({MAX_DAILY_TRADES})")

        # Drawdown halt
        if self.state.drawdown >= HALT_DRAWDOWN:
            if not self.state.drawdown_halt:
                self.state.drawdown_halt = True
                logger.critical(f"CIRCUIT BREAKER: Drawdown halt ({self.state.drawdown*100:.1f}%)")

        # Consecutive loss halt
        if self.state.consecutive_losses >= HALT_AFTER_LOSSES:
            if not self.state.consecutive_loss_halt:
                self.state.consecutive_loss_halt = True
                logger.critical(f"CIRCUIT BREAKER: Consecutive loss halt ({HALT_AFTER_LOSSES} losses)")

    def _check_new_day(self):
        """Check if it's a new trading day and reset if needed."""
        today = date.today()
        if self.state.last_trading_date != today:
            self.reset_daily()

    def _get_sector(self, ticker: str) -> str:
        """Get sector for ticker (placeholder - implement with real data)."""
        # TODO: Implement real sector lookup
        from config.constants import SECTOR_MAP
        for sector, tickers in SECTOR_MAP.items():
            if ticker in tickers:
                return sector
        return "unknown"


# Singleton for global access
_risk_manager: Optional[RiskManager] = None

def get_risk_manager() -> RiskManager:
    """Get global risk manager instance."""
    global _risk_manager
    if _risk_manager is None:
        raise RuntimeError("RiskManager not initialized. Call init_risk_manager() first.")
    return _risk_manager

def init_risk_manager(initial_equity: float) -> RiskManager:
    """Initialize global risk manager."""
    global _risk_manager
    _risk_manager = RiskManager(initial_equity)
    return _risk_manager
