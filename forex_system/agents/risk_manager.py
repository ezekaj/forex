"""Risk Manager Agent — position limits, circuit breakers, diversification checks."""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from forex_system.agents.fund_manager import Decision
from forex_system.training.config import TrainingConfig, get_asset

log = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    max_position_pct: float = 0.05        # 5% max per position
    max_portfolio_exposure: float = 0.40   # 40% total invested
    max_sector_pct: float = 0.25           # 25% max per sector
    max_asset_class_pct: float = 0.40      # 40% max per asset class
    max_concurrent_positions: int = 20     # max open positions
    min_confidence: int = 60               # minimum confidence to trade
    stop_loss_pct: float = 0.03            # 3% stop loss
    take_profit_pct: float = 0.06          # 6% take profit
    max_holding_days: int = 7              # max days to hold a position
    daily_loss_warn: float = -0.01         # -1% daily P&L: warning
    daily_loss_throttle: float = -0.02     # -2% daily P&L: throttle (close-only)
    daily_loss_halt: float = -0.03         # -3% daily P&L: halt all
    cumulative_loss_halt: float = -0.10    # -10% total: full stop


class RiskManager:
    """
    Enforces position limits, circuit breakers, and diversification.
    Pure code — no LLM calls. Fast and deterministic.
    """

    def __init__(self, limits: RiskLimits = None):
        self.limits = limits or RiskLimits()
        self._circuit_breaker_state = "ok"
        self._daily_pnl = 0.0
        self._cumulative_pnl = 0.0

    def approve(self, decision: Decision, portfolio: dict) -> bool:
        """
        Check if a trade should be allowed.

        Args:
            decision: the Fund Manager's decision
            portfolio: current portfolio state {
                "positions": {symbol: {direction, entry_price, size_pct, sector, asset_class}},
                "total_exposure": float,
                "daily_pnl": float,
                "cumulative_pnl": float,
            }

        Returns:
            True if trade is approved, False if rejected.
        """
        # Check circuit breakers first
        cb_state = self.check_circuit_breakers(
            portfolio.get("daily_pnl", 0),
            portfolio.get("cumulative_pnl", 0),
        )
        if cb_state in ("throttle", "halt"):
            log.warning(f"RISK: Trade REJECTED — circuit breaker: {cb_state}")
            return False

        # Check minimum confidence
        if decision.confidence < self.limits.min_confidence:
            log.info(f"RISK: {decision.symbol} REJECTED — confidence {decision.confidence} < {self.limits.min_confidence}")
            return False

        # Check maximum concurrent positions
        positions = portfolio.get("positions", {})
        if len(positions) >= self.limits.max_concurrent_positions:
            log.warning(f"RISK: {decision.symbol} REJECTED — max positions ({self.limits.max_concurrent_positions})")
            return False

        # Check total portfolio exposure
        total_exposure = portfolio.get("total_exposure", 0)
        if total_exposure >= self.limits.max_portfolio_exposure:
            log.warning(f"RISK: {decision.symbol} REJECTED — exposure {total_exposure:.1%} >= {self.limits.max_portfolio_exposure:.1%}")
            return False

        # Check if already have a position in this asset
        if decision.symbol in positions:
            log.info(f"RISK: {decision.symbol} REJECTED — already holding position")
            return False

        # Check sector concentration
        asset = get_asset(decision.symbol)
        sector_exposure = sum(
            p.get("size_pct", 0) for p in positions.values()
            if p.get("sector") == asset.sector
        )
        if sector_exposure >= self.limits.max_sector_pct:
            log.warning(f"RISK: {decision.symbol} REJECTED — sector {asset.sector} at {sector_exposure:.1%}")
            return False

        # Check asset class concentration
        class_exposure = sum(
            p.get("size_pct", 0) for p in positions.values()
            if p.get("asset_class") == asset.asset_class
        )
        if class_exposure >= self.limits.max_asset_class_pct:
            log.warning(f"RISK: {decision.symbol} REJECTED — {asset.asset_class} at {class_exposure:.1%}")
            return False

        log.info(f"RISK: {decision.symbol} APPROVED (conf={decision.confidence}, exposure={total_exposure:.1%})")
        return True

    def get_position_size(self, decision: Decision, portfolio: dict) -> float:
        """
        Calculate position size as percentage of portfolio.
        Higher confidence = larger position, capped at max_position_pct.
        """
        base_size = self.limits.max_position_pct
        confidence_factor = decision.confidence / 100.0
        size = base_size * confidence_factor
        return min(size, self.limits.max_position_pct)

    def check_circuit_breakers(self, daily_pnl: float, cumulative_pnl: float = 0) -> str:
        """
        Check circuit breaker state based on P&L.
        Returns: "ok", "warn", "throttle", or "halt"
        """
        if cumulative_pnl <= self.limits.cumulative_loss_halt:
            self._circuit_breaker_state = "halt"
            log.critical(f"CIRCUIT BREAKER: HALT — cumulative P&L {cumulative_pnl:.2%}")
            return "halt"

        if daily_pnl <= self.limits.daily_loss_halt:
            self._circuit_breaker_state = "halt"
            log.critical(f"CIRCUIT BREAKER: HALT — daily P&L {daily_pnl:.2%}")
            return "halt"

        if daily_pnl <= self.limits.daily_loss_throttle:
            self._circuit_breaker_state = "throttle"
            log.warning(f"CIRCUIT BREAKER: THROTTLE — daily P&L {daily_pnl:.2%}")
            return "throttle"

        if daily_pnl <= self.limits.daily_loss_warn:
            self._circuit_breaker_state = "warn"
            log.warning(f"CIRCUIT BREAKER: WARNING — daily P&L {daily_pnl:.2%}")
            return "warn"

        self._circuit_breaker_state = "ok"
        return "ok"

    def get_stop_target(self, decision: Decision) -> tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        return self.limits.stop_loss_pct, self.limits.take_profit_pct
