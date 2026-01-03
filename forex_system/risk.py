"""
Risk profiles and sizing guardrails for trading.

Translates a user's stated risk appetite into bounded limits for
position sizing and circuit breakers.
"""
from dataclasses import dataclass, replace
from typing import Dict


@dataclass(frozen=True)
class RiskProfile:
    """Risk limits for execution and backtesting."""

    name: str
    risk_per_trade_pct: float  # recommended per-trade risk fraction of equity
    risk_per_trade_ceiling_pct: float  # hard ceiling regardless of user ask
    max_position_pct: float  # cap on notional per position
    max_daily_loss_pct: float
    max_weekly_loss_pct: float
    max_drawdown_pct: float
    max_leverage: int
    note: str = ""

    def with_requested_risk(self, requested_pct: float) -> "RiskProfile":
        """
        Clamp a user-requested per-trade risk (0-1.0) to safe ceilings.
        """
        safe_risk = max(0.0, min(requested_pct, self.risk_per_trade_ceiling_pct))
        return replace(self, risk_per_trade_pct=safe_risk)

    def describe(self) -> Dict[str, float]:
        """Return a serializable view of limits."""
        return {
            "name": self.name,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "risk_per_trade_ceiling_pct": self.risk_per_trade_ceiling_pct,
            "max_position_pct": self.max_position_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_weekly_loss_pct": self.max_weekly_loss_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_leverage": self.max_leverage,
            "note": self.note,
        }


CONSERVATIVE = RiskProfile(
    name="conservative",
    risk_per_trade_pct=0.005,
    risk_per_trade_ceiling_pct=0.01,
    max_position_pct=0.1,
    max_daily_loss_pct=0.02,
    max_weekly_loss_pct=0.05,
    max_drawdown_pct=0.10,
    max_leverage=5,
    note="Capital preservation first",
)

MODERATE = RiskProfile(
    name="moderate",
    risk_per_trade_pct=0.01,
    risk_per_trade_ceiling_pct=0.02,
    max_position_pct=0.15,
    max_daily_loss_pct=0.03,
    max_weekly_loss_pct=0.07,
    max_drawdown_pct=0.20,
    max_leverage=10,
    note="Balanced risk vs return",
)

AGGRESSIVE = RiskProfile(
    name="aggressive",
    risk_per_trade_pct=0.02,
    risk_per_trade_ceiling_pct=0.03,
    max_position_pct=0.20,
    max_daily_loss_pct=0.05,
    max_weekly_loss_pct=0.12,
    max_drawdown_pct=0.30,
    max_leverage=20,
    note="Seeks higher ROI with defined brakes",
)

ULTRA = RiskProfile(
    name="ultra",
    risk_per_trade_pct=0.03,
    risk_per_trade_ceiling_pct=0.05,
    max_position_pct=0.25,
    max_daily_loss_pct=0.15,
    max_weekly_loss_pct=0.25,
    max_drawdown_pct=0.40,
    max_leverage=50,
    note="Max risk allowed with guardrails; 80% per-trade is never permitted",
)


def profile_from_score(risk_score_0_100: float) -> RiskProfile:
    """
    Map a 0-100 user risk score to a profile, clamped to sane ceilings.
    """
    score = max(0.0, min(100.0, risk_score_0_100))
    if score < 20:
        return CONSERVATIVE
    if score < 50:
        return MODERATE
    if score < 80:
        return AGGRESSIVE
    return ULTRA
