"""
Position Sizer - Kelly Criterion based sizing with adaptive scaling.
Calculates exact position size based on signal confidence and risk state.
"""
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from config.constants import (
    BASE_RISK_PERCENT,
    MIN_RISK_PERCENT,
    MAX_RISK_PERCENT,
    CONFIDENCE_RISK_MAP,
    STOP_LOSS_ATR_MULTIPLIER,
    MIN_REWARD_RISK_RATIO,
)


logger = logging.getLogger("sentiment_bot.position_sizer")


@dataclass
class PositionSize:
    """Calculated position parameters."""
    ticker: str
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit_1: float  # 50% exit
    take_profit_2: float  # 30% exit
    take_profit_3: float  # 20% exit (runner)

    dollar_risk: float
    risk_percent: float
    reward_risk_ratio: float

    def __str__(self) -> str:
        return (
            f"{self.ticker}: {self.quantity:.2f} shares @ ${self.entry_price:.2f}\n"
            f"  Stop: ${self.stop_loss:.2f} | TP1: ${self.take_profit_1:.2f} | "
            f"TP2: ${self.take_profit_2:.2f} | TP3: ${self.take_profit_3:.2f}\n"
            f"  Risk: ${self.dollar_risk:.2f} ({self.risk_percent*100:.1f}%) | R:R = {self.reward_risk_ratio:.1f}"
        )


class PositionSizer:
    """
    Quarter-Kelly position sizing with adaptive scaling.

    Base risk: 4% of account
    Scaled by:
    - Signal confidence (0.5x to 1.25x)
    - Consecutive losses (0.5x after 3 losses)
    - Drawdown level (0.25x to 1.0x)
    """

    def __init__(self, account_equity: float, risk_manager=None):
        self.account_equity = account_equity
        self.risk_manager = risk_manager

    def update_equity(self, new_equity: float):
        """Update account equity for sizing calculations."""
        self.account_equity = new_equity

    def calculate_position(
        self,
        ticker: str,
        entry_price: float,
        atr: float,
        signal_score: int,
        target_rr: float = 2.0,
        override_risk_pct: Optional[float] = None
    ) -> Optional[PositionSize]:
        """
        Calculate position size and levels.

        Args:
            ticker: Stock symbol
            entry_price: Expected entry price
            atr: Average True Range for stop calculation
            signal_score: Signal score (0-100)
            target_rr: Target reward:risk ratio
            override_risk_pct: Override calculated risk percent

        Returns:
            PositionSize with all calculated values, or None if invalid
        """
        # Get base risk percent
        if override_risk_pct is not None:
            risk_pct = override_risk_pct
        else:
            risk_pct = self._calculate_risk_percent(signal_score)

        # Calculate stop loss based on ATR
        stop_distance = atr * STOP_LOSS_ATR_MULTIPLIER
        stop_loss = entry_price - stop_distance

        # Validate reward:risk is acceptable
        potential_reward = entry_price * target_rr * (stop_distance / entry_price)
        actual_rr = potential_reward / stop_distance

        if actual_rr < MIN_REWARD_RISK_RATIO:
            logger.warning(
                f"{ticker}: R:R {actual_rr:.2f} below minimum {MIN_REWARD_RISK_RATIO}"
            )
            return None

        # Calculate dollar risk
        dollar_risk = self.account_equity * risk_pct

        # Calculate quantity
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            logger.error(f"{ticker}: Invalid stop loss (entry={entry_price}, stop={stop_loss})")
            return None

        quantity = dollar_risk / risk_per_share

        # Round down to avoid exceeding risk
        quantity = int(quantity)
        if quantity < 1:
            logger.warning(f"{ticker}: Position size too small ({quantity} shares)")
            return None

        # Recalculate actual dollar risk with rounded quantity
        actual_dollar_risk = quantity * risk_per_share
        actual_risk_pct = actual_dollar_risk / self.account_equity

        # Calculate take profit levels (scaled exits)
        # TP1 at 1.5R, TP2 at 2.5R, TP3 at 4R
        tp1 = entry_price + (stop_distance * 1.5)
        tp2 = entry_price + (stop_distance * 2.5)
        tp3 = entry_price + (stop_distance * 4.0)

        return PositionSize(
            ticker=ticker,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=round(stop_loss, 2),
            take_profit_1=round(tp1, 2),
            take_profit_2=round(tp2, 2),
            take_profit_3=round(tp3, 2),
            dollar_risk=round(actual_dollar_risk, 2),
            risk_percent=actual_risk_pct,
            reward_risk_ratio=actual_rr
        )

    def calculate_add_to_position(
        self,
        ticker: str,
        current_quantity: float,
        current_avg_price: float,
        new_entry_price: float,
        atr: float,
        signal_score: int,
        max_position_pct: float = 0.35
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate quantity to add to existing position.

        Returns:
            Tuple of (add_quantity, new_stop_loss) or None if max reached
        """
        # Calculate current position value
        current_value = current_quantity * new_entry_price
        current_pct = current_value / self.account_equity

        # Check if we can add more
        remaining_pct = max_position_pct - current_pct
        if remaining_pct <= 0.01:  # Less than 1% room
            logger.info(f"{ticker}: Position at max size ({current_pct*100:.1f}%)")
            return None

        # Calculate risk for additional shares
        risk_pct = self._calculate_risk_percent(signal_score)
        risk_pct = min(risk_pct, remaining_pct)  # Don't exceed position limit

        # Calculate stop and quantity
        stop_distance = atr * STOP_LOSS_ATR_MULTIPLIER
        new_stop = new_entry_price - stop_distance

        dollar_risk = self.account_equity * risk_pct
        risk_per_share = new_entry_price - new_stop
        add_quantity = int(dollar_risk / risk_per_share)

        if add_quantity < 1:
            return None

        return add_quantity, round(new_stop, 2)

    def _calculate_risk_percent(self, signal_score: int) -> float:
        """
        Calculate risk percent based on signal score and risk manager state.
        """
        # Base risk from signal score
        base_risk = BASE_RISK_PERCENT

        for (low, high), multiplier in CONFIDENCE_RISK_MAP.items():
            if low <= signal_score <= high:
                base_risk = BASE_RISK_PERCENT * multiplier
                break

        # Apply risk manager multiplier if available
        if self.risk_manager:
            base_risk *= self.risk_manager.get_risk_multiplier()

        # Enforce bounds
        return max(MIN_RISK_PERCENT, min(MAX_RISK_PERCENT, base_risk))

    def size_for_dollar_amount(
        self,
        ticker: str,
        entry_price: float,
        dollar_amount: float
    ) -> int:
        """Simple sizing: how many shares for dollar amount."""
        return int(dollar_amount / entry_price)

    def size_for_risk_amount(
        self,
        ticker: str,
        entry_price: float,
        stop_loss: float,
        risk_amount: float
    ) -> int:
        """Size position for specific dollar risk."""
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0
        return int(risk_amount / risk_per_share)

    def get_max_position_value(self, max_pct: float = 0.35) -> float:
        """Get maximum position value allowed."""
        return self.account_equity * max_pct

    def estimate_position_value(self, quantity: float, price: float) -> float:
        """Estimate position value."""
        return quantity * price

    def estimate_position_pct(self, quantity: float, price: float) -> float:
        """Estimate position as percent of account."""
        return (quantity * price) / self.account_equity
