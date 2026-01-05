"""
Tests for Risk Manager - Circuit breaker validation.
These tests verify the critical risk controls work correctly.
"""
import pytest
from trading.risk_manager import RiskManager, RiskState
from config.constants import (
    MAX_DAILY_LOSS_PERCENT,
    HALT_DRAWDOWN,
    HALT_AFTER_LOSSES,
    REDUCE_AFTER_LOSSES,
)


class TestRiskManager:
    """Test suite for RiskManager."""

    def test_initialization(self):
        """Test risk manager initializes correctly."""
        rm = RiskManager(initial_equity=10000)
        assert rm.state.current_equity == 10000
        assert rm.state.high_water_mark == 10000
        assert not rm.state.is_halted

    def test_daily_loss_circuit_breaker(self):
        """Test daily loss limit halts trading."""
        rm = RiskManager(initial_equity=10000)

        # Simulate daily loss
        rm.state.daily_starting_equity = 10000
        rm.update_equity(8900)  # -11% loss

        assert rm.state.daily_loss_halt
        assert rm.state.is_halted

        can_trade, reason = rm.can_trade()
        assert not can_trade
        assert "Daily loss limit" in reason

    def test_drawdown_circuit_breaker(self):
        """Test drawdown halt at 25%."""
        rm = RiskManager(initial_equity=10000)

        # Simulate drawdown
        rm.update_equity(7400)  # -26% from peak

        assert rm.state.drawdown >= HALT_DRAWDOWN
        assert rm.state.drawdown_halt
        assert rm.state.is_halted

    def test_consecutive_loss_circuit_breaker(self):
        """Test halt after consecutive losses."""
        rm = RiskManager(initial_equity=10000)

        # Simulate consecutive losses
        for _ in range(HALT_AFTER_LOSSES):
            rm.record_trade_result(pnl=-100, is_win=False)

        assert rm.state.consecutive_loss_halt
        assert rm.state.is_halted

    def test_risk_reduction_after_losses(self):
        """Test risk is reduced after consecutive losses."""
        rm = RiskManager(initial_equity=10000)

        # Initial risk multiplier should be 1.0
        assert rm.get_risk_multiplier() == 1.0

        # After 3 losses, should reduce
        for _ in range(REDUCE_AFTER_LOSSES):
            rm.record_trade_result(pnl=-100, is_win=False)

        assert rm.get_risk_multiplier() < 1.0

    def test_win_resets_loss_streak(self):
        """Test winning trade resets loss streak."""
        rm = RiskManager(initial_equity=10000)

        # Build up losses
        for _ in range(2):
            rm.record_trade_result(pnl=-100, is_win=False)

        assert rm.state.consecutive_losses == 2

        # Win should reset
        rm.record_trade_result(pnl=200, is_win=True)

        assert rm.state.consecutive_losses == 0
        assert rm.state.consecutive_wins == 1

    def test_high_water_mark_updates(self):
        """Test high water mark only increases."""
        rm = RiskManager(initial_equity=10000)

        rm.update_equity(11000)
        assert rm.state.high_water_mark == 11000

        rm.update_equity(10500)
        assert rm.state.high_water_mark == 11000  # Should not decrease

        rm.update_equity(12000)
        assert rm.state.high_water_mark == 12000

    def test_position_limits(self):
        """Test position size limits are enforced."""
        rm = RiskManager(initial_equity=10000)

        # Max single position is 35%
        allowed, reason = rm.check_position_limits(
            ticker="AAPL",
            proposed_value=3600,  # 36%
            current_positions={}
        )
        assert not allowed
        assert "35%" in reason

    def test_max_positions_limit(self):
        """Test max concurrent positions enforced."""
        rm = RiskManager(initial_equity=10000)

        current = {
            "AAPL": 1000,
            "NVDA": 1000,
            "TSLA": 1000,
        }

        # Should reject 4th position
        allowed, reason = rm.check_position_limits(
            ticker="GME",
            proposed_value=1000,
            current_positions=current
        )
        assert not allowed
        assert "Max positions" in reason

    def test_daily_reset(self):
        """Test daily counters reset properly."""
        rm = RiskManager(initial_equity=10000)

        rm.record_trade_result(pnl=100, is_win=True)
        rm.record_trade_result(pnl=-50, is_win=False)
        assert rm.state.daily_trades == 2

        rm.reset_daily()
        assert rm.state.daily_trades == 0
        assert rm.state.daily_pnl == 0


class TestPositionSizing:
    """Test position sizing calculations."""

    def test_risk_percent_calculation(self):
        """Test risk percent respects bounds."""
        rm = RiskManager(initial_equity=10000)

        risk = rm.calculate_position_risk()

        # Should be within bounds
        from config.constants import MIN_RISK_PERCENT, MAX_RISK_PERCENT
        assert MIN_RISK_PERCENT <= risk <= MAX_RISK_PERCENT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
