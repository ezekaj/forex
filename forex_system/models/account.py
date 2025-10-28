"""
Account models - account history and performance metrics.
"""
from sqlalchemy import Column, String, Float, Integer, DateTime, JSON
from sqlalchemy.sql import func
import uuid

from . import Base


class AccountHistory(Base):
    """
    Stores account balance snapshots over time.
    """
    __tablename__ = 'account_history'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    balance = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    margin_used = Column(Float, default=0.0)
    margin_free = Column(Float, nullable=True)

    # Drawdown tracking
    peak_balance = Column(Float, nullable=True)
    current_drawdown_percentage = Column(Float, default=0.0)

    # Daily/weekly tracking
    daily_pnl = Column(Float, default=0.0)
    weekly_pnl = Column(Float, default=0.0)

    snapshot_time = Column(DateTime, default=func.now(), nullable=False, index=True)

    def __repr__(self) -> str:
        return f"<AccountHistory(balance={self.balance}, time={self.snapshot_time})>"


class PerformanceMetrics(Base):
    """
    Stores aggregated performance metrics (daily, weekly, monthly).
    """
    __tablename__ = 'performance_metrics'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    period_type = Column(String(10), nullable=False)  # daily, weekly, monthly
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False)

    # Trade statistics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)

    # P&L statistics
    gross_profit = Column(Float, default=0.0)
    gross_loss = Column(Float, default=0.0)
    net_profit = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)

    # Risk metrics
    largest_win = Column(Float, default=0.0)
    largest_loss = Column(Float, default=0.0)
    average_win = Column(Float, default=0.0)
    average_loss = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)

    # Strategy breakdown
    strategy_performance = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=func.now(), nullable=False)

    def __repr__(self) -> str:
        return f"<PerformanceMetrics(period={self.period_type}, pnl={self.net_profit})>"