"""
Trade model - stores all executed trades with complete details.
"""
from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, Text, JSON
from sqlalchemy.sql import func
from datetime import datetime
import uuid

from . import Base


class Trade(Base):
    """
    Represents a completed trade (both entry and exit recorded).
    """
    __tablename__ = 'trades'

    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Trade identification
    broker_trade_id = Column(String(100), nullable=True)  # Broker's trade ID
    opportunity_id = Column(String(36), nullable=True)  # Link to opportunity that generated this

    # Basic trade info
    pair = Column(String(10), nullable=False, index=True)  # e.g., EURUSD
    direction = Column(String(5), nullable=False)  # BUY or SELL
    strategy = Column(String(50), nullable=False, index=True)  # scalping, day_trading, swing_trading

    # Position sizing
    size = Column(Float, nullable=False)  # Lot size
    leverage = Column(Integer, nullable=False)  # Leverage used

    # Entry details
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False, index=True)

    # Exit details
    exit_price = Column(Float, nullable=True)  # Null if position still open
    exit_time = Column(DateTime, nullable=True)
    exit_reason = Column(String(50), nullable=True)  # TP, SL, manual, time_based, etc.

    # Risk management
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=True)  # Can have multiple TPs
    risk_reward_ratio = Column(Float, nullable=True)

    # Performance
    pips = Column(Float, nullable=True)  # Pips gained/lost
    profit_loss = Column(Float, nullable=True)  # P&L in account currency
    profit_loss_percentage = Column(Float, nullable=True)  # P&L as % of account

    # Trade quality metrics
    confidence_score = Column(Float, nullable=True)  # AI confidence (0-1)
    indicators_aligned = Column(Integer, nullable=True)  # Number of indicators that agreed
    indicator_details = Column(JSON, nullable=True)  # Which indicators fired

    # Execution details
    was_auto_executed = Column(Boolean, default=False)  # True if auto-executed
    user_approved = Column(Boolean, default=False)  # True if user approved
    execution_delay_seconds = Column(Float, nullable=True)  # Time from signal to execution

    # Market conditions at entry
    spread_at_entry = Column(Float, nullable=True)  # Spread in pips
    volatility_at_entry = Column(Float, nullable=True)  # ATR or similar
    trend_at_entry = Column(String(20), nullable=True)  # bullish, bearish, ranging

    # Notes and metadata
    notes = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)  # Additional custom data

    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self) -> str:
        return f"<Trade(id={self.id}, pair={self.pair}, direction={self.direction}, pnl={self.profit_loss})>"

    def to_dict(self) -> dict:
        """Convert trade to dictionary."""
        return {
            'id': self.id,
            'broker_trade_id': self.broker_trade_id,
            'opportunity_id': self.opportunity_id,
            'pair': self.pair,
            'direction': self.direction,
            'strategy': self.strategy,
            'size': self.size,
            'leverage': self.leverage,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_reason': self.exit_reason,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_reward_ratio': self.risk_reward_ratio,
            'pips': self.pips,
            'profit_loss': self.profit_loss,
            'profit_loss_percentage': self.profit_loss_percentage,
            'confidence_score': self.confidence_score,
            'indicators_aligned': self.indicators_aligned,
            'was_auto_executed': self.was_auto_executed,
            'user_approved': self.user_approved,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

    @property
    def duration_minutes(self) -> float:
        """Calculate trade duration in minutes."""
        if self.entry_time and self.exit_time:
            delta = self.exit_time - self.entry_time
            return delta.total_seconds() / 60
        return 0.0

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.profit_loss is not None and self.profit_loss > 0

    @property
    def is_loser(self) -> bool:
        """Check if trade was a loss."""
        return self.profit_loss is not None and self.profit_loss < 0