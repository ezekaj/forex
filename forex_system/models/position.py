"""
Position model - stores currently open positions.
"""
from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, JSON
from sqlalchemy.sql import func
import uuid

from . import Base


class Position(Base):
    """
    Represents a currently open trading position.
    """
    __tablename__ = 'positions'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    broker_position_id = Column(String(100), nullable=True)
    trade_id = Column(String(36), nullable=True)  # Link to Trade record

    # Position details
    pair = Column(String(10), nullable=False, index=True)
    direction = Column(String(5), nullable=False)
    size = Column(Float, nullable=False)
    leverage = Column(Integer, nullable=False)

    # Entry and management
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=True)

    # Performance tracking
    unrealized_pnl = Column(Float, default=0.0)
    unrealized_pips = Column(Float, default=0.0)

    # Trailing stop
    trailing_stop_enabled = Column(Boolean, default=False)
    trailing_stop_distance_pips = Column(Float, nullable=True)

    # Timestamps
    opened_at = Column(DateTime, default=func.now(), nullable=False)
    last_updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    metadata = Column(JSON, nullable=True)

    def __repr__(self) -> str:
        return f"<Position(id={self.id}, pair={self.pair}, pnl={self.unrealized_pnl})>"