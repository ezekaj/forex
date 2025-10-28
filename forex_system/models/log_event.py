"""
LogEvent model - stores structured log events in database for analytics.
"""
from sqlalchemy import Column, String, DateTime, Text, JSON, Index
from sqlalchemy.sql import func
import uuid

from . import Base


class LogEvent(Base):
    """
    Stores structured log events for querying and analytics.
    """
    __tablename__ = 'log_events'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    level = Column(String(20), nullable=False)  # INFO, WARNING, ERROR, CRITICAL
    service = Column(String(50), nullable=False, index=True)
    message = Column(Text, nullable=False)
    context = Column(JSON, nullable=True)

    timestamp = Column(DateTime, default=func.now(), nullable=False, index=True)

    # Composite index for common queries
    __table_args__ = (
        Index('ix_log_events_service_timestamp', 'service', 'timestamp'),
        Index('ix_log_events_level_timestamp', 'level', 'timestamp'),
    )

    def __repr__(self) -> str:
        return f"<LogEvent(level={self.level}, service={self.service}, time={self.timestamp})>"