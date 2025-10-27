"""
Configuration model - stores user settings and preferences.
"""
from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, JSON
from sqlalchemy.sql import func
import uuid

from . import Base


class Configuration(Base):
    """
    Stores user configuration and trading preferences.
    """
    __tablename__ = 'configuration'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(JSON, nullable=False)
    description = Column(String(255), nullable=True)

    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    def __repr__(self) -> str:
        return f"<Configuration(key={self.key})>"