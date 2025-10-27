"""
Database models for the Forex Trading System.
All models use SQLAlchemy ORM for PostgreSQL.
"""
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from typing import Optional

Base = declarative_base()

# Global session factory (initialized at startup)
_engine = None
_session_factory = None


def init_db(database_url: str) -> None:
    """
    Initialize database connection and session factory.

    Args:
        database_url: PostgreSQL connection string
    """
    global _engine, _session_factory

    _engine = create_engine(
        database_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,  # Verify connections before using
        echo=False  # Set to True for SQL query logging
    )

    _session_factory = scoped_session(
        sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=_engine
        )
    )


def get_session():
    """
    Get database session.

    Returns:
        SQLAlchemy session
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _session_factory()


def create_all_tables() -> None:
    """Create all database tables."""
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    Base.metadata.create_all(bind=_engine)


def drop_all_tables() -> None:
    """Drop all database tables (use with caution)."""
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    Base.metadata.drop_all(bind=_engine)


# Import all models to register them with Base
from .trade import Trade
from .opportunity import Opportunity
from .position import Position
from .account import AccountHistory, PerformanceMetrics
from .configuration import Configuration
from .log_event import LogEvent

__all__ = [
    'Base',
    'init_db',
    'get_session',
    'create_all_tables',
    'drop_all_tables',
    'Trade',
    'Opportunity',
    'Position',
    'AccountHistory',
    'PerformanceMetrics',
    'Configuration',
    'LogEvent',
]