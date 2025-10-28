"""
Pytest configuration and fixtures for forex trading system tests.

This module provides shared fixtures for database, broker, and configuration mocking.
"""
import pytest
import asyncio
from typing import Generator, AsyncGenerator
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# Import models and config
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from forex_system.models import Base, Trade, Position, Opportunity
from forex_system.config import Config
from forex_system.brokers.demo_connector import DemoBroker
from forex_system.brokers.base import Quote, OrderSide


# ==================== Event Loop ====================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ==================== Database Fixtures ====================

@pytest.fixture(scope="function")
def db_engine():
    """Create in-memory SQLite database engine for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine) -> Generator[Session, None, None]:
    """Create a new database session for testing."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = SessionLocal()

    try:
        yield session
    finally:
        session.rollback()
        session.close()


# ==================== Config Fixtures ====================

@pytest.fixture
def test_config():
    """Create a test configuration with demo broker settings."""
    config = Config()
    config.ENVIRONMENT = "testing"
    config.BROKER = "demo"
    config.DEMO_INITIAL_BALANCE = 10000.0
    config.DEMO_SLIPPAGE_PIPS = 0.5
    config.RISK_PER_TRADE = 0.02
    config.DAILY_LOSS_LIMIT = 0.10
    config.MAX_POSITIONS = 5
    config.TRADING_PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
    config.ENABLED_STRATEGIES = ["scalping", "day_trading"]
    config.AUTO_EXECUTE_CONFIDENCE = 0.80
    config.LOG_LEVEL = "ERROR"  # Reduce noise in tests

    # Make Telegram optional for tests
    config.TELEGRAM_BOT_TOKEN = None
    config.TELEGRAM_USER_ID = None

    # Database URL for in-memory
    config.DATABASE_URL = "sqlite:///:memory:"

    return config


# ==================== Broker Fixtures ====================

@pytest.fixture
async def demo_broker(test_config):
    """Create a demo broker instance for testing."""
    broker_config = test_config.get_broker_config()
    broker = DemoBroker(broker_config)
    await broker.connect()
    yield broker
    await broker.disconnect()


@pytest.fixture
def mock_quote():
    """Create a mock quote for testing."""
    return Quote(
        pair="EURUSD",
        bid=1.0850,
        ask=1.0851,
        spread=1.0,
        timestamp=datetime.utcnow()
    )


# ==================== Model Fixtures ====================

@pytest.fixture
def sample_trade(db_session) -> Trade:
    """Create a sample trade for testing."""
    trade = Trade(
        pair="EURUSD",
        direction="BUY",
        strategy="scalping",
        size=1.0,
        leverage=50,
        entry_price=1.0850,
        entry_time=datetime.utcnow(),
        stop_loss=1.0800,
        take_profit=1.0950,
        risk_reward_ratio=2.0,
        confidence_score=0.85,
        indicators_aligned=4,
        was_auto_executed=True,
        user_approved=False,
        spread_at_entry=1.0,
        volatility_at_entry=0.0015,
        trend_at_entry="bullish"
    )
    db_session.add(trade)
    db_session.commit()
    db_session.refresh(trade)
    return trade


@pytest.fixture
def sample_position(db_session) -> Position:
    """Create a sample open position for testing."""
    position = Position(
        pair="GBPUSD",
        direction="SELL",
        size=0.5,
        leverage=30,
        entry_price=1.2650,
        current_price=1.2640,
        stop_loss=1.2700,
        take_profit=1.2550,
        unrealized_pnl=50.0,
        unrealized_pips=10.0
    )
    db_session.add(position)
    db_session.commit()
    db_session.refresh(position)
    return position


@pytest.fixture
def sample_opportunity(db_session) -> Opportunity:
    """Create a sample trading opportunity for testing."""
    opportunity = Opportunity(
        pair="USDJPY",
        direction="BUY",
        strategy="day_trading",
        timeframe="1h",
        proposed_entry_price=149.50,
        proposed_size=1.0,
        proposed_stop_loss=149.00,
        proposed_take_profit=150.50,
        leverage=30,
        confidence_score=0.82,
        risk_reward_ratio=2.0,
        indicators_aligned=5,
        current_price=149.50,
        spread_pips=2.0,
        volatility=0.003,
        trend="bullish",
        qualifies_for_auto_execute=True,
        was_sent_to_user=False,
        was_executed=False,
        expires_at=datetime.utcnow() + timedelta(minutes=5)
    )
    db_session.add(opportunity)
    db_session.commit()
    db_session.refresh(opportunity)
    return opportunity


# ==================== Mock Historical Data ====================

@pytest.fixture
def sample_historical_data():
    """Generate sample OHLCV data for backtesting."""
    from faker import Faker
    fake = Faker()

    data = []
    base_price = 1.0850

    for i in range(100):
        change = fake.pyfloat(min_value=-0.002, max_value=0.002)
        base_price += change

        data.append({
            'timestamp': datetime.utcnow() - timedelta(hours=100-i),
            'open': base_price,
            'high': base_price + abs(fake.pyfloat(min_value=0, max_value=0.001)),
            'low': base_price - abs(fake.pyfloat(min_value=0, max_value=0.001)),
            'close': base_price + fake.pyfloat(min_value=-0.0005, max_value=0.0005),
            'volume': fake.random_int(min=1000, max=10000)
        })

    return data


# ==================== Async Helpers ====================

@pytest.fixture
def anyio_backend():
    """Specify asyncio as the async backend for pytest-asyncio."""
    return "asyncio"


# ==================== Parametrize Helpers ====================

CURRENCY_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
ORDER_SIDES = [OrderSide.BUY, OrderSide.SELL]
STRATEGIES = ["scalping", "day_trading", "swing_trading"]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "asyncio: marks tests as async"
    )
