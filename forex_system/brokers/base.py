"""
Abstract broker interface for unified trading operations.
All broker connectors must implement this interface.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class OrderType(Enum):
    """Order types supported by all brokers."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderSide(Enum):
    """Order side (direction)."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Quote:
    """Real-time price quote for a currency pair."""
    pair: str
    bid: float
    ask: float
    spread: float  # in pips
    timestamp: datetime


@dataclass
class OrderResult:
    """Result of order placement."""
    success: bool
    broker_order_id: Optional[str]
    filled_price: Optional[float]
    filled_size: Optional[float]
    message: str
    error: Optional[str] = None


@dataclass
class PositionInfo:
    """Information about an open position."""
    broker_position_id: str
    pair: str
    side: OrderSide
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    opened_at: datetime


@dataclass
class AccountInfo:
    """Account balance and margin information."""
    balance: float
    equity: float
    margin_used: float
    margin_free: float
    margin_level: Optional[float]  # Margin level percentage
    currency: str


class BrokerInterface(ABC):
    """
    Abstract base class for broker connectors.
    All broker implementations (MT5, OANDA, Demo) must implement this interface.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize broker with configuration.

        Args:
            config: Broker-specific configuration dictionary
        """
        self.config = config
        self.connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the broker.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the broker."""
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        """
        Check if connected to broker.

        Returns:
            True if connected, False otherwise
        """
        pass

    @abstractmethod
    async def get_quote(self, pair: str) -> Optional[Quote]:
        """
        Get current quote for a currency pair.

        Args:
            pair: Currency pair (e.g., "EURUSD")

        Returns:
            Quote object or None if unavailable
        """
        pass

    @abstractmethod
    async def get_quotes(self, pairs: List[str]) -> Dict[str, Quote]:
        """
        Get quotes for multiple currency pairs.

        Args:
            pairs: List of currency pairs

        Returns:
            Dictionary mapping pairs to quotes
        """
        pass

    @abstractmethod
    async def place_order(
        self,
        pair: str,
        side: OrderSide,
        size: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: Optional[str] = None
    ) -> OrderResult:
        """
        Place a trading order.

        Args:
            pair: Currency pair
            side: BUY or SELL
            size: Lot size
            order_type: Market, limit, or stop order
            price: Price for limit/stop orders
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Optional comment/tag

        Returns:
            OrderResult with success status and details
        """
        pass

    @abstractmethod
    async def close_position(
        self,
        broker_position_id: str,
        size: Optional[float] = None
    ) -> OrderResult:
        """
        Close an open position.

        Args:
            broker_position_id: Broker's position ID
            size: Optional partial close size (None = close all)

        Returns:
            OrderResult with success status
        """
        pass

    @abstractmethod
    async def modify_position(
        self,
        broker_position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> bool:
        """
        Modify stop loss or take profit of an open position.

        Args:
            broker_position_id: Broker's position ID
            stop_loss: New stop loss price (None = don't change)
            take_profit: New take profit price (None = don't change)

        Returns:
            True if modification successful
        """
        pass

    @abstractmethod
    async def get_positions(self) -> List[PositionInfo]:
        """
        Get all open positions.

        Returns:
            List of open positions
        """
        pass

    @abstractmethod
    async def get_position(self, broker_position_id: str) -> Optional[PositionInfo]:
        """
        Get specific position by ID.

        Args:
            broker_position_id: Broker's position ID

        Returns:
            PositionInfo or None if not found
        """
        pass

    @abstractmethod
    async def get_account_info(self) -> Optional[AccountInfo]:
        """
        Get account balance and margin information.

        Returns:
            AccountInfo or None if unavailable
        """
        pass

    @abstractmethod
    async def get_historical_data(
        self,
        pair: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get historical OHLC data.

        Args:
            pair: Currency pair
            timeframe: Timeframe (e.g., "1m", "5m", "1h", "1d")
            start_time: Start of data range
            end_time: End of data range

        Returns:
            List of OHLC bars or None if unavailable
        """
        pass

    def validate_pair(self, pair: str) -> bool:
        """
        Validate currency pair format.

        Args:
            pair: Currency pair string

        Returns:
            True if valid format
        """
        # Basic validation: should be 6 characters
        return isinstance(pair, str) and len(pair) == 6

    def calculate_spread_pips(self, bid: float, ask: float, pair: str) -> float:
        """
        Calculate spread in pips.

        Args:
            bid: Bid price
            ask: Ask price
            pair: Currency pair (to determine pip value)

        Returns:
            Spread in pips
        """
        # JPY pairs have different pip calculation
        if "JPY" in pair:
            return (ask - bid) * 100
        else:
            return (ask - bid) * 10000

    @property
    @abstractmethod
    def broker_name(self) -> str:
        """
        Get broker name.

        Returns:
            Broker name string
        """
        pass