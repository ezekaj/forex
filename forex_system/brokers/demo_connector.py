"""
Demo broker connector - simulated trading for testing and development.
Provides realistic fills, configurable slippage, no real money risk.
"""
import asyncio
import random
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .base import (
    BrokerInterface, OrderType, OrderSide, Quote, OrderResult,
    PositionInfo, AccountInfo
)


class DemoBroker(BrokerInterface):
    """
    Simulated broker for testing without real money.
    Generates realistic market data and simulates order fills.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.initial_balance = config.get("initial_balance", 10000.0)
        self.slippage_pips = config.get("slippage_pips", 0.5)

        # State
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.positions: Dict[str, PositionInfo] = {}
        self.base_prices = self._initialize_base_prices()

    def _initialize_base_prices(self) -> Dict[str, float]:
        """Initialize base prices for major pairs."""
        return {
            "EURUSD": 1.0850,
            "GBPUSD": 1.2650,
            "USDJPY": 149.50,
            "AUDUSD": 0.6550,
            "USDCAD": 1.3550,
            "EURJPY": 162.50,
            "GBPJPY": 189.00,
        }

    async def connect(self) -> bool:
        """Connect to demo broker (instant)."""
        self.connected = True
        return True

    async def disconnect(self) -> None:
        """Disconnect from demo broker."""
        self.connected = False

    async def is_connected(self) -> bool:
        """Check connection status."""
        return self.connected

    async def get_quote(self, pair: str) -> Optional[Quote]:
        """
        Get current quote with realistic spread.
        Prices fluctuate randomly to simulate live market.
        """
        if pair not in self.base_prices:
            return None

        # Simulate price movement
        base_price = self.base_prices[pair]
        volatility = 0.0002  # 0.02% volatility per update
        price_change = random.uniform(-volatility, volatility) * base_price
        mid_price = base_price + price_change

        # Realistic spread (wider for exotic pairs)
        if "JPY" in pair:
            spread_pips = random.uniform(1.0, 2.0)
            spread_price = spread_pips / 100
        else:
            spread_pips = random.uniform(0.5, 1.5)
            spread_price = spread_pips / 10000

        bid = mid_price - (spread_price / 2)
        ask = mid_price + (spread_price / 2)

        return Quote(
            pair=pair,
            bid=bid,
            ask=ask,
            spread=self.calculate_spread_pips(bid, ask, pair),
            timestamp=datetime.utcnow()
        )

    async def get_quotes(self, pairs: List[str]) -> Dict[str, Quote]:
        """Get quotes for multiple pairs."""
        quotes = {}
        for pair in pairs:
            quote = await self.get_quote(pair)
            if quote:
                quotes[pair] = quote
        return quotes

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
        Place order with simulated fill.
        Market orders fill instantly with slippage.
        """
        if not self.connected:
            return OrderResult(
                success=False,
                broker_order_id=None,
                filled_price=None,
                filled_size=None,
                message="Not connected to broker",
                error="NOT_CONNECTED"
            )

        # Get current quote
        quote = await self.get_quote(pair)
        if not quote:
            return OrderResult(
                success=False,
                broker_order_id=None,
                filled_price=None,
                filled_size=None,
                message=f"No quote available for {pair}",
                error="NO_QUOTE"
            )

        # Calculate fill price with slippage
        if side == OrderSide.BUY:
            base_price = quote.ask
            slippage = self.slippage_pips / (100 if "JPY" in pair else 10000)
            fill_price = base_price + slippage
        else:
            base_price = quote.bid
            slippage = self.slippage_pips / (100 if "JPY" in pair else 10000)
            fill_price = base_price - slippage

        # Check if we have enough margin
        required_margin = self._calculate_required_margin(size, fill_price)
        if required_margin > (self.balance - self._calculate_margin_used()):
            return OrderResult(
                success=False,
                broker_order_id=None,
                filled_price=None,
                filled_size=None,
                message="Insufficient margin",
                error="INSUFFICIENT_MARGIN"
            )

        # Create position
        position_id = str(uuid.uuid4())
        position = PositionInfo(
            broker_position_id=position_id,
            pair=pair,
            side=side,
            size=size,
            entry_price=fill_price,
            current_price=fill_price,
            unrealized_pnl=0.0,
            stop_loss=stop_loss,
            take_profit=take_profit,
            opened_at=datetime.utcnow()
        )

        self.positions[position_id] = position

        return OrderResult(
            success=True,
            broker_order_id=position_id,
            filled_price=fill_price,
            filled_size=size,
            message=f"Order filled: {side.value} {size} lots of {pair} at {fill_price}"
        )

    async def close_position(
        self,
        broker_position_id: str,
        size: Optional[float] = None
    ) -> OrderResult:
        """Close position with simulated fill."""
        if broker_position_id not in self.positions:
            return OrderResult(
                success=False,
                broker_order_id=None,
                filled_price=None,
                filled_size=None,
                message="Position not found",
                error="POSITION_NOT_FOUND"
            )

        position = self.positions[broker_position_id]

        # Get current quote for close price
        quote = await self.get_quote(position.pair)
        if not quote:
            return OrderResult(
                success=False,
                broker_order_id=None,
                filled_price=None,
                filled_size=None,
                message="No quote available",
                error="NO_QUOTE"
            )

        # Calculate close price (opposite side with slippage)
        if position.side == OrderSide.BUY:
            close_price = quote.bid - (self.slippage_pips / (100 if "JPY" in position.pair else 10000))
        else:
            close_price = quote.ask + (self.slippage_pips / (100 if "JPY" in position.pair else 10000))

        # Calculate P&L
        pnl = self._calculate_pnl(position, close_price)

        # Update balance
        self.balance += pnl

        # Remove position
        del self.positions[broker_position_id]

        return OrderResult(
            success=True,
            broker_order_id=broker_position_id,
            filled_price=close_price,
            filled_size=position.size,
            message=f"Position closed at {close_price}, P&L: {pnl:.2f}"
        )

    async def modify_position(
        self,
        broker_position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> bool:
        """Modify position SL/TP."""
        if broker_position_id not in self.positions:
            return False

        position = self.positions[broker_position_id]
        if stop_loss is not None:
            position.stop_loss = stop_loss
        if take_profit is not None:
            position.take_profit = take_profit

        return True

    async def get_positions(self) -> List[PositionInfo]:
        """Get all open positions."""
        # Update unrealized P&L for all positions
        for position in self.positions.values():
            quote = await self.get_quote(position.pair)
            if quote:
                current_price = quote.bid if position.side == OrderSide.BUY else quote.ask
                position.current_price = current_price
                position.unrealized_pnl = self._calculate_pnl(position, current_price)

        return list(self.positions.values())

    async def get_position(self, broker_position_id: str) -> Optional[PositionInfo]:
        """Get specific position."""
        if broker_position_id not in self.positions:
            return None

        position = self.positions[broker_position_id]

        # Update unrealized P&L
        quote = await self.get_quote(position.pair)
        if quote:
            current_price = quote.bid if position.side == OrderSide.BUY else quote.ask
            position.current_price = current_price
            position.unrealized_pnl = self._calculate_pnl(position, current_price)

        return position

    async def get_account_info(self) -> Optional[AccountInfo]:
        """Get account information."""
        margin_used = self._calculate_margin_used()
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())

        equity = self.balance + unrealized_pnl
        margin_free = equity - margin_used
        margin_level = (equity / margin_used * 100) if margin_used > 0 else None

        return AccountInfo(
            balance=self.balance,
            equity=equity,
            margin_used=margin_used,
            margin_free=margin_free,
            margin_level=margin_level,
            currency="USD"
        )

    async def get_historical_data(
        self,
        pair: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate simulated historical data.
        For demo purposes, creates realistic-looking OHLC bars.
        """
        if pair not in self.base_prices:
            return None

        # Parse timeframe
        timeframe_minutes = self._parse_timeframe(timeframe)
        if not timeframe_minutes:
            return None

        # Generate bars
        bars = []
        current_time = start_time
        base_price = self.base_prices[pair]

        while current_time < end_time:
            # Simulate price movement
            open_price = base_price
            high_price = open_price * (1 + random.uniform(0, 0.002))
            low_price = open_price * (1 - random.uniform(0, 0.002))
            close_price = random.uniform(low_price, high_price)

            bars.append({
                "time": current_time,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": random.randint(100, 1000)
            })

            # Update base for next bar
            base_price = close_price
            current_time += timedelta(minutes=timeframe_minutes)

        return bars

    def _parse_timeframe(self, timeframe: str) -> Optional[int]:
        """Convert timeframe string to minutes."""
        mapping = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440
        }
        return mapping.get(timeframe)

    def _calculate_pnl(self, position: PositionInfo, current_price: float) -> float:
        """Calculate P&L for a position."""
        if position.side == OrderSide.BUY:
            pips = (current_price - position.entry_price)
        else:
            pips = (position.entry_price - current_price)

        # Convert pips to P&L
        if "JPY" in position.pair:
            pips *= 100
        else:
            pips *= 10000

        # Standard lot size
        pnl = pips * position.size * 10  # Approximate pip value

        return pnl

    def _calculate_required_margin(self, size: float, price: float) -> float:
        """Calculate required margin for position (assuming 1:100 leverage)."""
        return (size * 100000 * price) / 100  # 1:100 leverage

    def _calculate_margin_used(self) -> float:
        """Calculate total margin used by open positions."""
        total = 0.0
        for position in self.positions.values():
            total += self._calculate_required_margin(position.size, position.entry_price)
        return total

    @property
    def broker_name(self) -> str:
        """Get broker name."""
        return "Demo"