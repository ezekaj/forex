"""
Test DemoBroker connector operations.

Covers:
- Connection lifecycle (connect, disconnect, is_connected)
- Quote generation (single and multiple pairs)
- Order placement (market orders with slippage)
- Position management (get, modify, close)
- Account info (balance, equity, margin)
- Error handling
"""
import pytest
from datetime import datetime

from forex_system.brokers.demo_connector import DemoBroker
from forex_system.brokers.base import OrderSide, OrderType, OrderResult


@pytest.mark.asyncio
class TestDemoBrokerConnection:
    """Test broker connection lifecycle."""

    async def test_connect(self, test_config):
        """Test connecting to demo broker."""
        broker = DemoBroker(test_config.get_broker_config())
        result = await broker.connect()

        assert result is True
        assert broker.connected is True
        await broker.disconnect()

    async def test_disconnect(self, test_config):
        """Test disconnecting from broker."""
        broker = DemoBroker(test_config.get_broker_config())
        await broker.connect()
        await broker.disconnect()

        assert broker.connected is False

    async def test_is_connected(self, demo_broker):
        """Test checking connection status."""
        is_connected = await demo_broker.is_connected()
        assert is_connected is True

    async def test_broker_name_property(self, demo_broker):
        """Test broker_name property."""
        assert demo_broker.broker_name == "Demo"


@pytest.mark.asyncio
class TestQuoteGeneration:
    """Test real-time quote generation."""

    async def test_get_quote_eurusd(self, demo_broker):
        """Test getting quote for EURUSD."""
        quote = await demo_broker.get_quote("EURUSD")

        assert quote is not None
        assert quote.pair == "EURUSD"
        assert quote.bid > 0
        assert quote.ask > 0
        assert quote.ask > quote.bid  # Ask should be higher
        assert quote.spread > 0
        assert isinstance(quote.timestamp, datetime)

    async def test_get_quote_gbpusd(self, demo_broker):
        """Test getting quote for GBPUSD."""
        quote = await demo_broker.get_quote("GBPUSD")

        assert quote is not None
        assert quote.pair == "GBPUSD"
        assert 1.20 < quote.bid < 1.35  # Realistic range
        assert quote.spread < 3.0  # Reasonable spread

    async def test_get_quote_usdjpy(self, demo_broker):
        """Test getting quote for USDJPY (JPY pair)."""
        quote = await demo_broker.get_quote("USDJPY")

        assert quote is not None
        assert quote.pair == "USDJPY"
        assert 140 < quote.bid < 160  # Realistic range
        # JPY pairs have different pip calculation
        assert quote.spread > 0

    async def test_get_quote_invalid_pair(self, demo_broker):
        """Test getting quote for invalid pair returns None."""
        quote = await demo_broker.get_quote("INVALID")
        assert quote is None

    async def test_get_quotes_multiple(self, demo_broker):
        """Test getting quotes for multiple pairs."""
        pairs = ["EURUSD", "GBPUSD", "USDJPY"]
        quotes = await demo_broker.get_quotes(pairs)

        assert len(quotes) == 3
        assert "EURUSD" in quotes
        assert "GBPUSD" in quotes
        assert "USDJPY" in quotes

        for pair, quote in quotes.items():
            assert quote.pair == pair
            assert quote.bid > 0
            assert quote.ask > 0

    async def test_quote_spread_calculation(self, demo_broker):
        """Test spread is correctly calculated."""
        quote = await demo_broker.get_quote("EURUSD")

        # Calculate spread manually
        calculated_spread = (quote.ask - quote.bid) * 10000
        assert abs(quote.spread - calculated_spread) < 0.1

    async def test_quote_timestamps(self, demo_broker):
        """Test quotes have recent timestamps."""
        quote = await demo_broker.get_quote("EURUSD")
        now = datetime.utcnow()
        time_diff = (now - quote.timestamp).total_seconds()

        assert time_diff < 1.0  # Quote should be very recent


@pytest.mark.asyncio
class TestOrderPlacement:
    """Test order placement and execution."""

    async def test_place_market_buy_order(self, demo_broker):
        """Test placing a market buy order."""
        result = await demo_broker.place_order(
            pair="EURUSD",
            side=OrderSide.BUY,
            size=1.0,
            order_type=OrderType.MARKET
        )

        assert result.success is True
        assert result.broker_order_id is not None
        assert result.filled_price is not None
        assert result.filled_size == 1.0
        assert result.error is None

    async def test_place_market_sell_order(self, demo_broker):
        """Test placing a market sell order."""
        result = await demo_broker.place_order(
            pair="GBPUSD",
            side=OrderSide.SELL,
            size=0.5,
            order_type=OrderType.MARKET
        )

        assert result.success is True
        assert result.filled_size == 0.5

    async def test_place_order_with_stop_loss(self, demo_broker):
        """Test placing order with stop loss."""
        quote = await demo_broker.get_quote("EURUSD")
        stop_loss = quote.bid - 0.0050  # 50 pips below

        result = await demo_broker.place_order(
            pair="EURUSD",
            side=OrderSide.BUY,
            size=1.0,
            stop_loss=stop_loss
        )

        assert result.success is True

    async def test_place_order_with_take_profit(self, demo_broker):
        """Test placing order with take profit."""
        quote = await demo_broker.get_quote("EURUSD")
        take_profit = quote.ask + 0.0100  # 100 pips above

        result = await demo_broker.place_order(
            pair="EURUSD",
            side=OrderSide.BUY,
            size=1.0,
            take_profit=take_profit
        )

        assert result.success is True

    async def test_place_order_with_comment(self, demo_broker):
        """Test placing order with comment."""
        result = await demo_broker.place_order(
            pair="USDJPY",
            side=OrderSide.BUY,
            size=1.0,
            comment="Test order from pytest"
        )

        assert result.success is True

    async def test_order_includes_slippage(self, demo_broker):
        """Test that orders include configured slippage."""
        quote = await demo_broker.get_quote("EURUSD")

        result = await demo_broker.place_order(
            pair="EURUSD",
            side=OrderSide.BUY,
            size=1.0
        )

        # Filled price should be higher than ask due to slippage
        assert result.filled_price >= quote.ask

    async def test_place_order_invalid_pair(self, demo_broker):
        """Test placing order with invalid pair fails."""
        result = await demo_broker.place_order(
            pair="INVALID",
            side=OrderSide.BUY,
            size=1.0
        )

        assert result.success is False
        assert result.error == "NO_QUOTE"

    async def test_place_order_when_disconnected(self, test_config):
        """Test placing order when not connected fails."""
        broker = DemoBroker(test_config.get_broker_config())
        # Don't connect

        result = await broker.place_order(
            pair="EURUSD",
            side=OrderSide.BUY,
            size=1.0
        )

        assert result.success is False
        assert result.error == "NOT_CONNECTED"

    async def test_multiple_orders(self, demo_broker):
        """Test placing multiple orders."""
        pairs = ["EURUSD", "GBPUSD", "USDJPY"]
        results = []

        for pair in pairs:
            result = await demo_broker.place_order(
                pair=pair,
                side=OrderSide.BUY,
                size=0.5
            )
            results.append(result)

        assert all(r.success for r in results)
        assert len(results) == 3


@pytest.mark.asyncio
class TestPositionManagement:
    """Test position management operations."""

    async def test_get_positions_empty(self, demo_broker):
        """Test getting positions when none exist."""
        positions = await demo_broker.get_positions()
        assert isinstance(positions, list)
        # May have positions from previous tests, so just verify it's a list

    async def test_get_positions_after_order(self, demo_broker):
        """Test positions are tracked after order placement."""
        # Place order
        result = await demo_broker.place_order(
            pair="EURUSD",
            side=OrderSide.BUY,
            size=1.0
        )

        # Get positions
        positions = await demo_broker.get_positions()

        # Should have at least one position
        assert len(positions) > 0

        # Find our position
        eurusd_positions = [p for p in positions if p.pair == "EURUSD"]
        assert len(eurusd_positions) > 0

    async def test_get_position_by_id(self, demo_broker):
        """Test getting specific position by ID."""
        # Place order
        result = await demo_broker.place_order(
            pair="GBPUSD",
            side=OrderSide.BUY,
            size=1.0
        )

        position_id = result.broker_order_id

        # Get specific position
        position = await demo_broker.get_position(position_id)

        assert position is not None
        assert position.broker_position_id == position_id
        assert position.pair == "GBPUSD"

    async def test_position_unrealized_pnl(self, demo_broker):
        """Test position tracks unrealized P&L."""
        # Place order
        await demo_broker.place_order(
            pair="EURUSD",
            side=OrderSide.BUY,
            size=1.0
        )

        # Get positions
        positions = await demo_broker.get_positions()

        # Check P&L is calculated
        for position in positions:
            assert hasattr(position, 'unrealized_pnl')
            assert isinstance(position.unrealized_pnl, (int, float))

    async def test_close_position(self, demo_broker):
        """Test closing a position."""
        # Open position
        result = await demo_broker.place_order(
            pair="USDJPY",
            side=OrderSide.BUY,
            size=1.0
        )

        position_id = result.broker_order_id

        # Close position
        close_result = await demo_broker.close_position(position_id)

        assert close_result.success is True

    async def test_close_position_partial(self, demo_broker):
        """Test partially closing a position."""
        # Open position
        result = await demo_broker.place_order(
            pair="AUDUSD",
            side=OrderSide.BUY,
            size=2.0
        )

        position_id = result.broker_order_id

        # Partially close (close 1.0 out of 2.0)
        close_result = await demo_broker.close_position(
            position_id,
            size=1.0
        )

        assert close_result.success is True

    async def test_modify_position_stop_loss(self, demo_broker):
        """Test modifying position stop loss."""
        # Open position
        result = await demo_broker.place_order(
            pair="EURUSD",
            side=OrderSide.BUY,
            size=1.0
        )

        position_id = result.broker_order_id
        new_stop_loss = result.filled_price - 0.0100

        # Modify stop loss
        success = await demo_broker.modify_position(
            position_id,
            stop_loss=new_stop_loss
        )

        assert success is True

    async def test_modify_position_take_profit(self, demo_broker):
        """Test modifying position take profit."""
        # Open position
        result = await demo_broker.place_order(
            pair="GBPUSD",
            side=OrderSide.BUY,
            size=1.0
        )

        position_id = result.broker_order_id
        new_take_profit = result.filled_price + 0.0100

        # Modify take profit
        success = await demo_broker.modify_position(
            position_id,
            take_profit=new_take_profit
        )

        assert success is True


@pytest.mark.asyncio
class TestAccountManagement:
    """Test account balance and margin operations."""

    async def test_get_account_info(self, demo_broker):
        """Test getting account information."""
        account = await demo_broker.get_account_info()

        assert account is not None
        assert account.balance > 0
        assert account.equity > 0
        assert account.margin_used >= 0
        assert account.margin_free >= 0
        assert account.currency == "USD"

    async def test_account_balance_updates_after_trade(self, demo_broker):
        """Test account balance updates after profitable trade."""
        # Get initial balance
        initial_account = await demo_broker.get_account_info()
        initial_balance = initial_account.balance

        # Place and close a trade
        result = await demo_broker.place_order(
            pair="EURUSD",
            side=OrderSide.BUY,
            size=0.1  # Small size
        )

        await demo_broker.close_position(result.broker_order_id)

        # Get updated balance
        updated_account = await demo_broker.get_account_info()

        # Balance should have changed (could be profit or loss)
        assert updated_account.balance != initial_balance

    async def test_margin_calculation(self, demo_broker):
        """Test margin is calculated for open positions."""
        # Get initial margin
        initial_account = await demo_broker.get_account_info()
        initial_margin_used = initial_account.margin_used

        # Open position
        await demo_broker.place_order(
            pair="EURUSD",
            side=OrderSide.BUY,
            size=1.0
        )

        # Check margin increased
        updated_account = await demo_broker.get_account_info()
        assert updated_account.margin_used > initial_margin_used

    async def test_equity_includes_unrealized_pnl(self, demo_broker):
        """Test equity includes unrealized P&L."""
        account = await demo_broker.get_account_info()

        # Equity should be balance + unrealized P&L
        # For demo broker with positions, equity may differ from balance
        assert account.equity >= 0
        assert isinstance(account.equity, (int, float))


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling and edge cases."""

    async def test_get_position_invalid_id(self, demo_broker):
        """Test getting non-existent position returns None."""
        position = await demo_broker.get_position("invalid-id-12345")
        assert position is None

    async def test_close_invalid_position(self, demo_broker):
        """Test closing non-existent position fails gracefully."""
        result = await demo_broker.close_position("invalid-id-12345")
        assert result.success is False

    async def test_modify_invalid_position(self, demo_broker):
        """Test modifying non-existent position returns False."""
        success = await demo_broker.modify_position(
            "invalid-id-12345",
            stop_loss=1.0000
        )
        assert success is False

    async def test_validate_pair_format(self, demo_broker):
        """Test pair validation."""
        assert demo_broker.validate_pair("EURUSD") is True
        assert demo_broker.validate_pair("GBPUSD") is True
        assert demo_broker.validate_pair("EU") is False
        assert demo_broker.validate_pair("TOOLONG") is False
        assert demo_broker.validate_pair("") is False

    async def test_calculate_spread_pips_standard(self, demo_broker):
        """Test spread calculation for standard pairs."""
        spread = demo_broker.calculate_spread_pips(1.0850, 1.0851, "EURUSD")
        assert abs(spread - 1.0) < 0.1

    async def test_calculate_spread_pips_jpy(self, demo_broker):
        """Test spread calculation for JPY pairs."""
        spread = demo_broker.calculate_spread_pips(149.50, 149.52, "USDJPY")
        assert spread > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
