"""
Alpaca API client for market data and trade execution.
Handles both paper and live trading through same interface.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, OrderType
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

from config.settings import AlpacaConfig


logger = logging.getLogger("sentiment_bot.alpaca")


@dataclass
class Quote:
    """Current price quote."""
    ticker: str
    bid: float
    ask: float
    last: float
    timestamp: datetime

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class Bar:
    """OHLCV bar data."""
    ticker: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None


@dataclass
class Position:
    """Current position."""
    ticker: str
    quantity: float
    side: str
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    market_value: float


@dataclass
class AccountInfo:
    """Account summary."""
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    day_trade_count: int
    pattern_day_trader: bool


class AlpacaClient:
    """Unified client for Alpaca trading and market data."""

    def __init__(self, config: AlpacaConfig):
        self.config = config
        config.validate()

        # Initialize trading client
        self.trading = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=config.is_paper
        )

        # Initialize data client (no auth needed for IEX data)
        self.data = StockHistoricalDataClient(
            api_key=config.api_key,
            secret_key=config.secret_key
        )

        logger.info(f"Alpaca client initialized (paper={config.is_paper})")

    # =========================================================================
    # ACCOUNT METHODS
    # =========================================================================

    def get_account(self) -> AccountInfo:
        """Get account information."""
        account = self.trading.get_account()
        return AccountInfo(
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            portfolio_value=float(account.portfolio_value),
            day_trade_count=account.daytrade_count,
            pattern_day_trader=account.pattern_day_trader
        )

    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        positions = self.trading.get_all_positions()
        return [
            Position(
                ticker=p.symbol,
                quantity=float(p.qty),
                side="long" if float(p.qty) > 0 else "short",
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                unrealized_pnl=float(p.unrealized_pl),
                unrealized_pnl_percent=float(p.unrealized_plpc) * 100,
                market_value=float(p.market_value)
            )
            for p in positions
        ]

    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for specific ticker."""
        try:
            p = self.trading.get_open_position(ticker)
            return Position(
                ticker=p.symbol,
                quantity=float(p.qty),
                side="long" if float(p.qty) > 0 else "short",
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                unrealized_pnl=float(p.unrealized_pl),
                unrealized_pnl_percent=float(p.unrealized_plpc) * 100,
                market_value=float(p.market_value)
            )
        except Exception:
            return None

    # =========================================================================
    # MARKET DATA METHODS
    # =========================================================================

    def get_quote(self, ticker: str) -> Quote:
        """Get latest quote for ticker."""
        request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
        quotes = self.data.get_stock_latest_quote(request)
        q = quotes[ticker]

        return Quote(
            ticker=ticker,
            bid=float(q.bid_price),
            ask=float(q.ask_price),
            last=float(q.ask_price),  # Use ask as last if not available
            timestamp=q.timestamp
        )

    def get_bars(
        self,
        ticker: str,
        timeframe: str = "1Day",
        limit: int = 100,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[Bar]:
        """Get historical bars."""
        # Map string to TimeFrame
        tf_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, "Min"),
            "15Min": TimeFrame(15, "Min"),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }
        tf = tf_map.get(timeframe, TimeFrame.Day)

        # Default date range
        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(days=limit * 2)  # Buffer for weekends

        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=tf,
            start=start,
            end=end,
            limit=limit
        )

        bars = self.data.get_stock_bars(request)

        return [
            Bar(
                ticker=ticker,
                timestamp=b.timestamp,
                open=float(b.open),
                high=float(b.high),
                low=float(b.low),
                close=float(b.close),
                volume=int(b.volume),
                vwap=float(b.vwap) if b.vwap else None
            )
            for b in bars[ticker]
        ]

    def get_latest_bar(self, ticker: str) -> Bar:
        """Get most recent bar."""
        bars = self.get_bars(ticker, limit=1)
        return bars[-1] if bars else None

    # =========================================================================
    # ORDER METHODS
    # =========================================================================

    def market_buy(
        self,
        ticker: str,
        quantity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> str:
        """Place market buy order."""
        request = MarketOrderRequest(
            symbol=ticker,
            qty=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )

        # Add bracket orders if provided
        if stop_loss:
            request.stop_loss = StopLossRequest(stop_price=stop_loss)
        if take_profit:
            request.take_profit = TakeProfitRequest(limit_price=take_profit)

        order = self.trading.submit_order(request)
        logger.info(f"Market BUY order placed: {ticker} x{quantity} (order_id={order.id})")
        return order.id

    def market_sell(
        self,
        ticker: str,
        quantity: float
    ) -> str:
        """Place market sell order."""
        request = MarketOrderRequest(
            symbol=ticker,
            qty=quantity,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )

        order = self.trading.submit_order(request)
        logger.info(f"Market SELL order placed: {ticker} x{quantity} (order_id={order.id})")
        return order.id

    def limit_buy(
        self,
        ticker: str,
        quantity: float,
        limit_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> str:
        """Place limit buy order."""
        request = LimitOrderRequest(
            symbol=ticker,
            qty=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price
        )

        if stop_loss:
            request.stop_loss = StopLossRequest(stop_price=stop_loss)
        if take_profit:
            request.take_profit = TakeProfitRequest(limit_price=take_profit)

        order = self.trading.submit_order(request)
        logger.info(f"Limit BUY order placed: {ticker} x{quantity} @{limit_price}")
        return order.id

    def close_position(self, ticker: str) -> str:
        """Close entire position for ticker."""
        order = self.trading.close_position(ticker)
        logger.info(f"Position closed: {ticker}")
        return order.id

    def close_all_positions(self) -> List[str]:
        """Close all open positions."""
        orders = self.trading.close_all_positions()
        logger.warning(f"All positions closed ({len(orders)} orders)")
        return [o.id for o in orders]

    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        try:
            self.trading.cancel_order_by_id(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def cancel_all_orders(self) -> int:
        """Cancel all pending orders."""
        statuses = self.trading.cancel_orders()
        count = len(statuses)
        logger.warning(f"All orders cancelled ({count} orders)")
        return count

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get order status."""
        order = self.trading.get_order_by_id(order_id)
        return {
            "id": order.id,
            "ticker": order.symbol,
            "side": order.side.value,
            "type": order.type.value,
            "status": order.status.value,
            "quantity": float(order.qty),
            "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            "submitted_at": order.submitted_at,
            "filled_at": order.filled_at,
        }

    def get_open_orders(self, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders, optionally filtered by ticker."""
        request = GetOrdersRequest(status=OrderStatus.OPEN)
        if ticker:
            request.symbols = [ticker]

        orders = self.trading.get_orders(request)
        return [
            {
                "id": o.id,
                "ticker": o.symbol,
                "side": o.side.value,
                "type": o.type.value,
                "quantity": float(o.qty),
                "limit_price": float(o.limit_price) if o.limit_price else None,
                "stop_price": float(o.stop_price) if o.stop_price else None,
            }
            for o in orders
        ]

    # =========================================================================
    # MARKET STATUS
    # =========================================================================

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        clock = self.trading.get_clock()
        return clock.is_open

    def get_next_open(self) -> datetime:
        """Get next market open time."""
        clock = self.trading.get_clock()
        return clock.next_open

    def get_next_close(self) -> datetime:
        """Get next market close time."""
        clock = self.trading.get_clock()
        return clock.next_close
