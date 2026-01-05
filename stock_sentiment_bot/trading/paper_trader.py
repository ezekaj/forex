"""
Paper Trading Simulator - Simulate trades without real money.
Tracks positions, P&L, and order history in SQLite.
"""
import logging
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, field
import uuid

logger = logging.getLogger("sentiment_bot.paper_trader")


@dataclass
class PaperPosition:
    """Simulated position."""
    ticker: str
    quantity: float
    avg_entry_price: float
    side: str  # 'long' or 'short'
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.side == 'long':
            return (current_price - self.avg_entry_price) * self.quantity
        else:
            return (self.avg_entry_price - current_price) * self.quantity

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage."""
        cost = self.avg_entry_price * self.quantity
        if cost == 0:
            return 0
        return self.unrealized_pnl(current_price) / cost * 100

    @property
    def market_value(self) -> float:
        """Position cost basis."""
        return self.avg_entry_price * self.quantity


@dataclass
class PaperOrder:
    """Simulated order."""
    order_id: str
    ticker: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: str  # 'market' or 'limit'
    limit_price: Optional[float] = None
    status: str = 'filled'  # We fill immediately for simplicity
    filled_price: Optional[float] = None
    filled_time: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class PaperAccount:
    """Simulated trading account."""
    initial_cash: float
    cash: float
    positions: Dict[str, PaperPosition] = field(default_factory=dict)
    order_history: List[PaperOrder] = field(default_factory=list)
    trade_history: List[Dict] = field(default_factory=list)

    @property
    def equity(self) -> float:
        """Total account value (cash + positions at cost)."""
        position_value = sum(p.market_value for p in self.positions.values())
        return self.cash + position_value

    def get_position_value(self, prices: Dict[str, float]) -> float:
        """Get total position value at current prices."""
        total = 0
        for ticker, pos in self.positions.items():
            price = prices.get(ticker, pos.avg_entry_price)
            total += pos.quantity * price
        return total

    def total_equity(self, prices: Dict[str, float]) -> float:
        """Total equity at current market prices."""
        return self.cash + self.get_position_value(prices)


class PaperTrader:
    """
    Paper trading simulator.

    Simulates order execution with immediate fills at current price.
    Tracks all positions and calculates P&L.
    """

    def __init__(self, initial_cash: float = 10000.0):
        self.account = PaperAccount(
            initial_cash=initial_cash,
            cash=initial_cash
        )
        self._price_getter = None
        logger.info(f"Paper trader initialized with ${initial_cash:,.2f}")

    def set_price_getter(self, func):
        """Set function to get current prices: func(ticker) -> float"""
        self._price_getter = func

    def _get_price(self, ticker: str) -> float:
        """Get current price for ticker."""
        if self._price_getter:
            return self._price_getter(ticker)
        raise ValueError("Price getter not set. Call set_price_getter() first.")

    def buy(
        self,
        ticker: str,
        quantity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> PaperOrder:
        """Execute a buy order."""
        price = self._get_price(ticker)
        cost = price * quantity

        if cost > self.account.cash:
            raise ValueError(f"Insufficient cash. Need ${cost:.2f}, have ${self.account.cash:.2f}")

        # Deduct cash
        self.account.cash -= cost

        # Update or create position
        if ticker in self.account.positions:
            pos = self.account.positions[ticker]
            total_qty = pos.quantity + quantity
            pos.avg_entry_price = (
                (pos.avg_entry_price * pos.quantity + price * quantity) / total_qty
            )
            pos.quantity = total_qty
            if stop_loss:
                pos.stop_loss = stop_loss
            if take_profit:
                pos.take_profit = take_profit
        else:
            self.account.positions[ticker] = PaperPosition(
                ticker=ticker,
                quantity=quantity,
                avg_entry_price=price,
                side='long',
                entry_time=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit
            )

        # Record order
        order = PaperOrder(
            order_id=str(uuid.uuid4())[:8],
            ticker=ticker,
            side='buy',
            quantity=quantity,
            order_type='market',
            status='filled',
            filled_price=price,
            filled_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        self.account.order_history.append(order)

        logger.info(f"BUY {quantity} {ticker} @ ${price:.2f} = ${cost:.2f}")
        return order

    def sell(
        self,
        ticker: str,
        quantity: Optional[float] = None
    ) -> PaperOrder:
        """Execute a sell order. If quantity is None, sells entire position."""
        if ticker not in self.account.positions:
            raise ValueError(f"No position in {ticker}")

        pos = self.account.positions[ticker]
        quantity = quantity or pos.quantity

        if quantity > pos.quantity:
            raise ValueError(f"Can't sell {quantity}, only have {pos.quantity}")

        price = self._get_price(ticker)
        proceeds = price * quantity

        # Calculate P&L
        cost_basis = pos.avg_entry_price * quantity
        pnl = proceeds - cost_basis
        pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0

        # Add cash
        self.account.cash += proceeds

        # Update position
        if quantity >= pos.quantity:
            del self.account.positions[ticker]
        else:
            pos.quantity -= quantity

        # Record order
        order = PaperOrder(
            order_id=str(uuid.uuid4())[:8],
            ticker=ticker,
            side='sell',
            quantity=quantity,
            order_type='market',
            status='filled',
            filled_price=price,
            filled_time=datetime.now()
        )
        self.account.order_history.append(order)

        # Record trade
        self.account.trade_history.append({
            'ticker': ticker,
            'quantity': quantity,
            'entry_price': pos.avg_entry_price,
            'exit_price': price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_time': datetime.now()
        })

        logger.info(f"SELL {quantity} {ticker} @ ${price:.2f} = ${proceeds:.2f} (P&L: ${pnl:+.2f})")
        return order

    def close_position(self, ticker: str) -> PaperOrder:
        """Close entire position."""
        return self.sell(ticker)

    def close_all_positions(self) -> List[PaperOrder]:
        """Close all positions."""
        orders = []
        for ticker in list(self.account.positions.keys()):
            orders.append(self.close_position(ticker))
        return orders

    def check_stops(self, prices: Dict[str, float]) -> List[PaperOrder]:
        """Check and execute stop losses / take profits."""
        triggered_orders = []

        for ticker, pos in list(self.account.positions.items()):
            price = prices.get(ticker)
            if not price:
                continue

            # Check stop loss
            if pos.stop_loss and price <= pos.stop_loss:
                logger.warning(f"STOP LOSS triggered for {ticker} @ ${price:.2f}")
                # Temporarily set price getter to return stop price
                old_getter = self._price_getter
                self._price_getter = lambda t, p=pos.stop_loss: p
                order = self.sell(ticker)
                self._price_getter = old_getter
                triggered_orders.append(order)

            # Check take profit
            elif pos.take_profit and price >= pos.take_profit:
                logger.info(f"TAKE PROFIT triggered for {ticker} @ ${price:.2f}")
                old_getter = self._price_getter
                self._price_getter = lambda t, p=pos.take_profit: p
                order = self.sell(ticker)
                self._price_getter = old_getter
                triggered_orders.append(order)

        return triggered_orders

    def get_positions(self) -> Dict[str, PaperPosition]:
        """Get all open positions."""
        return self.account.positions.copy()

    def get_position(self, ticker: str) -> Optional[PaperPosition]:
        """Get position for specific ticker."""
        return self.account.positions.get(ticker)

    def get_account_summary(self, prices: Optional[Dict[str, float]] = None) -> Dict:
        """Get account summary."""
        if prices is None:
            prices = {t: p.avg_entry_price for t, p in self.account.positions.items()}

        total_equity = self.account.total_equity(prices)
        total_pnl = total_equity - self.account.initial_cash
        total_pnl_pct = (total_pnl / self.account.initial_cash) * 100

        # Calculate win rate from closed trades
        trades = self.account.trade_history
        wins = sum(1 for t in trades if t['pnl'] > 0)
        losses = sum(1 for t in trades if t['pnl'] < 0)
        win_rate = wins / len(trades) * 100 if trades else 0

        return {
            'initial_cash': self.account.initial_cash,
            'cash': self.account.cash,
            'position_value': self.account.get_position_value(prices),
            'total_equity': total_equity,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'open_positions': len(self.account.positions),
            'total_trades': len(trades),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
        }

    def print_summary(self, prices: Optional[Dict[str, float]] = None):
        """Print account summary to console."""
        summary = self.get_account_summary(prices)
        print("\n" + "=" * 50)
        print("PAPER TRADING ACCOUNT SUMMARY")
        print("=" * 50)
        print(f"Initial Cash:    ${summary['initial_cash']:>12,.2f}")
        print(f"Current Cash:    ${summary['cash']:>12,.2f}")
        print(f"Position Value:  ${summary['position_value']:>12,.2f}")
        print(f"Total Equity:    ${summary['total_equity']:>12,.2f}")
        print(f"Total P&L:       ${summary['total_pnl']:>+12,.2f} ({summary['total_pnl_pct']:+.1f}%)")
        print("-" * 50)
        print(f"Open Positions:  {summary['open_positions']}")
        print(f"Total Trades:    {summary['total_trades']}")
        print(f"Win Rate:        {summary['win_rate']:.1f}%")
        print("=" * 50 + "\n")
