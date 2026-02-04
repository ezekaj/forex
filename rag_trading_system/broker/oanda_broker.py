"""
OANDA Broker Integration
========================
Connects to OANDA for live/practice trading.

Setup:
1. Create account at https://www.oanda.com
2. Get API key from Account Settings > API
3. Set environment variables:
   - OANDA_ACCOUNT_ID
   - OANDA_ACCESS_TOKEN
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OANDA_ACCOUNT_ID,
    OANDA_ACCESS_TOKEN,
    OANDA_ENVIRONMENT,
    MAX_RISK_PER_TRADE,
    SL_ATR_MULTIPLIER,
    TP_ATR_MULTIPLIER
)

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position."""
    id: str
    instrument: str
    units: int
    entry_price: float
    unrealized_pnl: float
    side: str  # "long" or "short"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    open_time: Optional[datetime] = None


@dataclass
class Order:
    """Represents a filled order."""
    id: str
    instrument: str
    units: int
    price: float
    side: str
    time: datetime
    pnl: float = 0.0


class OandaBroker:
    """
    OANDA REST API v20 client.

    Supports both practice and live accounts.
    """

    PRACTICE_URL = "https://api-fxpractice.oanda.com/v3"
    LIVE_URL = "https://api-fxtrade.oanda.com/v3"

    # Symbol mapping (internal -> OANDA format)
    SYMBOL_MAP = {
        "EURUSD": "EUR_USD",
        "GBPUSD": "GBP_USD",
        "USDJPY": "USD_JPY",
        "AUDUSD": "AUD_USD",
        "USDCAD": "USD_CAD",
        "USDCHF": "USD_CHF",
        "NZDUSD": "NZD_USD",
        "EURGBP": "EUR_GBP",
        "EURJPY": "EUR_JPY",
        "GBPJPY": "GBP_JPY",
    }

    def __init__(
        self,
        account_id: str = None,
        access_token: str = None,
        environment: str = None
    ):
        self.account_id = account_id or OANDA_ACCOUNT_ID
        self.access_token = access_token or OANDA_ACCESS_TOKEN
        self.environment = environment or OANDA_ENVIRONMENT

        if self.environment == "live":
            self.base_url = self.LIVE_URL
        else:
            self.base_url = self.PRACTICE_URL

        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339"
        }

        self.connected = False
        self.account_info = {}

    def _to_oanda_symbol(self, symbol: str) -> str:
        """Convert internal symbol to OANDA format."""
        return self.SYMBOL_MAP.get(symbol.upper(), symbol.replace("/", "_"))

    def _from_oanda_symbol(self, symbol: str) -> str:
        """Convert OANDA symbol to internal format."""
        for internal, oanda in self.SYMBOL_MAP.items():
            if oanda == symbol:
                return internal
        return symbol.replace("_", "")

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Dict = None,
        params: Dict = None
    ) -> Optional[Dict]:
        """Make an API request."""
        url = f"{self.base_url}{endpoint}"

        try:
            if method == "GET":
                response = requests.get(url, headers=self.headers, params=params)
            elif method == "POST":
                response = requests.post(url, headers=self.headers, json=data)
            elif method == "PUT":
                response = requests.put(url, headers=self.headers, json=data)
            else:
                raise ValueError(f"Unknown method: {method}")

            if response.status_code in [200, 201]:
                return response.json()
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Request error: {e}")
            return None

    def connect(self) -> bool:
        """Test connection and get account info."""
        if not self.account_id or not self.access_token:
            logger.error("Missing OANDA credentials. Set OANDA_ACCOUNT_ID and OANDA_ACCESS_TOKEN")
            return False

        data = self._request("GET", f"/accounts/{self.account_id}")

        if data and "account" in data:
            self.account_info = data["account"]
            self.connected = True
            balance = float(self.account_info.get("balance", 0))
            logger.info(f"Connected to OANDA ({self.environment}): Balance ${balance:,.2f}")
            return True

        return False

    def get_account_summary(self) -> Dict:
        """Get account summary."""
        data = self._request("GET", f"/accounts/{self.account_id}/summary")

        if data and "account" in data:
            account = data["account"]
            return {
                "balance": float(account.get("balance", 0)),
                "unrealized_pnl": float(account.get("unrealizedPL", 0)),
                "nav": float(account.get("NAV", 0)),
                "margin_used": float(account.get("marginUsed", 0)),
                "margin_available": float(account.get("marginAvailable", 0)),
                "open_trades": int(account.get("openTradeCount", 0)),
                "pending_orders": int(account.get("pendingOrderCount", 0))
            }

        return {}

    def get_balance(self) -> float:
        """Get current account balance."""
        summary = self.get_account_summary()
        return summary.get("balance", 0.0)

    def get_quote(self, symbol: str) -> Dict:
        """Get current bid/ask prices."""
        oanda_symbol = self._to_oanda_symbol(symbol)

        data = self._request(
            "GET",
            f"/accounts/{self.account_id}/pricing",
            params={"instruments": oanda_symbol}
        )

        if data and "prices" in data and data["prices"]:
            price = data["prices"][0]
            bid = float(price["bids"][0]["price"])
            ask = float(price["asks"][0]["price"])

            return {
                "symbol": symbol,
                "bid": bid,
                "ask": ask,
                "spread": ask - bid,
                "spread_pips": (ask - bid) * (100 if "JPY" in symbol else 10000),
                "time": datetime.now()
            }

        return {}

    def get_candles(
        self,
        symbol: str,
        granularity: str = "D",
        count: int = 100,
        from_time: datetime = None,
        to_time: datetime = None
    ) -> List[Dict]:
        """
        Get historical candle data.

        Granularity options: S5, S10, S15, S30, M1, M2, M4, M5, M10, M15, M30,
                            H1, H2, H3, H4, H6, H8, H12, D, W, M
        """
        oanda_symbol = self._to_oanda_symbol(symbol)

        params = {
            "granularity": granularity,
            "count": count
        }

        if from_time:
            params["from"] = from_time.isoformat() + "Z"
        if to_time:
            params["to"] = to_time.isoformat() + "Z"

        data = self._request(
            "GET",
            f"/instruments/{oanda_symbol}/candles",
            params=params
        )

        if data and "candles" in data:
            candles = []
            for c in data["candles"]:
                if c["complete"]:
                    mid = c["mid"]
                    candles.append({
                        "time": datetime.fromisoformat(c["time"].replace("Z", "+00:00")),
                        "open": float(mid["o"]),
                        "high": float(mid["h"]),
                        "low": float(mid["l"]),
                        "close": float(mid["c"]),
                        "volume": int(c["volume"])
                    })
            return candles

        return []

    def place_market_order(
        self,
        symbol: str,
        units: int,
        stop_loss: float = None,
        take_profit: float = None
    ) -> Optional[Order]:
        """
        Place a market order.

        Args:
            symbol: Currency pair (e.g., "EURUSD")
            units: Positive for buy, negative for sell
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Order object if successful, None otherwise
        """
        oanda_symbol = self._to_oanda_symbol(symbol)

        order_data = {
            "order": {
                "type": "MARKET",
                "instrument": oanda_symbol,
                "units": str(units),
                "positionFill": "DEFAULT"
            }
        }

        # Add stop loss
        if stop_loss:
            order_data["order"]["stopLossOnFill"] = {
                "price": f"{stop_loss:.5f}"
            }

        # Add take profit
        if take_profit:
            order_data["order"]["takeProfitOnFill"] = {
                "price": f"{take_profit:.5f}"
            }

        data = self._request(
            "POST",
            f"/accounts/{self.account_id}/orders",
            data=order_data
        )

        if data and "orderFillTransaction" in data:
            fill = data["orderFillTransaction"]

            order = Order(
                id=fill["id"],
                instrument=symbol,
                units=abs(int(fill["units"])),
                price=float(fill["price"]),
                side="buy" if int(fill["units"]) > 0 else "sell",
                time=datetime.fromisoformat(fill["time"].replace("Z", "+00:00"))
            )

            logger.info(f"Order filled: {order.side.upper()} {order.units} {symbol} @ {order.price:.5f}")
            return order

        return None

    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        data = self._request("GET", f"/accounts/{self.account_id}/openTrades")

        if data and "trades" in data:
            positions = []
            for t in data["trades"]:
                units = int(t["currentUnits"])
                position = Position(
                    id=t["id"],
                    instrument=self._from_oanda_symbol(t["instrument"]),
                    units=abs(units),
                    entry_price=float(t["price"]),
                    unrealized_pnl=float(t["unrealizedPL"]),
                    side="long" if units > 0 else "short",
                    open_time=datetime.fromisoformat(t["openTime"].replace("Z", "+00:00"))
                )

                # Get SL/TP if set
                if "stopLossOrder" in t:
                    position.stop_loss = float(t["stopLossOrder"]["price"])
                if "takeProfitOrder" in t:
                    position.take_profit = float(t["takeProfitOrder"]["price"])

                positions.append(position)

            return positions

        return []

    def close_position(self, trade_id: str, units: int = None) -> Optional[float]:
        """
        Close a position.

        Args:
            trade_id: The trade ID to close
            units: Units to close (None = all)

        Returns:
            Realized P&L if successful, None otherwise
        """
        close_data = {}
        if units:
            close_data["units"] = str(units)
        else:
            close_data["units"] = "ALL"

        data = self._request(
            "PUT",
            f"/accounts/{self.account_id}/trades/{trade_id}/close",
            data=close_data
        )

        if data and "orderFillTransaction" in data:
            fill = data["orderFillTransaction"]
            pnl = float(fill.get("pl", 0))
            logger.info(f"Position {trade_id} closed. P&L: ${pnl:.2f}")
            return pnl

        return None

    def modify_position(
        self,
        trade_id: str,
        stop_loss: float = None,
        take_profit: float = None
    ) -> bool:
        """Modify stop loss or take profit on an existing position."""
        modify_data = {}

        if stop_loss:
            modify_data["stopLoss"] = {"price": f"{stop_loss:.5f}"}
        if take_profit:
            modify_data["takeProfit"] = {"price": f"{take_profit:.5f}"}

        if not modify_data:
            return False

        data = self._request(
            "PUT",
            f"/accounts/{self.account_id}/trades/{trade_id}/orders",
            data=modify_data
        )

        return data is not None

    def calculate_units(
        self,
        symbol: str,
        risk_amount: float,
        stop_loss_pips: float
    ) -> int:
        """
        Calculate position size based on risk.

        Args:
            symbol: Currency pair
            risk_amount: Amount willing to risk in account currency
            stop_loss_pips: Stop loss distance in pips

        Returns:
            Number of units to trade
        """
        # Pip value depends on the pair
        if "JPY" in symbol:
            pip_value = 0.01
        else:
            pip_value = 0.0001

        # Get current price to calculate pip value in account currency
        quote = self.get_quote(symbol)
        if not quote:
            return 0

        # For pairs where USD is the quote currency (EURUSD, GBPUSD, etc.)
        # pip value per unit is already in USD
        if symbol.endswith("USD"):
            pip_value_per_unit = pip_value
        else:
            # Need to convert - simplified for now
            pip_value_per_unit = pip_value / quote["bid"]

        # Calculate units
        units = int(risk_amount / (stop_loss_pips * pip_value_per_unit))

        return units

    def get_transaction_history(
        self,
        from_time: datetime = None,
        count: int = 100
    ) -> List[Dict]:
        """Get recent transactions."""
        params = {"count": count}
        if from_time:
            params["from"] = from_time.isoformat() + "Z"

        data = self._request(
            "GET",
            f"/accounts/{self.account_id}/transactions",
            params=params
        )

        if data and "transactions" in data:
            return data["transactions"]

        return []


def get_broker(environment: str = None) -> OandaBroker:
    """Factory function to get configured broker."""
    env = environment or OANDA_ENVIRONMENT
    broker = OandaBroker(environment=env)
    return broker


def test_connection():
    """Test OANDA connection."""
    print("\n" + "="*50)
    print("   OANDA CONNECTION TEST")
    print("="*50)

    broker = get_broker()

    if not broker.connect():
        print("\n✗ Connection FAILED")
        print("  Make sure you've set:")
        print("    - OANDA_ACCOUNT_ID")
        print("    - OANDA_ACCESS_TOKEN")
        print("\n  Get credentials from: https://www.oanda.com/account/tpa/personal_token")
        return False

    print(f"\n✓ Connected to OANDA ({broker.environment})")

    # Get account summary
    summary = broker.get_account_summary()
    print(f"\nAccount Summary:")
    print(f"  Balance:        ${summary['balance']:,.2f}")
    print(f"  NAV:            ${summary['nav']:,.2f}")
    print(f"  Unrealized P&L: ${summary['unrealized_pnl']:,.2f}")
    print(f"  Margin Used:    ${summary['margin_used']:,.2f}")
    print(f"  Open Trades:    {summary['open_trades']}")

    # Get quotes for primary pairs
    print(f"\nCurrent Quotes:")
    for pair in ["EURUSD", "GBPUSD", "USDJPY"]:
        quote = broker.get_quote(pair)
        if quote:
            print(f"  {pair}: Bid {quote['bid']:.5f} / Ask {quote['ask']:.5f} (Spread: {quote['spread_pips']:.1f} pips)")

    # Get open positions
    positions = broker.get_open_positions()
    if positions:
        print(f"\nOpen Positions:")
        for p in positions:
            print(f"  {p.side.upper()} {p.units} {p.instrument} @ {p.entry_price:.5f} (P&L: ${p.unrealized_pnl:.2f})")
    else:
        print(f"\nNo open positions")

    print("\n" + "="*50)
    print("   TEST COMPLETE")
    print("="*50 + "\n")

    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    test_connection()
