"""
Real Broker Connection Module
Supports multiple brokers with minimal cost entry
"""

import os
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
import MetaTrader5 as mt5  # For MT5 brokers
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class BrokerConnector:
    """Base class for broker connections"""
    
    def __init__(self):
        self.connected = False
        self.account_info = {}
        
    def connect(self) -> bool:
        """Connect to broker"""
        raise NotImplementedError
        
    def get_balance(self) -> float:
        """Get account balance"""
        raise NotImplementedError
        
    def get_quote(self, symbol: str) -> Dict:
        """Get current price quote"""
        raise NotImplementedError
        
    def place_order(self, symbol: str, volume: float, order_type: str, 
                   sl: Optional[float] = None, tp: Optional[float] = None) -> Dict:
        """Place an order"""
        raise NotImplementedError
        
    def close_position(self, position_id: str) -> bool:
        """Close a position"""
        raise NotImplementedError


class MT5Connector(BrokerConnector):
    """
    MetaTrader 5 Connector - Most popular for forex
    Works with: XM, FXTM, Exness, ICMarkets, etc.
    Minimum deposit: Usually $5-$10
    """
    
    def __init__(self, login: str = None, password: str = None, server: str = None):
        super().__init__()
        self.login = login or os.getenv("MT5_LOGIN")
        self.password = password or os.getenv("MT5_PASSWORD")
        self.server = server or os.getenv("MT5_SERVER", "XMGlobal-MT5 3")
        
    def connect(self) -> bool:
        """Connect to MT5"""
        try:
            # Initialize MT5
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
                
            # Login to account
            if self.login and self.password:
                authorized = mt5.login(
                    login=int(self.login),
                    password=self.password,
                    server=self.server
                )
                
                if not authorized:
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                    return False
                    
                # Get account info
                account_info = mt5.account_info()
                if account_info:
                    self.account_info = account_info._asdict()
                    self.connected = True
                    logger.info(f"Connected to MT5: Balance ${account_info.balance:.2f}")
                    return True
                    
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            
        return False
        
    def get_balance(self) -> float:
        """Get account balance"""
        if not self.connected:
            return 0.0
            
        account_info = mt5.account_info()
        if account_info:
            return account_info.balance
        return 0.0
        
    def get_quote(self, symbol: str = "EURUSD") -> Dict:
        """Get current price quote"""
        if not self.connected:
            return {}
            
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            return {
                "bid": tick.bid,
                "ask": tick.ask,
                "spread": (tick.ask - tick.bid) * 10000,  # In pips
                "time": datetime.fromtimestamp(tick.time)
            }
        return {}
        
    def place_order(self, symbol: str = "EURUSD", volume: float = 0.01, 
                   order_type: str = "BUY", sl: Optional[float] = None, 
                   tp: Optional[float] = None) -> Dict:
        """
        Place an order
        volume: Lot size (0.01 = micro lot = $1000)
        """
        if not self.connected:
            return {"error": "Not connected"}
            
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return {"error": f"Symbol {symbol} not found"}
            
        # Check if symbol is visible
        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)
            
        # Prepare request
        point = symbol_info.point
        price = mt5.symbol_info_tick(symbol).ask if order_type == "BUY" else mt5.symbol_info_tick(symbol).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "Python forex bot",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Add stop loss if provided
        if sl:
            request["sl"] = sl
            
        # Add take profit if provided
        if tp:
            request["tp"] = tp
            
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"error": f"Order failed: {result.comment}"}
            
        return {
            "order_id": result.order,
            "volume": volume,
            "price": result.price,
            "symbol": symbol,
            "type": order_type
        }
        
    def close_position(self, position_id: str) -> bool:
        """Close a position"""
        if not self.connected:
            return False
            
        position = mt5.positions_get(ticket=int(position_id))
        if not position:
            return False
            
        position = position[0]
        
        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position.ticket,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(position.symbol).bid if position.type == 0 else mt5.symbol_info_tick(position.symbol).ask,
            "deviation": 20,
            "magic": 234000,
            "comment": "Close position",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE


class OandaConnector(BrokerConnector):
    """
    OANDA Connector - Good for beginners
    Minimum deposit: $0 (practice), $1 (real)
    API: REST API, no special software needed
    """
    
    def __init__(self, api_key: str = None, account_id: str = None, practice: bool = True):
        super().__init__()
        self.api_key = api_key or os.getenv("OANDA_API_KEY")
        self.account_id = account_id or os.getenv("OANDA_ACCOUNT_ID")
        self.practice = practice
        
        # API endpoints
        if practice:
            self.base_url = "https://api-fxpractice.oanda.com/v3"
        else:
            self.base_url = "https://api-fxtrade.oanda.com/v3"
            
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    def connect(self) -> bool:
        """Connect to OANDA"""
        try:
            # Test connection by getting account info
            url = f"{self.base_url}/accounts/{self.account_id}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                self.account_info = data["account"]
                self.connected = True
                balance = float(self.account_info["balance"])
                logger.info(f"Connected to OANDA: Balance ${balance:.2f}")
                return True
                
        except Exception as e:
            logger.error(f"OANDA connection error: {e}")
            
        return False
        
    def get_balance(self) -> float:
        """Get account balance"""
        if not self.connected:
            return 0.0
            
        url = f"{self.base_url}/accounts/{self.account_id}"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            data = response.json()
            return float(data["account"]["balance"])
            
        return 0.0
        
    def get_quote(self, symbol: str = "EUR_USD") -> Dict:
        """Get current price quote"""
        if not self.connected:
            return {}
            
        url = f"{self.base_url}/accounts/{self.account_id}/pricing"
        params = {"instruments": symbol}
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data["prices"]:
                price = data["prices"][0]
                bid = float(price["bids"][0]["price"])
                ask = float(price["asks"][0]["price"])
                return {
                    "bid": bid,
                    "ask": ask,
                    "spread": (ask - bid) * 10000,
                    "time": datetime.now()
                }
                
        return {}
        
    def place_order(self, symbol: str = "EUR_USD", units: int = 1000, 
                   order_type: str = "BUY", sl: Optional[float] = None, 
                   tp: Optional[float] = None) -> Dict:
        """
        Place an order
        units: Number of units (1000 units = micro lot)
        """
        if not self.connected:
            return {"error": "Not connected"}
            
        url = f"{self.base_url}/accounts/{self.account_id}/orders"
        
        # Prepare order data
        order_data = {
            "order": {
                "units": str(units if order_type == "BUY" else -units),
                "instrument": symbol,
                "type": "MARKET",
                "positionFill": "DEFAULT"
            }
        }
        
        # Add stop loss if provided
        if sl:
            order_data["order"]["stopLossOnFill"] = {"price": str(sl)}
            
        # Add take profit if provided
        if tp:
            order_data["order"]["takeProfitOnFill"] = {"price": str(tp)}
            
        response = requests.post(url, headers=self.headers, json=order_data)
        
        if response.status_code == 201:
            data = response.json()
            return {
                "order_id": data["orderFillTransaction"]["id"],
                "units": units,
                "price": float(data["orderFillTransaction"]["price"]),
                "symbol": symbol,
                "type": order_type
            }
            
        return {"error": f"Order failed: {response.text}"}


def get_broker_connector(broker_type: str = "demo") -> BrokerConnector:
    """
    Factory function to get appropriate broker connector
    
    broker_type: "demo", "mt5", "oanda"
    """
    
    if broker_type == "mt5":
        return MT5Connector()
    elif broker_type == "oanda":
        return OandaConnector()
    else:
        # Return demo connector for testing
        return DemoConnector()


class DemoConnector(BrokerConnector):
    """Demo/Paper trading connector for testing"""
    
    def __init__(self, initial_balance: float = 1000.0):
        super().__init__()
        self.balance = initial_balance
        self.positions = {}
        self.next_position_id = 1
        
    def connect(self) -> bool:
        """Connect to demo"""
        self.connected = True
        logger.info(f"Demo mode: Starting balance ${self.balance:.2f}")
        return True
        
    def get_balance(self) -> float:
        """Get account balance"""
        return self.balance
        
    def get_quote(self, symbol: str = "EURUSD") -> Dict:
        """Get simulated quote (you'd use real data here)"""
        # In real implementation, fetch from data provider
        import random
        base_price = 1.0850
        spread = 0.0002
        
        bid = base_price + random.uniform(-0.0010, 0.0010)
        ask = bid + spread
        
        return {
            "bid": bid,
            "ask": ask, 
            "spread": spread * 10000,
            "time": datetime.now()
        }
        
    def place_order(self, symbol: str = "EURUSD", volume: float = 0.01,
                   order_type: str = "BUY", sl: Optional[float] = None,
                   tp: Optional[float] = None) -> Dict:
        """Place a demo order"""
        quote = self.get_quote(symbol)
        price = quote["ask"] if order_type == "BUY" else quote["bid"]
        
        position_id = str(self.next_position_id)
        self.next_position_id += 1
        
        self.positions[position_id] = {
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "entry_price": price,
            "sl": sl,
            "tp": tp,
            "open_time": datetime.now()
        }
        
        logger.info(f"Demo order placed: {order_type} {volume} lots of {symbol} @ {price:.5f}")
        
        return {
            "order_id": position_id,
            "volume": volume,
            "price": price,
            "symbol": symbol,
            "type": order_type
        }
        
    def close_position(self, position_id: str) -> bool:
        """Close a demo position"""
        if position_id in self.positions:
            position = self.positions[position_id]
            quote = self.get_quote(position["symbol"])
            
            exit_price = quote["bid"] if position["type"] == "BUY" else quote["ask"]
            
            # Calculate P&L
            if position["type"] == "BUY":
                pnl_pips = (exit_price - position["entry_price"]) * 10000
            else:
                pnl_pips = (position["entry_price"] - exit_price) * 10000
                
            pnl_dollars = pnl_pips * position["volume"] * 10  # Simplified calculation
            
            self.balance += pnl_dollars
            
            logger.info(f"Demo position closed: P&L = {pnl_pips:.1f} pips (${pnl_dollars:.2f})")
            logger.info(f"New balance: ${self.balance:.2f}")
            
            del self.positions[position_id]
            return True
            
        return False