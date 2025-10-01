"""
REAL MT5 TRADING AGENT - Live Demo Account Trading
==================================================
Direct connection to MT5 demo account for real trading
"""

import MetaTrader5 as mt5
import asyncio
import numpy as np
from datetime import datetime, timedelta
import time
import json
from collections import deque
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class MT5TradingAgent:
    """Real MT5 trading agent with live account integration"""
    
    def __init__(self, login: int = None, password: str = None, server: str = None):
        # MT5 Connection Details (your actual demo account details)
        self.login = login or 95948709  # Your demo account login
        self.password = password or 'To-4KyLg'  # Your demo account password  
        self.server = server or 'MetaQuotes-Demo'  # Your demo server
        
        # Trading Configuration
        self.config = {
            'magic_number': 12345678,    # Unique magic number for our trades
            'min_confidence': 0.45,      # 45% minimum confidence
            'max_positions': 3,          # Maximum 3 open positions
            'risk_per_trade': 0.01,      # 1% risk per trade
            'stop_loss_pips': 20,        # 20 pip stop loss
            'take_profit_pips': 30,      # 30 pip take profit
            'lot_size': 0.01,            # Base lot size (0.01 = micro lot)
            'slippage': 10,              # 10 points slippage tolerance
        }
        
        # Symbols to trade (ensure these are available in your MT5)
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        
        # Trading state
        self.connected = False
        self.account_info = None
        self.symbol_info = {}
        self.open_positions = {}
        self.trade_history = []
        
        # Market data storage
        self.price_data = {symbol: deque(maxlen=100) for symbol in self.symbols}
        self.latest_prices = {}
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'start_time': datetime.now(),
            'opportunities_found': 0
        }
        
        # Control flags
        self.running = False
        self.trading_enabled = True
    
    def connect_to_mt5(self) -> bool:
        """Connect to MT5 terminal"""
        print("🔄 Connecting to MT5...")
        
        # Initialize MT5 connection
        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            print(f"Error code: {mt5.last_error()}")
            return False
        
        # Login to account
        if not mt5.login(self.login, password=self.password, server=self.server):
            print("❌ MT5 login failed")
            print(f"Error code: {mt5.last_error()}")
            mt5.shutdown()
            return False
        
        # Get account information
        self.account_info = mt5.account_info()
        if self.account_info is None:
            print("❌ Failed to get account information")
            mt5.shutdown()
            return False
        
        self.connected = True
        
        print("✅ MT5 Connected Successfully!")
        print(f"Account: {self.account_info.login}")
        print(f"Server: {self.account_info.server}")
        print(f"Balance: ${self.account_info.balance:.2f}")
        print(f"Equity: ${self.account_info.equity:.2f}")
        print(f"Margin: ${self.account_info.margin:.2f}")
        print(f"Free Margin: ${self.account_info.margin_free:.2f}")
        
        # Get symbol information
        self.load_symbol_info()
        
        return True
    
    def load_symbol_info(self):
        """Load symbol information and ensure symbols are available"""
        print("\n📊 Loading symbol information...")
        
        for symbol in self.symbols:
            # Select symbol in market watch
            if not mt5.symbol_select(symbol, True):
                print(f"⚠️ Failed to select symbol {symbol}")
                continue
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"⚠️ Failed to get info for {symbol}")
                continue
            
            self.symbol_info[symbol] = symbol_info
            
            print(f"✅ {symbol}: Spread={symbol_info.spread} points, "
                  f"Digits={symbol_info.digits}, "
                  f"Min Lot={symbol_info.volume_min}")
    
    def get_point_value(self, symbol: str) -> float:
        """Get point value for a symbol"""
        if symbol not in self.symbol_info:
            return 0.0001 if 'JPY' not in symbol else 0.01
        
        return self.symbol_info[symbol].point
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current market price"""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        
        return {
            'symbol': symbol,
            'bid': tick.bid,
            'ask': tick.ask,
            'spread': tick.ask - tick.bid,
            'time': tick.time
        }
    
    def collect_price_data(self):
        """Collect current price data for all symbols"""
        for symbol in self.symbols:
            price = self.get_current_price(symbol)
            if price:
                self.price_data[symbol].append(price)
                self.latest_prices[symbol] = price
    
    def analyze_market_signal(self, symbol: str) -> Tuple[float, float]:
        """Analyze market and generate trading signal"""
        
        if len(self.price_data[symbol]) < 20:
            return 0.0, 0.0
        
        # Get recent price data
        recent_prices = list(self.price_data[symbol])[-20:]
        bid_prices = [p['bid'] for p in recent_prices]
        
        # Simple but effective signal generation
        signals = []
        
        # 1. Short-term momentum
        if len(bid_prices) >= 5:
            short_ma = np.mean(bid_prices[-3:])
            medium_ma = np.mean(bid_prices[-8:])
            momentum = (short_ma - medium_ma) / medium_ma
            signals.append(momentum * 1000)  # Scale for forex
        
        # 2. Price action pattern
        if len(bid_prices) >= 10:
            recent_high = max(bid_prices[-5:])
            recent_low = min(bid_prices[-5:])
            prev_high = max(bid_prices[-10:-5])
            prev_low = min(bid_prices[-10:-5])
            
            # Higher highs and higher lows = bullish
            if recent_high > prev_high and recent_low > prev_low:
                signals.append(0.3)
            # Lower highs and lower lows = bearish
            elif recent_high < prev_high and recent_low < prev_low:
                signals.append(-0.3)
        
        # 3. Volatility breakout
        if len(bid_prices) >= 15:
            recent_vol = np.std(bid_prices[-5:])
            avg_vol = np.std(bid_prices[-15:])
            
            if avg_vol > 0 and recent_vol > avg_vol * 1.5:
                # High volatility - trend continuation
                direction = 1 if bid_prices[-1] > bid_prices[-5] else -1
                signals.append(direction * 0.2)
        
        # Combine signals
        if not signals:
            return 0.0, 0.0
        
        direction = np.mean(signals)
        
        # Calculate confidence
        signal_strength = abs(direction)
        base_confidence = 0.35 + min(signal_strength, 0.4)
        
        # Add spread penalty (wider spreads = lower confidence)
        spread_penalty = 0
        if symbol in self.latest_prices:
            spread_pips = self.latest_prices[symbol]['spread'] / self.get_point_value(symbol)
            if spread_pips > 3:  # High spread
                spread_penalty = min((spread_pips - 3) * 0.05, 0.2)
        
        confidence = max(base_confidence - spread_penalty, 0.1)
        
        # Normalize direction
        direction = np.tanh(direction)
        
        return direction, confidence
    
    def calculate_lot_size(self, symbol: str, confidence: float) -> float:
        """Calculate optimal lot size based on risk and confidence"""
        
        if not self.account_info:
            return 0.01
        
        # Base risk calculation
        account_balance = self.account_info.equity
        risk_amount = account_balance * self.config['risk_per_trade']
        
        # Adjust for confidence
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5x to 1.0x
        adjusted_risk = risk_amount * confidence_multiplier
        
        # Calculate lot size based on stop loss
        stop_loss_pips = self.config['stop_loss_pips']
        point_value = self.get_point_value(symbol)
        
        if symbol not in self.symbol_info:
            return 0.01
        
        # Pip value calculation
        if 'JPY' in symbol:
            pip_value = self.symbol_info[symbol].trade_tick_value
        else:
            pip_value = self.symbol_info[symbol].trade_tick_value * 10
        
        # Calculate lot size
        lot_size = adjusted_risk / (stop_loss_pips * pip_value)
        
        # Apply symbol limits
        min_lot = self.symbol_info[symbol].volume_min
        max_lot = min(self.symbol_info[symbol].volume_max, 0.1)  # Cap at 0.1 lots
        
        lot_size = max(min_lot, min(lot_size, max_lot))
        
        # Round to symbol's volume step
        volume_step = self.symbol_info[symbol].volume_step
        lot_size = round(lot_size / volume_step) * volume_step
        
        return lot_size
    
    def can_open_position(self, symbol: str) -> Tuple[bool, str]:
        """Check if we can open a new position"""
        
        # Check if already have position in this symbol
        existing_positions = [p for p in self.open_positions.values() if p['symbol'] == symbol]
        if len(existing_positions) >= 1:  # Only 1 position per symbol
            return False, f"Already have position in {symbol}"
        
        # Check maximum positions
        if len(self.open_positions) >= self.config['max_positions']:
            return False, "Maximum positions reached"
        
        # Check account status
        if not self.account_info:
            return False, "No account information"
        
        # Check free margin
        required_margin = 100  # Rough estimate
        if self.account_info.margin_free < required_margin:
            return False, "Insufficient free margin"
        
        # Check if trading is allowed
        if not self.trading_enabled:
            return False, "Trading disabled"
        
        return True, "Can open position"
    
    def open_position(self, symbol: str, direction: float, confidence: float) -> bool:
        """Open real MT5 position"""
        
        print(f"\n🎯 OPENING POSITION: {symbol}")
        print(f"Direction: {direction:.3f} ({'BUY' if direction > 0 else 'SELL'})")
        print(f"Confidence: {confidence:.1%}")
        
        # Get current price
        current_price = self.get_current_price(symbol)
        if not current_price:
            print("❌ Failed to get current price")
            return False
        
        # Determine order type and price
        is_buy = direction > 0
        order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
        price = current_price['ask'] if is_buy else current_price['bid']
        
        # Calculate lot size
        lot_size = self.calculate_lot_size(symbol, confidence)
        
        # Calculate stop loss and take profit
        point = self.get_point_value(symbol)
        sl_distance = self.config['stop_loss_pips'] * point
        tp_distance = self.config['take_profit_pips'] * point
        
        if is_buy:
            sl_price = price - sl_distance
            tp_price = price + tp_distance
        else:
            sl_price = price + sl_distance
            tp_price = price - tp_distance
        
        print(f"Lot Size: {lot_size}")
        print(f"Entry Price: {price:.5f}")
        print(f"Stop Loss: {sl_price:.5f}")
        print(f"Take Profit: {tp_price:.5f}")
        
        # Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": self.config['slippage'],
            "magic": self.config['magic_number'],
            "comment": "AI Trading Agent",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        print("🔄 Sending order to MT5...")
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            print("❌ Order failed - no result returned")
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Order failed: {result.comment} (Code: {result.retcode})")
            return False
        
        # Order successful
        print("✅ ORDER EXECUTED SUCCESSFULLY!")
        print(f"Ticket: {result.order}")
        print(f"Actual Price: {result.price:.5f}")
        print(f"Volume: {result.volume}")
        
        # Store position info
        position_info = {
            'ticket': result.order,
            'symbol': symbol,
            'direction': 'BUY' if is_buy else 'SELL',
            'volume': result.volume,
            'open_price': result.price,
            'sl': sl_price,
            'tp': tp_price,
            'open_time': datetime.now(),
            'confidence': confidence
        }
        
        self.open_positions[result.order] = position_info
        self.stats['total_trades'] += 1
        
        return True
    
    def check_positions(self):
        """Check and update all open positions"""
        
        if not self.open_positions:
            return
        
        # Get current positions from MT5
        positions = mt5.positions_get(group=f"*{self.config['magic_number']}*")
        if positions is None:
            positions = []
        
        # Convert to dict for easy lookup
        mt5_positions = {pos.ticket: pos for pos in positions}
        
        # Check our tracked positions
        closed_positions = []
        
        for ticket, our_pos in self.open_positions.items():
            if ticket not in mt5_positions:
                # Position was closed
                print(f"\n🔄 POSITION CLOSED: {our_pos['symbol']} (Ticket: {ticket})")
                
                # Get closed position details from history
                history = mt5.history_deals_get(position=ticket)
                if history and len(history) >= 2:  # Open and close deals
                    close_deal = history[-1]  # Last deal is close
                    profit = close_deal.profit
                    close_price = close_deal.price
                    
                    print(f"Close Price: {close_price:.5f}")
                    print(f"Profit: ${profit:.2f}")
                    
                    # Update statistics
                    self.stats['total_profit'] += profit
                    if profit > 0:
                        self.stats['winning_trades'] += 1
                    else:
                        self.stats['losing_trades'] += 1
                    
                    # Add to trade history
                    trade_record = {
                        'ticket': ticket,
                        'symbol': our_pos['symbol'],
                        'direction': our_pos['direction'],
                        'volume': our_pos['volume'],
                        'open_price': our_pos['open_price'],
                        'close_price': close_price,
                        'open_time': our_pos['open_time'],
                        'close_time': datetime.now(),
                        'profit': profit,
                        'confidence': our_pos['confidence']
                    }
                    
                    self.trade_history.append(trade_record)
                
                closed_positions.append(ticket)
        
        # Remove closed positions from tracking
        for ticket in closed_positions:
            del self.open_positions[ticket]
    
    def print_status(self):
        """Print current trading status"""
        
        # Refresh account info
        self.account_info = mt5.account_info()
        
        runtime = (datetime.now() - self.stats['start_time']).total_seconds() / 60
        win_rate = self.stats['winning_trades'] / max(self.stats['total_trades'], 1)
        
        print(f"\n📊 MT5 TRADING STATUS - {datetime.now().strftime('%H:%M:%S')}")
        print(f"Runtime: {runtime:.1f} minutes")
        print(f"Account Balance: ${self.account_info.balance:.2f}")
        print(f"Account Equity: ${self.account_info.equity:.2f}")
        print(f"Free Margin: ${self.account_info.margin_free:.2f}")
        print(f"Open Positions: {len(self.open_positions)}")
        print(f"Total Trades: {self.stats['total_trades']}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Total Profit: ${self.stats['total_profit']:.2f}")
        
        if self.open_positions:
            print("\nOpen Positions:")
            for ticket, pos in self.open_positions.items():
                age = (datetime.now() - pos['open_time']).total_seconds() / 60
                print(f"  Ticket {ticket}: {pos['symbol']} {pos['direction']} "
                      f"{pos['volume']} lots (Age: {age:.1f}m)")
    
    async def run_trading_session(self, minutes: float = 30.0):
        """Run live MT5 trading session"""
        
        print("🚀 REAL MT5 TRADING AGENT STARTING")
        print(f"Session Duration: {minutes:.1f} minutes")
        print("=" * 50)
        
        # Connect to MT5
        if not self.connect_to_mt5():
            print("❌ Failed to connect to MT5")
            return
        
        self.running = True
        end_time = datetime.now() + timedelta(minutes=minutes)
        
        print("\n📈 Starting live trading...")
        
        cycle_count = 0
        last_status_time = datetime.now()
        
        while datetime.now() < end_time and self.running:
            try:
                cycle_count += 1
                
                # 1. Collect current market data
                self.collect_price_data()
                
                # 2. Check existing positions
                self.check_positions()
                
                # 3. Look for new trading opportunities
                for symbol in self.symbols:
                    if symbol not in self.symbol_info:
                        continue  # Skip unavailable symbols
                    
                    # Analyze market
                    direction, confidence = self.analyze_market_signal(symbol)
                    
                    # Count opportunities
                    if confidence > 0.35:
                        self.stats['opportunities_found'] += 1
                    
                    # Check if signal is strong enough
                    if confidence >= self.config['min_confidence']:
                        
                        can_open, reason = self.can_open_position(symbol)
                        
                        if can_open:
                            print(f"\n🎯 TRADING OPPORTUNITY DETECTED!")
                            print(f"Symbol: {symbol}")
                            print(f"Confidence: {confidence:.1%}")
                            print(f"Direction: {'BUY' if direction > 0 else 'SELL'}")
                            
                            # Open position
                            if self.open_position(symbol, direction, confidence):
                                print("🚀 Position opened successfully!")
                            else:
                                print("❌ Failed to open position")
                        # else:
                        #     print(f"Cannot open {symbol}: {reason}")
                
                # 4. Print status every 60 seconds
                if (datetime.now() - last_status_time).total_seconds() > 60:
                    self.print_status()
                    last_status_time = datetime.now()
                
                # 5. Wait between cycles
                await asyncio.sleep(2)  # 2-second cycle
                
            except Exception as e:
                print(f"❌ Trading cycle error: {e}")
                await asyncio.sleep(5)
        
        # Session ended
        self.running = False
        
        print("\n⏹️ Trading session ended")
        
        # Close all positions (optional)
        if self.open_positions:
            choice = input("Close all open positions? (y/n): ").lower()
            if choice == 'y':
                await self.close_all_positions()
        
        await self.generate_final_report()
        
        # Disconnect from MT5
        mt5.shutdown()
        print("🔌 Disconnected from MT5")
    
    async def close_all_positions(self):
        """Close all open positions"""
        print("\n🔄 Closing all open positions...")
        
        positions = mt5.positions_get(group=f"*{self.config['magic_number']}*")
        if not positions:
            print("No positions to close")
            return
        
        for pos in positions:
            # Prepare close request
            order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": order_type,
                "position": pos.ticket,
                "price": price,
                "deviation": self.config['slippage'],
                "magic": self.config['magic_number'],
                "comment": "Agent Close All",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ Closed {pos.symbol} position (Ticket: {pos.ticket})")
            else:
                print(f"❌ Failed to close {pos.symbol} position")
    
    async def generate_final_report(self):
        """Generate final trading report"""
        
        self.account_info = mt5.account_info()
        runtime = (datetime.now() - self.stats['start_time']).total_seconds() / 60
        
        print(f"""
🎯 REAL MT5 TRADING AGENT - FINAL REPORT
========================================
Session Duration: {runtime:.1f} minutes
Final Account Balance: ${self.account_info.balance:.2f}
Final Account Equity: ${self.account_info.equity:.2f}
Session Profit/Loss: ${self.stats['total_profit']:.2f}

TRADING STATISTICS:
Opportunities Found: {self.stats['opportunities_found']}
Total Trades: {self.stats['total_trades']}
Winning Trades: {self.stats['winning_trades']}
Losing Trades: {self.stats['losing_trades']}
Win Rate: {(self.stats['winning_trades']/max(self.stats['total_trades'],1)):.1%}
Open Positions: {len(self.open_positions)}

RECENT TRADES:""")
        
        for trade in self.trade_history[-5:]:
            duration = (trade['close_time'] - trade['open_time']).total_seconds() / 60
            print(f"  {trade['open_time'].strftime('%H:%M:%S')} | "
                  f"{trade['symbol']} {trade['direction']} | "
                  f"${trade['profit']:>7.2f} | {duration:.1f}m")
        
        # Save results
        results = {
            'session_info': {
                'start_time': self.stats['start_time'].isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_minutes': runtime,
                'final_balance': self.account_info.balance,
                'final_equity': self.account_info.equity,
                'session_profit': self.stats['total_profit']
            },
            'statistics': self.stats,
            'configuration': self.config,
            'trade_history': self.trade_history,
            'account_details': {
                'login': self.account_info.login,
                'server': self.account_info.server,
                'currency': self.account_info.currency
            }
        }
        
        filename = f'mt5_trading_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📊 Results saved to: {filename}")

if __name__ == "__main__":
    print("""
🎯 REAL MT5 TRADING AGENT
=========================
Live demo account trading with real MT5 connection

BEFORE RUNNING:
1. Ensure MT5 terminal is running
2. Update login credentials in the code if needed
3. Ensure demo account has sufficient balance

Options:
1. Quick Test (15 minutes)
2. Standard Session (30 minutes)  
3. Extended Session (60 minutes)
4. Custom Duration
    """)
    
    choice = input("Select option (1-4): ").strip()
    
    # Optional: Get custom login details
    custom_setup = input("Use custom MT5 login details? (y/n): ").lower()
    login, password, server = None, None, None
    
    if custom_setup == 'y':
        login = int(input("Enter MT5 Login: "))
        password = input("Enter MT5 Password: ")
        server = input("Enter MT5 Server: ")
    
    agent = MT5TradingAgent(login, password, server)
    
    if choice == "1":
        asyncio.run(agent.run_trading_session(minutes=15))
    elif choice == "2":
        asyncio.run(agent.run_trading_session(minutes=30))
    elif choice == "3":
        asyncio.run(agent.run_trading_session(minutes=60))
    elif choice == "4":
        duration = float(input("Enter duration in minutes: "))
        asyncio.run(agent.run_trading_session(minutes=duration))
    else:
        print("Invalid choice. Please run again.")