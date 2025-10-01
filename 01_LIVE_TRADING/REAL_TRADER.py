#!/usr/bin/env python
"""
REAL FOREX TRADER - Live Trading with Actual Money
NO MOCK DATA - REAL TRADES ONLY
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')
import sys

# Add path for new components - handle both running from main.py and directly
if os.path.exists('03_CORE_ENGINE'):
    # Running from main forex directory
    sys.path.append('03_CORE_ENGINE')
else:
    # Running from 01_LIVE_TRADING directory
    sys.path.append('../03_CORE_ENGINE')

# Import the 5 valuable components
from smart_position_sizer import SmartPositionSizer
from win_rate_optimizer import WinRateOptimizer, Trade
from market_timing_system import MarketTimingSystem
from advanced_features import AdvancedFeatureEngineering
from performance_analytics import PerformanceAnalytics

# Load environment variables
if os.path.exists('BayloZzi/.env'):
    load_dotenv('BayloZzi/.env')  # Running from main directory
else:
    load_dotenv('../BayloZzi/.env')  # Running from subdirectory

# ============================================================================
# BROKER CONNECTION - OANDA API
# ============================================================================

class OandaBroker:
    """Real broker connection for live trading"""
    
    def __init__(self):
        # OANDA API Configuration
        self.api_key = os.getenv('OANDA_API_KEY', '')
        self.account_id = os.getenv('OANDA_ACCOUNT_ID', '')
        
        # Use practice environment first, then switch to live
        self.is_live = os.getenv('TRADING_ENABLED', 'false').lower() == 'true'
        
        if self.is_live:
            self.base_url = "https://api-fxtrade.oanda.com"  # LIVE TRADING
            print("[LIVE] Connected to OANDA LIVE Trading")
        else:
            self.base_url = "https://api-fxpractice.oanda.com"  # DEMO
            print("[DEMO] Connected to OANDA Practice Account")
            
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Check if we have credentials
        if not self.api_key or not self.account_id:
            print("\n[ERROR] No OANDA credentials found!")
            print("To set up real trading:")
            print("1. Go to: https://www.oanda.com")
            print("2. Create account (practice first)")
            print("3. Get API key from: https://www.oanda.com/account/api")
            print("4. Add to BayloZzi/.env file:")
            print("   OANDA_API_KEY=your_key_here")
            print("   OANDA_ACCOUNT_ID=your_account_id")
            print("   TRADING_ENABLED=true  (for live trading)")
            self.connected = False
        else:
            self.connected = self._test_connection()
    
    def _test_connection(self):
        """Test broker connection"""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                account = response.json()['account']
                balance = float(account['balance'])
                print(f"[CONNECTED] Account Balance: {account['currency']} {balance:.2f}")
                return True
            else:
                print(f"[ERROR] Connection failed: {response.text}")
                return False
        except Exception as e:
            print(f"[ERROR] Cannot connect to broker: {e}")
            return False
    
    def get_real_price(self, pair='EUR_USD'):
        """Get REAL current market price"""
        try:
            url = f"{self.base_url}/v3/instruments/{pair}/candles"
            params = {
                'count': 1,
                'granularity': 'M1',
                'price': 'MBA'
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                candle = data['candles'][0]
                bid = float(candle['bid']['c'])
                ask = float(candle['ask']['c'])
                mid = (bid + ask) / 2
                
                return {
                    'bid': bid,
                    'ask': ask,
                    'mid': mid,
                    'spread': ask - bid,
                    'time': candle['time']
                }
            else:
                print(f"[ERROR] Cannot get price: {response.text}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Price fetch failed: {e}")
            return None
    
    def place_market_order(self, pair='EUR_USD', units=100, side='BUY'):
        """Place REAL market order"""
        if not self.connected:
            print("[ERROR] Not connected to broker")
            return None
            
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/orders"
            
            # Prepare order
            order_data = {
                'order': {
                    'type': 'MARKET',
                    'instrument': pair,
                    'units': str(units if side == 'BUY' else -units),
                    'timeInForce': 'FOK',
                    'positionFill': 'DEFAULT'
                }
            }
            
            # Add stop loss and take profit
            current_price = self.get_real_price(pair)
            if current_price:
                if side == 'BUY':
                    stop_loss = current_price['bid'] - 0.0010  # 10 pips SL
                    take_profit = current_price['ask'] + 0.0020  # 20 pips TP
                else:
                    stop_loss = current_price['ask'] + 0.0010
                    take_profit = current_price['bid'] - 0.0020
                
                order_data['order']['stopLossOnFill'] = {
                    'price': str(round(stop_loss, 5))
                }
                order_data['order']['takeProfitOnFill'] = {
                    'price': str(round(take_profit, 5))
                }
            
            # Send order
            response = requests.post(
                url, 
                headers=self.headers,
                data=json.dumps(order_data)
            )
            
            if response.status_code == 201:
                result = response.json()
                fill = result.get('orderFillTransaction', {})
                
                print(f"\n[ORDER EXECUTED]")
                print(f"  Pair: {pair}")
                print(f"  Side: {side}")
                print(f"  Units: {units}")
                print(f"  Price: {fill.get('price', 'N/A')}")
                print(f"  P&L: {fill.get('pl', '0')}")
                
                return result
            else:
                print(f"[ERROR] Order failed: {response.text}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Order execution failed: {e}")
            return None
    
    def get_open_positions(self):
        """Get all open positions"""
        if not self.connected:
            return []
            
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/openPositions"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json().get('positions', [])
            else:
                return []
        except:
            return []
    
    def close_position(self, pair='EUR_USD'):
        """Close position for a pair"""
        if not self.connected:
            return None
            
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/positions/{pair}/close"
            data = {'longUnits': 'ALL', 'shortUnits': 'ALL'}
            
            response = requests.put(
                url,
                headers=self.headers,
                data=json.dumps(data)
            )
            
            if response.status_code == 200:
                print(f"[CLOSED] Position closed for {pair}")
                return response.json()
            else:
                print(f"[ERROR] Cannot close position: {response.text}")
                return None
        except Exception as e:
            print(f"[ERROR] Close position failed: {e}")
            return None

# ============================================================================
# REAL MARKET DATA
# ============================================================================

class RealMarketData:
    """Get REAL market data from Alpha Vantage"""
    
    def __init__(self):
        self.api_key = os.getenv('ALPHAVANTAGE_API_KEY', 'KNF41ZTAUM44W2LN')
        self.calls_today = 0
        self.max_calls = 25
        
    def get_real_forex_data(self, from_symbol='EUR', to_symbol='USD'):
        """Get REAL forex data"""
        if self.calls_today >= self.max_calls:
            print(f"[WARNING] API limit reached ({self.max_calls}/day)")
            return None
            
        try:
            # Try to get intraday data first
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'FX_INTRADAY',
                'from_symbol': from_symbol,
                'to_symbol': to_symbol,
                'interval': '5min',
                'apikey': self.api_key
            }
            
            response = requests.get(url, params=params)
            self.calls_today += 1
            
            if response.status_code == 200:
                data = response.json()
                
                if 'Time Series FX (5min)' in data:
                    time_series = data['Time Series FX (5min)']
                    
                    # Convert to DataFrame
                    df = pd.DataFrame.from_dict(time_series, orient='index')
                    df.index = pd.to_datetime(df.index)
                    df = df.astype(float)
                    df.columns = ['open', 'high', 'low', 'close']
                    df = df.sort_index()
                    
                    print(f"[DATA] Got {len(df)} real data points")
                    print(f"[DATA] Latest price: {df['close'].iloc[-1]:.5f}")
                    return df
                else:
                    print("[WARNING] No intraday data, trying daily...")
                    return self.get_daily_data(from_symbol, to_symbol)
            else:
                print(f"[ERROR] API call failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Data fetch failed: {e}")
            return None
    
    def get_daily_data(self, from_symbol='EUR', to_symbol='USD'):
        """Get daily forex data as fallback"""
        try:
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'FX_DAILY',
                'from_symbol': from_symbol,
                'to_symbol': to_symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'Time Series FX (Daily)' in data:
                    time_series = data['Time Series FX (Daily)']
                    
                    df = pd.DataFrame.from_dict(time_series, orient='index')
                    df.index = pd.to_datetime(df.index)
                    df = df.astype(float)
                    df.columns = ['open', 'high', 'low', 'close']
                    df = df.sort_index()
                    
                    return df
                    
        except Exception as e:
            print(f"[ERROR] Daily data fetch failed: {e}")
            
        return None

# ============================================================================
# REAL TRADING SIGNALS
# ============================================================================

class RealTradingSignals:
    """Generate REAL trading signals from real data"""
    
    def __init__(self):
        self.last_signal = None
        self.confidence_threshold = 0.60
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices):
        """Calculate MACD"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal
    
    def get_signal(self, df):
        """Get trading signal from real data"""
        if df is None or len(df) < 50:
            return 'HOLD', 0.0
            
        # Calculate indicators
        df['rsi'] = self.calculate_rsi(df['close'])
        df['macd'], df['macd_signal'] = self.calculate_macd(df['close'])
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # Get latest values
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = []
        
        # RSI signals
        if current['rsi'] < 30:
            signals.append(('BUY', 0.7))  # Oversold
        elif current['rsi'] > 70:
            signals.append(('SELL', 0.7))  # Overbought
            
        # MACD signals
        if current['macd'] > current['macd_signal'] and prev['macd'] <= prev['macd_signal']:
            signals.append(('BUY', 0.65))  # MACD crossover
        elif current['macd'] < current['macd_signal'] and prev['macd'] >= prev['macd_signal']:
            signals.append(('SELL', 0.65))  # MACD crossunder
            
        # Moving average signals
        if current['sma_20'] > current['sma_50'] and current['close'] > current['sma_20']:
            signals.append(('BUY', 0.6))  # Uptrend
        elif current['sma_20'] < current['sma_50'] and current['close'] < current['sma_20']:
            signals.append(('SELL', 0.6))  # Downtrend
            
        # Combine signals
        if not signals:
            return 'HOLD', 0.0
            
        buy_signals = [s for s in signals if s[0] == 'BUY']
        sell_signals = [s for s in signals if s[0] == 'SELL']
        
        buy_confidence = sum(s[1] for s in buy_signals) / len(signals) if buy_signals else 0
        sell_confidence = sum(s[1] for s in sell_signals) / len(signals) if sell_signals else 0
        
        if buy_confidence > sell_confidence and buy_confidence >= self.confidence_threshold:
            return 'BUY', buy_confidence
        elif sell_confidence > buy_confidence and sell_confidence >= self.confidence_threshold:
            return 'SELL', sell_confidence
        else:
            return 'HOLD', max(buy_confidence, sell_confidence)

# ============================================================================
# AUTOMATIC TRADER - MAIN ENGINE
# ============================================================================

class AutomaticForexTrader:
    """Fully automatic forex trading system with advanced components"""
    
    def __init__(self, initial_capital=10.0):
        self.capital = initial_capital
        self.broker = OandaBroker()
        self.market_data = RealMarketData()
        self.signal_generator = RealTradingSignals()
        
        # Initialize the 5 valuable components
        self.position_sizer = SmartPositionSizer(
            risk_per_trade=0.02,  # 2% risk per trade
            max_position_size=0.10,  # 10% max position
            min_position_size=0.01   # 1% min position
        )
        
        self.win_optimizer = WinRateOptimizer(target_win_rate=0.65)
        self.timing_system = MarketTimingSystem()
        self.feature_engineer = AdvancedFeatureEngineering()
        self.performance_tracker = PerformanceAnalytics(initial_capital=initial_capital)
        
        self.position_size = 100  # Base position size
        self.max_positions = 3
        self.current_positions = []
        self.trades_executed = []
        self.recent_performance = []  # Track recent trade outcomes
        
        print("\n" + "="*60)
        print("AUTOMATIC FOREX TRADER - ENHANCED WITH AI COMPONENTS")
        print("="*60)
        print(f"Initial Capital: EUR {self.capital}")
        print(f"Base Position Size: {self.position_size} units")
        print(f"Max Positions: {self.max_positions}")
        print("\n[COMPONENTS] Advanced Systems Loaded:")
        print("  [OK] Smart Position Sizer")
        print("  [OK] Win Rate Optimizer")
        print("  [OK] Market Timing System")
        print("  [OK] Advanced Feature Engineering")
        print("  [OK] Performance Analytics")
        
        if self.broker.connected:
            print("\n[STATUS] Ready to trade with REAL money")
        else:
            print("\n[WARNING] Broker not connected - trades won't execute")
        print("="*60)
    
    def analyze_and_trade(self, pair='EUR_USD'):
        """Analyze market and execute trades automatically with advanced components"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Analyzing {pair}...")
        
        # 1. CHECK MARKET TIMING (Component #3)
        should_trade, timing_reason = self.timing_system.should_trade_now(pair)
        market_status = self.timing_system.get_current_market_status()
        
        print(f"[TIMING] Liquidity Score: {market_status['liquidity_score']:.2f}")
        print(f"[TIMING] {timing_reason}")
        
        if not should_trade:
            print(f"[SKIP] Poor market timing: {timing_reason}")
            return
        
        # Get REAL market data
        from_symbol = pair[:3]
        to_symbol = pair[4:]
        df = self.market_data.get_real_forex_data(from_symbol, to_symbol)
        
        if df is None:
            print("[ERROR] No market data available")
            return
        
        # 2. APPLY ADVANCED FEATURE ENGINEERING (Component #4)
        df_enhanced = self.feature_engineer.engineer_all_features(df)
        
        # Get trading signal with enhanced features
        signal, confidence = self.signal_generator.get_signal(df_enhanced)
        
        # 3. OPTIMIZE TRADE DECISION (Component #2)
        # Calculate current volatility
        volatility = df_enhanced['volatility_ratio'].iloc[-1] if 'volatility_ratio' in df_enhanced else 0.01
        
        trade_decision = self.win_optimizer.get_trade_decision(
            signal_confidence=confidence,
            market_volatility=volatility,
            recent_performance=self.recent_performance[-3:] if self.recent_performance else None
        )
        
        if not trade_decision['trade']:
            print(f"[OPTIMIZER] Trade rejected: {trade_decision['reason']}")
            return
        
        print(f"[SIGNAL] {signal} with {confidence:.1%} confidence")
        print(f"[OPTIMIZER] Trade approved with adjustments")
        
        # Check if we should trade
        if signal == 'HOLD':
            print("[ACTION] No trade - holding position")
            return
        
        # Check current positions
        if self.broker.connected:
            positions = self.broker.get_open_positions()
            if len(positions) >= self.max_positions:
                print(f"[WARNING] Max positions reached ({len(positions)}/{self.max_positions})")
                return
        
        # Execute trade with optimized parameters
        if signal in ['BUY', 'SELL']:
            self.execute_trade(pair, signal, confidence, trade_decision, df_enhanced)
    
    def execute_trade(self, pair, signal, confidence, trade_decision, df):
        """Execute REAL trade with Smart Position Sizing"""
        print(f"\n[EXECUTING] {signal} order for {pair}")
        
        # Get current price
        if self.broker.connected:
            price_data = self.broker.get_real_price(pair)
            if price_data:
                current_price = price_data['mid']
                print(f"[PRICE] Bid: {price_data['bid']:.5f} | Ask: {price_data['ask']:.5f}")
                print(f"[SPREAD] {price_data['spread']:.5f}")
        else:
            current_price = df['close'].iloc[-1]
        
        # 4. CALCULATE SMART POSITION SIZE (Component #1)
        # Calculate stop loss and take profit from optimizer
        stop_loss_pips = trade_decision.get('stop_loss_pips', 20)
        take_profit_pips = trade_decision.get('take_profit_pips', 40)
        
        if signal == 'BUY':
            stop_loss = current_price - (stop_loss_pips * 0.0001)
            take_profit = current_price + (take_profit_pips * 0.0001)
        else:  # SELL
            stop_loss = current_price + (stop_loss_pips * 0.0001)
            take_profit = current_price - (take_profit_pips * 0.0001)
        
        # Calculate volatility
        volatility = df['volatility_ratio'].iloc[-1] if 'volatility_ratio' in df else 0.01
        
        # Use Smart Position Sizer
        position_info = self.position_sizer.calculate_position_size(
            account_equity=self.capital,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            volatility=volatility,
            confidence=confidence,
            market_condition='normal'
        )
        
        adjusted_size = position_info['units']
        
        print(f"[POSITION SIZER] Calculated size: {adjusted_size} units")
        print(f"[RISK] {position_info['risk_percentage']:.2f}% of capital")
        print(f"[RECOMMENDATION] {position_info['recommendation']}")
        
        # Place order with calculated parameters
        if self.broker.connected:
            result = self.broker.place_market_order(
                pair=pair,
                units=adjusted_size,
                side=signal
            )
            
            if result:
                # Record trade for performance tracking
                trade_record = {
                    'timestamp': datetime.now(),
                    'pair': pair,
                    'signal': signal,
                    'confidence': confidence,
                    'size': adjusted_size,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'result': result
                }
                
                self.trades_executed.append(trade_record)
                
                # 5. TRACK PERFORMANCE (Component #5)
                self.performance_tracker.record_trade({
                    'timestamp': datetime.now(),
                    'pair': pair,
                    'direction': signal,
                    'entry_price': current_price,
                    'position_size': adjusted_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                })
                
                print("[SUCCESS] Trade executed!")
            else:
                print("[FAILED] Trade execution failed")
                self.recent_performance.append(False)
        else:
            print("[SIMULATION] Would execute:")
            print(f"  - {signal} {adjusted_size} units of {pair}")
            print(f"  - Entry: {current_price:.5f}")
            print(f"  - Stop Loss: {stop_loss:.5f}")
            print(f"  - Take Profit: {take_profit:.5f}")
            print(f"  - Risk: {position_info['risk_percentage']:.2f}%")
            print(f"  - Confidence: {confidence:.1%}")
    
    def monitor_positions(self):
        """Monitor open positions"""
        if not self.broker.connected:
            return
            
        positions = self.broker.get_open_positions()
        
        if positions:
            print(f"\n[POSITIONS] {len(positions)} open positions:")
            for pos in positions:
                instrument = pos.get('instrument', 'N/A')
                units = pos.get('long', {}).get('units', 0) or pos.get('short', {}).get('units', 0)
                pl = pos.get('unrealizedPL', 0)
                print(f"  - {instrument}: {units} units | P&L: {pl}")
        else:
            print("[POSITIONS] No open positions")
    
    def run_automatic_trading(self, hours=1, interval_minutes=5):
        """Run fully automatic trading"""
        print(f"\n[STARTING] Automatic trading for {hours} hour(s)")
        print(f"[INTERVAL] Checking every {interval_minutes} minutes")
        print("[CTRL+C] Press Ctrl+C to stop\n")
        
        end_time = datetime.now() + timedelta(hours=hours)
        
        try:
            while datetime.now() < end_time:
                # Analyze and trade multiple pairs
                pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY']
                
                for pair in pairs:
                    self.analyze_and_trade(pair)
                    time.sleep(2)  # Small delay between pairs
                
                # Monitor positions
                self.monitor_positions()
                
                # Show summary
                if self.trades_executed:
                    print(f"\n[SUMMARY] {len(self.trades_executed)} trades executed today")
                
                # Wait for next interval
                print(f"\n[WAITING] Next check in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\n[STOPPED] Trading stopped by user")
        
        # Final report
        self.print_final_report()
    
    def print_final_report(self):
        """Print comprehensive trading report with performance analytics"""
        print("\n" + "="*60)
        print("ENHANCED TRADING SESSION REPORT")
        print("="*60)
        
        if self.trades_executed:
            print(f"Total Trades: {len(self.trades_executed)}")
            
            for i, trade in enumerate(self.trades_executed, 1):
                print(f"\nTrade #{i}:")
                print(f"  Time: {trade['timestamp'].strftime('%H:%M:%S')}")
                print(f"  Pair: {trade['pair']}")
                print(f"  Signal: {trade['signal']}")
                print(f"  Confidence: {trade['confidence']:.1%}")
                print(f"  Size: {trade['size']} units")
                print(f"  Entry: {trade['entry_price']:.5f}")
                print(f"  Stop Loss: {trade['stop_loss']:.5f}")
                print(f"  Take Profit: {trade['take_profit']:.5f}")
        else:
            print("No trades executed")
        
        # Check final positions
        if self.broker.connected:
            self.monitor_positions()
        
        # Print Performance Analytics Report
        print("\n" + "-"*60)
        print("PERFORMANCE ANALYTICS")
        print("-"*60)
        print(self.performance_tracker.generate_report())
        
        # Print Win Rate Optimizer Report
        print("\n" + "-"*60)
        print("WIN RATE OPTIMIZATION")
        print("-"*60)
        print(self.win_optimizer.generate_report())
        
        # Print Market Timing Report
        print("\n" + "-"*60)
        print("MARKET TIMING ANALYSIS")
        print("-"*60)
        print(self.timing_system.generate_timing_report())
        
        print("="*60)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automatic Forex Trader')
    parser.add_argument('--capital', type=float, default=10.0,
                       help='Starting capital')
    parser.add_argument('--hours', type=float, default=1,
                       help='Trading duration in hours')
    parser.add_argument('--interval', type=int, default=5,
                       help='Check interval in minutes')
    parser.add_argument('--setup', action='store_true',
                       help='Show setup instructions')
    
    args = parser.parse_args()
    
    if args.setup:
        print("\n" + "="*60)
        print("SETUP INSTRUCTIONS")
        print("="*60)
        print("\n1. CREATE OANDA ACCOUNT:")
        print("   - Go to: https://www.oanda.com")
        print("   - Create Practice Account (free)")
        print("   - Get API key from account settings")
        print("\n2. ADD TO BayloZzi/.env FILE:")
        print("   OANDA_API_KEY=your_api_key")
        print("   OANDA_ACCOUNT_ID=your_account_id")
        print("   TRADING_ENABLED=false  (use 'true' for live)")
        print("\n3. RUN AUTOMATIC TRADING:")
        print("   python REAL_TRADER.py --hours 1")
        print("="*60)
        return
    
    # Initialize trader
    trader = AutomaticForexTrader(initial_capital=args.capital)
    
    # Run automatic trading
    trader.run_automatic_trading(
        hours=args.hours,
        interval_minutes=args.interval
    )

if __name__ == "__main__":
    main()