"""
Safe Live Trading Script with Multiple Broker Support
Start with minimal amounts to test the system
"""

import os
import sys
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.broker_connector import get_broker_connector
from core.model import load_model
from core.features import add_features
from core.data_loader import download_alpha_fx_daily
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SafeLiveTrader:
    """Safe live trading with multiple safety checks"""
    
    def __init__(self, broker_type="demo", symbol="EURUSD", max_risk=0.01):
        """
        Initialize trader
        
        broker_type: "demo", "mt5", "oanda"
        symbol: Trading pair
        max_risk: Maximum risk per trade (0.01 = 1%)
        """
        self.broker_type = broker_type
        self.symbol = symbol
        self.max_risk = max_risk
        
        # Risk parameters
        self.max_daily_loss = 0.05  # 5% max daily loss
        self.max_positions = 1  # Only 1 position at a time
        self.min_balance = 10.0  # Stop if balance < $10
        self.confidence_threshold = 0.70  # 70% confidence required
        
        # Trading parameters
        self.stop_loss_pips = 20  # Conservative stop loss
        self.take_profit_pips = 40  # 1:2 risk/reward ratio
        self.lot_size = 0.01  # Micro lot (1000 units)
        
        # Initialize components
        self.broker = None
        self.model = None
        self.daily_pnl = 0.0
        self.daily_start_balance = 0.0
        self.current_position = None
        
        # Safety flags
        self.emergency_stop = False
        self.trading_enabled = os.getenv("TRADING_ENABLED", "false").lower() == "true"
        
    def initialize(self):
        """Initialize broker connection and model"""
        logger.info("="*60)
        logger.info("SAFE LIVE TRADING SYSTEM")
        logger.info("="*60)
        
        # Safety check
        if not self.trading_enabled:
            logger.warning("TRADING_ENABLED is false in .env - Running in observation mode only")
            
        # Connect to broker
        logger.info(f"Connecting to {self.broker_type} broker...")
        self.broker = get_broker_connector(self.broker_type)
        
        if not self.broker.connect():
            logger.error("Failed to connect to broker!")
            return False
            
        # Get initial balance
        balance = self.broker.get_balance()
        self.daily_start_balance = balance
        logger.info(f"Account balance: ${balance:.2f}")
        
        # Safety check: minimum balance
        if balance < self.min_balance:
            logger.error(f"Balance ${balance:.2f} is below minimum ${self.min_balance}")
            return False
            
        # Load ML model
        logger.info("Loading ML model...")
        self.model = load_model()
        
        logger.info("Initialization complete!")
        return True
        
    def calculate_position_size(self, stop_loss_pips: float) -> float:
        """Calculate safe position size based on risk"""
        balance = self.broker.get_balance()
        
        # Calculate risk amount
        risk_amount = balance * self.max_risk
        
        # Calculate position size (simplified)
        # In forex: 1 pip = $0.10 for 0.01 lot
        pip_value = 0.10  # For micro lot
        position_size = risk_amount / (stop_loss_pips * pip_value)
        
        # Limit to micro lots
        position_size = min(position_size, 0.01)
        
        return round(position_size, 2)
        
    def check_safety_conditions(self) -> bool:
        """Check if it's safe to trade"""
        
        # Check emergency stop
        if self.emergency_stop:
            logger.error("Emergency stop activated!")
            return False
            
        # Check trading enabled
        if not self.trading_enabled:
            logger.info("Trading disabled - observation mode only")
            return False
            
        # Check balance
        balance = self.broker.get_balance()
        if balance < self.min_balance:
            logger.error(f"Balance too low: ${balance:.2f}")
            self.emergency_stop = True
            return False
            
        # Check daily loss limit
        daily_loss = (self.daily_start_balance - balance) / self.daily_start_balance
        if daily_loss > self.max_daily_loss:
            logger.error(f"Daily loss limit reached: {daily_loss:.1%}")
            self.emergency_stop = True
            return False
            
        # Check if we already have a position
        if self.current_position is not None:
            logger.info("Already have an open position")
            return False
            
        return True
        
    def get_market_data(self) -> pd.DataFrame:
        """Get latest market data"""
        try:
            # Download latest data from Alpha Vantage
            df = download_alpha_fx_daily()
            
            # Add technical indicators
            df = add_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return None
            
    def make_prediction(self, df: pd.DataFrame) -> tuple:
        """Make trading prediction"""
        try:
            # Get latest data
            latest = df.iloc[-1]
            
            # Prepare features (must match training)
            features = [
                'open','high','low','close','volume',
                'MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'MACD_signal',
                'BB_upper', 'BB_lower', 'Momentum', 'Volatility',
                'Volume_MA', 'Price_Range', 'Prev_Direction',
                'Resistance', 'Support'
            ]
            
            X = latest[features].values.reshape(1, -1)
            
            # Get prediction
            proba = self.model.predict_proba(X)[0]
            prediction = proba.argmax()
            confidence = proba.max()
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0, 0.0
            
    def execute_trade(self, signal: int, confidence: float):
        """Execute trade with safety checks"""
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            logger.info(f"Confidence {confidence:.2%} below threshold {self.confidence_threshold:.2%}")
            return
            
        # Get current quote
        quote = self.broker.get_quote(self.symbol)
        if not quote:
            logger.error("Failed to get quote")
            return
            
        current_price = quote["ask"] if signal == 1 else quote["bid"]
        spread = quote["spread"]
        
        # Check spread (avoid high spread)
        if spread > 3.0:  # 3 pips max
            logger.warning(f"Spread too high: {spread:.1f} pips")
            return
            
        # Calculate position size
        position_size = self.calculate_position_size(self.stop_loss_pips)
        
        # Calculate SL and TP
        pip_value = 0.0001  # For EUR/USD
        
        if signal == 1:  # BUY
            stop_loss = current_price - (self.stop_loss_pips * pip_value)
            take_profit = current_price + (self.take_profit_pips * pip_value)
            order_type = "BUY"
        else:  # SELL
            stop_loss = current_price + (self.stop_loss_pips * pip_value)
            take_profit = current_price - (self.take_profit_pips * pip_value)
            order_type = "SELL"
            
        # Log trade details
        logger.info("="*40)
        logger.info(f"TRADE SIGNAL: {order_type}")
        logger.info(f"Confidence: {confidence:.2%}")
        logger.info(f"Price: {current_price:.5f}")
        logger.info(f"Stop Loss: {stop_loss:.5f} ({self.stop_loss_pips} pips)")
        logger.info(f"Take Profit: {take_profit:.5f} ({self.take_profit_pips} pips)")
        logger.info(f"Position Size: {position_size} lots")
        logger.info(f"Risk: ${position_size * self.stop_loss_pips * 10:.2f}")
        logger.info("="*40)
        
        # Final confirmation
        if self.trading_enabled:
            # Place the order
            result = self.broker.place_order(
                symbol=self.symbol,
                volume=position_size,
                order_type=order_type,
                sl=stop_loss,
                tp=take_profit
            )
            
            if "error" in result:
                logger.error(f"Order failed: {result['error']}")
            else:
                self.current_position = result
                logger.info(f"✓ Order placed successfully: {result}")
        else:
            logger.info(">> DEMO MODE: Order not placed (set TRADING_ENABLED=true to trade)")
            
    def monitor_position(self):
        """Monitor open position"""
        if self.current_position is None:
            return
            
        # Get current quote
        quote = self.broker.get_quote(self.symbol)
        if not quote:
            return
            
        current_price = quote["bid"] if self.current_position["type"] == "BUY" else quote["ask"]
        entry_price = self.current_position["price"]
        
        # Calculate current P&L
        if self.current_position["type"] == "BUY":
            pnl_pips = (current_price - entry_price) * 10000
        else:
            pnl_pips = (entry_price - current_price) * 10000
            
        pnl_dollars = pnl_pips * self.current_position["volume"] * 10
        
        logger.info(f"Position Update: {self.current_position['type']} @ {entry_price:.5f} | "
                   f"Current: {current_price:.5f} | P&L: {pnl_pips:.1f} pips (${pnl_dollars:.2f})")
                   
    def run(self):
        """Main trading loop"""
        
        if not self.initialize():
            logger.error("Initialization failed!")
            return
            
        logger.info("\nStarting trading loop...")
        logger.info("Press Ctrl+C to stop\n")
        
        try:
            while not self.emergency_stop:
                try:
                    # Check safety conditions
                    if not self.check_safety_conditions():
                        # Just monitor if we can't trade
                        self.monitor_position()
                        time.sleep(60)  # Wait 1 minute
                        continue
                        
                    # Get market data
                    logger.info(f"\n[{datetime.now():%H:%M:%S}] Analyzing market...")
                    df = self.get_market_data()
                    if df is None or df.empty:
                        logger.warning("No market data available")
                        time.sleep(300)  # Wait 5 minutes
                        continue
                        
                    # Make prediction
                    signal, confidence = self.make_prediction(df)
                    
                    # Log prediction
                    signal_name = "BUY" if signal == 1 else "SELL" if signal == 2 else "HOLD"
                    logger.info(f"Signal: {signal_name} (Confidence: {confidence:.2%})")
                    
                    # Execute trade if conditions are met
                    if signal != 0:  # 0 = HOLD
                        self.execute_trade(signal, confidence)
                        
                    # Wait before next iteration
                    # In production, you might want to sync with market hours
                    time.sleep(300)  # Check every 5 minutes
                    
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    time.sleep(60)
                    
        except KeyboardInterrupt:
            logger.info("\nShutting down gracefully...")
            
        finally:
            # Close any open positions
            if self.current_position:
                logger.info("Closing open position...")
                self.broker.close_position(self.current_position["order_id"])
                
            # Final report
            final_balance = self.broker.get_balance()
            total_pnl = final_balance - self.daily_start_balance
            
            logger.info("\n" + "="*60)
            logger.info("TRADING SESSION SUMMARY")
            logger.info("="*60)
            logger.info(f"Starting Balance: ${self.daily_start_balance:.2f}")
            logger.info(f"Final Balance: ${final_balance:.2f}")
            logger.info(f"Total P&L: ${total_pnl:.2f} ({(total_pnl/self.daily_start_balance)*100:.2f}%)")
            logger.info("="*60)


def main():
    """Main entry point"""
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Safe Live Forex Trading")
    parser.add_argument("--broker", choices=["demo", "mt5", "oanda"], 
                       default="demo", help="Broker to use")
    parser.add_argument("--symbol", default="EURUSD", help="Trading symbol")
    parser.add_argument("--risk", type=float, default=0.01, 
                       help="Max risk per trade (0.01 = 1%)")
    
    args = parser.parse_args()
    
    # Create and run trader
    trader = SafeLiveTrader(
        broker_type=args.broker,
        symbol=args.symbol,
        max_risk=args.risk
    )
    
    trader.run()


if __name__ == "__main__":
    main()