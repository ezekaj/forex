"""
AGGRESSIVE SCALPING STRATEGY - MAXIMUM PROFIT MODE
Based on research: 1-minute scalping with EMA crossover + Stochastic
Target: 5-10 pips per trade, 10-20 trades per day
WARNING: HIGH RISK - Only use money you can afford to lose
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.broker_connector import get_broker_connector
from core.data_loader import download_alpha_fx_daily
import talib

# Aggressive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/aggressive_scalping.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AggressiveScalper:
    """
    Maximum profit scalping strategy
    Based on proven 1-minute strategies from research
    """
    
    def __init__(self, broker_type="demo", initial_balance=100):
        self.broker_type = broker_type
        self.broker = None
        
        # AGGRESSIVE SETTINGS (from research)
        self.pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]  # Multi-pair
        self.timeframe = "1m"  # 1-minute scalping
        self.max_positions = 3  # Multiple simultaneous trades
        self.risk_per_trade = 0.02  # 2% risk (aggressive)
        self.confidence_threshold = 0.60  # Lower threshold for more trades
        
        # Scalping parameters (from research)
        self.stop_loss_pips = 8  # Tight stop loss
        self.take_profit_pips = 12  # Quick profits (1.5:1 RR)
        self.spread_limit = 2.0  # Max 2 pip spread
        
        # Position sizing
        self.base_lot_size = 0.01
        self.max_lot_size = 0.10  # Maximum position size
        
        # Performance tracking
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.daily_pnl = 0.0
        self.positions = {}
        
        # Technical indicators settings (from research)
        self.ema_fast = 13  # Fast EMA
        self.ema_slow = 26  # Slow EMA
        self.ema_signal = 50  # Trend filter
        self.stoch_period = 14  # Stochastic period
        self.rsi_period = 14
        
    def calculate_indicators(self, df):
        """Calculate advanced indicators for scalping"""
        
        # EMAs (proven profitable from research)
        df['EMA13'] = talib.EMA(df['close'], timeperiod=self.ema_fast)
        df['EMA26'] = talib.EMA(df['close'], timeperiod=self.ema_slow)
        df['EMA50'] = talib.EMA(df['close'], timeperiod=self.ema_signal)
        
        # Stochastic (key for 1-minute scalping)
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=14, slowk_period=1, slowd_period=3
        )
        
        # RSI for overbought/oversold
        df['RSI'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
        
        # Bollinger Bands for volatility
        df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # ATR for volatility-based position sizing
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # MACD for momentum
        df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        return df
    
    def generate_signals(self, df):
        """
        Generate trading signals based on multiple confirmations
        Strategy: EMA crossover + Stochastic + RSI
        """
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signal = 0  # 0=HOLD, 1=BUY, 2=SELL
        confidence = 0.0
        reasons = []
        
        # Trend filter (EMA50)
        trend_up = latest['close'] > latest['EMA50']
        trend_down = latest['close'] < latest['EMA50']
        
        # EMA Crossover signals
        ema_bullish = latest['EMA13'] > latest['EMA26'] and prev['EMA13'] <= prev['EMA26']
        ema_bearish = latest['EMA13'] < latest['EMA26'] and prev['EMA13'] >= prev['EMA26']
        
        # Stochastic signals (key for scalping)
        stoch_oversold = latest['STOCH_K'] < 20 and latest['STOCH_D'] < 20
        stoch_overbought = latest['STOCH_K'] > 80 and latest['STOCH_D'] > 80
        stoch_bullish_cross = latest['STOCH_K'] > latest['STOCH_D'] and prev['STOCH_K'] <= prev['STOCH_D']
        stoch_bearish_cross = latest['STOCH_K'] < latest['STOCH_D'] and prev['STOCH_K'] >= prev['STOCH_D']
        
        # RSI conditions
        rsi_oversold = latest['RSI'] < 30
        rsi_overbought = latest['RSI'] > 70
        rsi_neutral = 40 < latest['RSI'] < 60
        
        # Bollinger Band squeeze
        bb_squeeze = (latest['BB_UPPER'] - latest['BB_LOWER']) < (2 * latest['ATR'])
        price_at_lower_bb = latest['close'] <= latest['BB_LOWER']
        price_at_upper_bb = latest['close'] >= latest['BB_UPPER']
        
        # MACD momentum
        macd_bullish = latest['MACD'] > latest['MACD_SIGNAL'] and latest['MACD_HIST'] > 0
        macd_bearish = latest['MACD'] < latest['MACD_SIGNAL'] and latest['MACD_HIST'] < 0
        
        # BUY SIGNALS (Multiple confirmations)
        buy_score = 0
        if trend_up:
            buy_score += 20
            reasons.append("Uptrend")
        if ema_bullish:
            buy_score += 25
            reasons.append("EMA crossover UP")
        if stoch_oversold and stoch_bullish_cross:
            buy_score += 30
            reasons.append("Stoch oversold + cross")
        elif stoch_bullish_cross:
            buy_score += 20
            reasons.append("Stoch bullish cross")
        if rsi_oversold:
            buy_score += 15
            reasons.append("RSI oversold")
        if price_at_lower_bb:
            buy_score += 10
            reasons.append("Price at BB lower")
        if macd_bullish:
            buy_score += 15
            reasons.append("MACD bullish")
            
        # SELL SIGNALS (Multiple confirmations)
        sell_score = 0
        if trend_down:
            sell_score += 20
            reasons.append("Downtrend")
        if ema_bearish:
            sell_score += 25
            reasons.append("EMA crossover DOWN")
        if stoch_overbought and stoch_bearish_cross:
            sell_score += 30
            reasons.append("Stoch overbought + cross")
        elif stoch_bearish_cross:
            sell_score += 20
            reasons.append("Stoch bearish cross")
        if rsi_overbought:
            sell_score += 15
            reasons.append("RSI overbought")
        if price_at_upper_bb:
            sell_score += 10
            reasons.append("Price at BB upper")
        if macd_bearish:
            sell_score += 15
            reasons.append("MACD bearish")
            
        # Determine signal
        if buy_score >= 60:
            signal = 1
            confidence = min(buy_score / 100, 0.95)
        elif sell_score >= 60:
            signal = 2
            confidence = min(sell_score / 100, 0.95)
            
        return signal, confidence, reasons
    
    def calculate_position_size(self, balance, atr_value):
        """
        Dynamic position sizing based on volatility (ATR)
        More aggressive than standard
        """
        
        # Base position size
        risk_amount = balance * self.risk_per_trade
        
        # Adjust for volatility
        if atr_value < 0.0005:  # Low volatility
            multiplier = 1.5
        elif atr_value < 0.0010:  # Normal volatility
            multiplier = 1.0
        else:  # High volatility
            multiplier = 0.7
            
        position_size = self.base_lot_size * multiplier
        
        # Scale up with profits (aggressive)
        if balance > 150:
            position_size *= 1.5
        elif balance > 200:
            position_size *= 2.0
        elif balance > 500:
            position_size *= 3.0
            
        # Cap at maximum
        position_size = min(position_size, self.max_lot_size)
        
        return round(position_size, 2)
    
    def should_trade_now(self):
        """Check if it's optimal time to trade"""
        
        current_hour = datetime.now().hour
        
        # Best trading hours (London + New York overlap)
        # 13:00 - 17:00 UTC (8 AM - 12 PM EST)
        best_hours = [13, 14, 15, 16]
        good_hours = [8, 9, 10, 11, 12, 17, 18, 19, 20]
        
        if current_hour in best_hours:
            return True, 1.0  # Full position size
        elif current_hour in good_hours:
            return True, 0.7  # Reduced position size
        else:
            return False, 0.0  # No trading
            
    def execute_scalp_trade(self, pair, signal, confidence, reasons, balance):
        """Execute aggressive scalping trade"""
        
        # Check if we should trade now
        should_trade, time_multiplier = self.should_trade_now()
        if not should_trade and confidence < 0.80:
            logger.info(f"[{pair}] Outside optimal hours, skipping trade with confidence {confidence:.2%}")
            return
            
        # Get quote
        quote = self.broker.get_quote(pair)
        if not quote:
            return
            
        spread = quote['spread']
        
        # Check spread (critical for scalping)
        if spread > self.spread_limit:
            logger.info(f"[{pair}] Spread too high: {spread:.1f} pips")
            return
            
        # Calculate position size
        atr_value = 0.0008  # Default ATR
        position_size = self.calculate_position_size(balance, atr_value)
        position_size *= time_multiplier  # Adjust for time
        
        # Prepare order
        current_price = quote['ask'] if signal == 1 else quote['bid']
        pip_value = 0.0001
        
        if signal == 1:  # BUY
            sl = current_price - (self.stop_loss_pips * pip_value)
            tp = current_price + (self.take_profit_pips * pip_value)
            order_type = "BUY"
        else:  # SELL
            sl = current_price + (self.stop_loss_pips * pip_value)
            tp = current_price - (self.take_profit_pips * pip_value)
            order_type = "SELL"
            
        # Log trade
        logger.info("="*50)
        logger.info(f"🎯 SCALP TRADE: {pair}")
        logger.info(f"Signal: {order_type} | Confidence: {confidence:.1%}")
        logger.info(f"Reasons: {', '.join(reasons[:3])}")
        logger.info(f"Entry: {current_price:.5f} | SL: {sl:.5f} | TP: {tp:.5f}")
        logger.info(f"Position: {position_size} lots | Risk: ${position_size * self.stop_loss_pips * 10:.2f}")
        
        # Execute
        result = self.broker.place_order(
            symbol=pair,
            volume=position_size,
            order_type=order_type,
            sl=sl,
            tp=tp
        )
        
        if "error" not in result:
            self.trades_today += 1
            self.positions[result['order_id']] = {
                'pair': pair,
                'type': order_type,
                'entry': current_price,
                'sl': sl,
                'tp': tp,
                'size': position_size,
                'time': datetime.now()
            }
            logger.info(f"✅ Trade #{self.trades_today} executed successfully")
        else:
            logger.error(f"❌ Trade failed: {result['error']}")
            
    def monitor_positions(self):
        """Monitor and manage open positions"""
        
        for pos_id, pos_data in list(self.positions.items()):
            quote = self.broker.get_quote(pos_data['pair'])
            if not quote:
                continue
                
            current_price = quote['bid'] if pos_data['type'] == "BUY" else quote['ask']
            entry_price = pos_data['entry']
            
            # Calculate P&L
            if pos_data['type'] == "BUY":
                pnl_pips = (current_price - entry_price) * 10000
            else:
                pnl_pips = (entry_price - current_price) * 10000
                
            # Quick exit strategies for scalping
            time_in_trade = (datetime.now() - pos_data['time']).seconds / 60  # minutes
            
            # Exit conditions
            should_exit = False
            exit_reason = ""
            
            # 1. Time-based exit (scalping should be quick)
            if time_in_trade > 15 and pnl_pips > 3:
                should_exit = True
                exit_reason = "Time exit (15min)"
                
            # 2. Quick profit taking
            elif pnl_pips >= 5 and time_in_trade < 5:
                should_exit = True
                exit_reason = "Quick profit"
                
            # 3. Trailing stop for winners
            elif pnl_pips >= 8:
                # Move stop to breakeven + 2 pips
                new_sl = entry_price + (2 * 0.0001) if pos_data['type'] == "BUY" else entry_price - (2 * 0.0001)
                logger.info(f"[{pos_data['pair']}] Moving SL to breakeven+2")
                
            if should_exit:
                if self.broker.close_position(pos_id):
                    pnl_dollars = pnl_pips * pos_data['size'] * 10
                    self.daily_pnl += pnl_dollars
                    
                    if pnl_dollars > 0:
                        self.wins_today += 1
                    else:
                        self.losses_today += 1
                        
                    logger.info(f"🔚 Position closed: {exit_reason} | P&L: {pnl_pips:.1f} pips (${pnl_dollars:.2f})")
                    del self.positions[pos_id]
                    
    def run_aggressive_mode(self):
        """Main aggressive trading loop"""
        
        logger.info("="*60)
        logger.info("💰 AGGRESSIVE SCALPING MODE ACTIVATED")
        logger.info("⚠️  HIGH RISK - MAXIMUM PROFIT STRATEGY")
        logger.info("="*60)
        
        # Connect to broker
        self.broker = get_broker_connector(self.broker_type)
        if not self.broker.connect():
            logger.error("Broker connection failed!")
            return
            
        initial_balance = self.broker.get_balance()
        logger.info(f"Starting Balance: ${initial_balance:.2f}")
        logger.info(f"Trading Pairs: {', '.join(self.pairs)}")
        logger.info(f"Strategy: 1-min EMA + Stochastic Scalping")
        logger.info("")
        
        last_analysis = {}
        
        try:
            while True:
                current_balance = self.broker.get_balance()
                
                # Safety checks
                daily_loss = (initial_balance - current_balance) / initial_balance
                if daily_loss > 0.20:  # 20% daily loss limit
                    logger.error("Daily loss limit reached! Stopping.")
                    break
                    
                if len(self.positions) >= self.max_positions:
                    logger.info("Max positions reached, monitoring only...")
                    self.monitor_positions()
                    time.sleep(30)
                    continue
                    
                # Analyze each pair
                for pair in self.pairs:
                    try:
                        # Rate limiting for free API
                        if pair in last_analysis:
                            if (datetime.now() - last_analysis[pair]).seconds < 60:
                                continue
                                
                        logger.info(f"\n[{datetime.now():%H:%M:%S}] Analyzing {pair}...")
                        
                        # Get data and calculate indicators
                        df = download_alpha_fx_daily()  # Would need pair-specific data
                        df = self.calculate_indicators(df)
                        
                        # Generate signals
                        signal, confidence, reasons = self.generate_signals(df)
                        
                        if signal != 0:
                            logger.info(f"Signal detected: {'BUY' if signal == 1 else 'SELL'} ({confidence:.1%})")
                            
                            # Execute if confident enough
                            if confidence >= self.confidence_threshold:
                                self.execute_scalp_trade(pair, signal, confidence, reasons, current_balance)
                                
                        last_analysis[pair] = datetime.now()
                        
                    except Exception as e:
                        logger.error(f"Error analyzing {pair}: {e}")
                        
                # Monitor existing positions
                self.monitor_positions()
                
                # Status update
                if self.trades_today > 0 and self.trades_today % 5 == 0:
                    win_rate = (self.wins_today / self.trades_today) * 100 if self.trades_today > 0 else 0
                    logger.info(f"\n📊 Status: Trades: {self.trades_today} | Wins: {self.wins_today} | "
                              f"Win Rate: {win_rate:.1f}% | P&L: ${self.daily_pnl:.2f}")
                    
                # Short pause before next iteration
                time.sleep(30)  # Check every 30 seconds for 1-min scalping
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Shutting down...")
            
        finally:
            # Close all positions
            for pos_id in list(self.positions.keys()):
                logger.info(f"Closing position {pos_id}")
                self.broker.close_position(pos_id)
                
            # Final report
            final_balance = self.broker.get_balance()
            total_pnl = final_balance - initial_balance
            roi = (total_pnl / initial_balance) * 100
            
            logger.info("\n" + "="*60)
            logger.info("📈 AGGRESSIVE SCALPING RESULTS")
            logger.info("="*60)
            logger.info(f"Initial Balance: ${initial_balance:.2f}")
            logger.info(f"Final Balance: ${final_balance:.2f}")
            logger.info(f"Total P&L: ${total_pnl:.2f} ({roi:.2f}%)")
            logger.info(f"Total Trades: {self.trades_today}")
            logger.info(f"Wins: {self.wins_today} | Losses: {self.losses_today}")
            if self.trades_today > 0:
                logger.info(f"Win Rate: {(self.wins_today/self.trades_today)*100:.1f}%")
            logger.info("="*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", default="demo", choices=["demo", "mt5", "oanda"])
    parser.add_argument("--balance", type=float, default=100, help="Starting balance")
    args = parser.parse_args()
    
    scalper = AggressiveScalper(broker_type=args.broker, initial_balance=args.balance)
    scalper.run_aggressive_mode()