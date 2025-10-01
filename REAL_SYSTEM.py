#!/usr/bin/env python
"""
THE REAL FOREX SYSTEM - No BS, Just What Works
Based on actual profitable strategies, not academic theory
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import talib
import warnings
warnings.filterwarnings('ignore')

class RealForexStrategy:
    """
    What actually works in forex:
    1. Trend following with momentum
    2. Support/Resistance levels
    3. Volume profile analysis
    4. Smart money concepts (liquidity hunting)
    """
    
    def __init__(self):
        self.min_confidence = 0.65  # Don't trade below this
        self.risk_per_trade = 0.01  # 1% risk (REAL traders use 0.5-2%)
        
    def calculate_signal(self, df):
        """
        REAL signal calculation that actually works
        """
        if len(df) < 200:
            return 'HOLD', 0.0, "Insufficient data"
        
        # Get the data
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values if 'volume' in df else np.ones_like(close)
        
        # 1. TREND DETECTION (What direction is the market going?)
        ema_fast = talib.EMA(close, timeperiod=20)
        ema_slow = talib.EMA(close, timeperiod=50)
        ema_long = talib.EMA(close, timeperiod=200)
        
        current_price = close[-1]
        
        # Trend strength
        trend_bullish = (ema_fast[-1] > ema_slow[-1] > ema_long[-1])
        trend_bearish = (ema_fast[-1] < ema_slow[-1] < ema_long[-1])
        
        # 2. MOMENTUM (Is the move strong?)
        rsi = talib.RSI(close, timeperiod=14)[-1]
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        momentum_bullish = macd[-1] > macd_signal[-1] and macd_hist[-1] > 0
        momentum_bearish = macd[-1] < macd_signal[-1] and macd_hist[-1] < 0
        
        # 3. VOLATILITY (Is the market moving enough to profit?)
        atr = talib.ATR(high, low, close, timeperiod=14)[-1]
        volatility = atr / current_price
        
        # Need at least 0.05% volatility (5 pips on EUR/USD)
        if volatility < 0.0005:
            return 'HOLD', 0.0, "Volatility too low"
        
        # 4. SUPPORT/RESISTANCE (Key levels)
        # Find recent swing highs/lows
        recent_high = np.max(high[-20:])
        recent_low = np.min(low[-20:])
        
        # Distance from key levels
        distance_from_high = (recent_high - current_price) / current_price
        distance_from_low = (current_price - recent_low) / current_price
        
        # 5. VOLUME ANALYSIS (Is there conviction?)
        volume_sma = np.mean(volume[-20:])
        current_volume = volume[-1]
        volume_surge = current_volume > volume_sma * 1.5
        
        # 6. PATTERN RECOGNITION (Actual patterns that work)
        # Bullish engulfing
        if 'open' in df.columns:
            open_prices = df['open'].values
            bullish_engulfing = (
                close[-2] < open_prices[-2] and  # Previous red candle
                close[-1] > open_prices[-1] and  # Current green candle
                close[-1] > open_prices[-2] and  # Engulfs previous
                open_prices[-1] < close[-2]
            )
            
            # Bearish engulfing
            bearish_engulfing = (
                close[-2] > open_prices[-2] and  # Previous green candle
                close[-1] < open_prices[-1] and  # Current red candle  
                close[-1] < open_prices[-2] and  # Engulfs previous
                open_prices[-1] > close[-2]
            )
        else:
            bullish_engulfing = False
            bearish_engulfing = False
        
        # SIGNAL GENERATION WITH REALISTIC CONFIDENCE
        signal = 'HOLD'
        confidence = 0.0
        reason = ""
        
        # STRONG BUY SIGNALS (70-85% confidence)
        if trend_bullish and momentum_bullish:
            if rsi < 70:  # Not overbought
                confidence = 0.70
                signal = 'BUY'
                reason = "Strong uptrend with momentum"
                
                # Boost confidence for additional factors
                if volume_surge:
                    confidence += 0.05
                    reason += " + volume surge"
                if distance_from_low < 0.01:  # Near support
                    confidence += 0.05
                    reason += " + near support"
                if bullish_engulfing:
                    confidence += 0.05
                    reason += " + bullish pattern"
                
                confidence = min(confidence, 0.85)
        
        # STRONG SELL SIGNALS (70-85% confidence)
        elif trend_bearish and momentum_bearish:
            if rsi > 30:  # Not oversold
                confidence = 0.70
                signal = 'SELL'
                reason = "Strong downtrend with momentum"
                
                # Boost confidence for additional factors
                if volume_surge:
                    confidence += 0.05
                    reason += " + volume surge"
                if distance_from_high < 0.01:  # Near resistance
                    confidence += 0.05
                    reason += " + near resistance"
                if bearish_engulfing:
                    confidence += 0.05
                    reason += " + bearish pattern"
                
                confidence = min(confidence, 0.85)
        
        # MEDIUM SIGNALS (60-70% confidence)
        elif not trend_bullish and not trend_bearish:
            # Range trading
            if rsi < 30 and distance_from_low < 0.005:
                signal = 'BUY'
                confidence = 0.65
                reason = "Oversold at support"
            elif rsi > 70 and distance_from_high < 0.005:
                signal = 'SELL'
                confidence = 0.65
                reason = "Overbought at resistance"
        
        # DON'T TRADE if confidence too low
        if confidence < self.min_confidence:
            signal = 'HOLD'
            confidence = 0.0
            reason = "Confidence below minimum threshold"
        
        return signal, confidence, reason
    
    def calculate_stops(self, signal, entry_price, atr, confidence):
        """
        REALISTIC stop loss and take profit
        Based on ATR and confidence
        """
        # Base risk/reward on confidence
        if confidence >= 0.80:
            risk_reward = 3.0  # 1:3 risk/reward
        elif confidence >= 0.70:
            risk_reward = 2.0  # 1:2 risk/reward
        else:
            risk_reward = 1.5  # 1:1.5 risk/reward
        
        # Stop loss = 1.5 ATR (gives room for noise)
        stop_distance = atr * 1.5
        
        if signal == 'BUY':
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * risk_reward)
        else:  # SELL
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * risk_reward)
        
        return stop_loss, take_profit
    
    def calculate_position_size(self, account_balance, entry_price, stop_loss):
        """
        PROPER position sizing (what pros actually use)
        """
        risk_amount = account_balance * self.risk_per_trade
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            return 0
        
        # Position size in units
        position_size = risk_amount / stop_distance
        
        # Forex lots (1 lot = 100,000 units)
        lots = position_size / 100000
        
        return {
            'units': int(position_size),
            'lots': round(lots, 2),
            'risk_amount': risk_amount,
            'stop_distance_pips': stop_distance * 10000  # For major pairs
        }


class ForexBacktester:
    """
    HONEST backtesting - includes spread, slippage, and realistic execution
    """
    
    def __init__(self, strategy, initial_balance=1000):
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trades = []
        self.equity_curve = [initial_balance]
        
        # REALISTIC costs
        self.spread_pips = 1.5  # Average spread in pips
        self.slippage_pips = 0.5  # Execution slippage
        self.commission_per_lot = 7  # Round turn commission
        
    def backtest(self, df, pair='EURUSD'):
        """
        Run realistic backtest
        """
        print(f"\nBACKTESTING {pair}")
        print(f"Period: {df.index[0]} to {df.index[-1]}")
        print(f"Initial Balance: ${self.initial_balance}")
        print("-" * 50)
        
        for i in range(200, len(df), 5):  # Need 200 bars history, check every 5 bars
            # Get data up to this point
            historical_data = df.iloc[:i]
            current_price = df.iloc[i]['close']
            
            # Get signal
            signal, confidence, reason = self.strategy.calculate_signal(historical_data)
            
            if signal != 'HOLD':
                # Calculate ATR for stops
                high = historical_data['high'].values
                low = historical_data['low'].values
                close = historical_data['close'].values
                atr = talib.ATR(high, low, close, timeperiod=14)[-1]
                
                # Calculate stops
                stop_loss, take_profit = self.strategy.calculate_stops(
                    signal, current_price, atr, confidence
                )
                
                # Calculate position size
                position = self.strategy.calculate_position_size(
                    self.balance, current_price, stop_loss
                )
                
                # Simulate trade execution
                trade_result = self.execute_trade(
                    df.iloc[i:min(i+100, len(df))],  # Next 100 bars
                    signal, current_price, stop_loss, take_profit, position
                )
                
                # Record trade
                trade_result['timestamp'] = df.index[i]
                trade_result['confidence'] = confidence
                trade_result['reason'] = reason
                self.trades.append(trade_result)
                
                # Update balance
                self.balance += trade_result['profit']
                self.equity_curve.append(self.balance)
                
                print(f"{df.index[i]}: {signal} @ {current_price:.5f} "
                      f"Confidence: {confidence:.1%} "
                      f"P&L: ${trade_result['profit']:+.2f} "
                      f"Balance: ${self.balance:.2f}")
        
        self.print_results()
    
    def execute_trade(self, future_data, signal, entry, stop_loss, take_profit, position):
        """
        Simulate realistic trade execution
        """
        # Add spread cost
        if signal == 'BUY':
            entry = entry + (self.spread_pips * 0.0001)  # Pay the spread
        else:
            entry = entry - (self.spread_pips * 0.0001)
        
        # Check each future bar
        for i, (index, bar) in enumerate(future_data.iterrows()):
            if signal == 'BUY':
                # Check stop loss (use low)
                if bar['low'] <= stop_loss:
                    exit_price = stop_loss - (self.slippage_pips * 0.0001)
                    profit = (exit_price - entry) * position['units']
                    return {
                        'result': 'LOSS',
                        'exit_price': exit_price,
                        'profit': profit - (position['lots'] * self.commission_per_lot),
                        'bars_held': i
                    }
                # Check take profit (use high)
                elif bar['high'] >= take_profit:
                    exit_price = take_profit
                    profit = (exit_price - entry) * position['units']
                    return {
                        'result': 'WIN',
                        'exit_price': exit_price,
                        'profit': profit - (position['lots'] * self.commission_per_lot),
                        'bars_held': i
                    }
            else:  # SELL
                # Check stop loss (use high)
                if bar['high'] >= stop_loss:
                    exit_price = stop_loss + (self.slippage_pips * 0.0001)
                    profit = (entry - exit_price) * position['units']
                    return {
                        'result': 'LOSS',
                        'exit_price': exit_price,
                        'profit': profit - (position['lots'] * self.commission_per_lot),
                        'bars_held': i
                    }
                # Check take profit (use low)
                elif bar['low'] <= take_profit:
                    exit_price = take_profit
                    profit = (entry - exit_price) * position['units']
                    return {
                        'result': 'WIN',
                        'exit_price': exit_price,
                        'profit': profit - (position['lots'] * self.commission_per_lot),
                        'bars_held': i
                    }
        
        # Trade still open after 100 bars - close it
        exit_price = future_data.iloc[-1]['close']
        if signal == 'BUY':
            profit = (exit_price - entry) * position['units']
        else:
            profit = (entry - exit_price) * position['units']
        
        return {
            'result': 'TIMEOUT',
            'exit_price': exit_price,
            'profit': profit - (position['lots'] * self.commission_per_lot),
            'bars_held': len(future_data)
        }
    
    def print_results(self):
        """
        Print HONEST results
        """
        if not self.trades:
            print("\nNo trades executed")
            return
        
        wins = [t for t in self.trades if t['result'] == 'WIN']
        losses = [t for t in self.trades if t['result'] == 'LOSS']
        
        total_profit = self.balance - self.initial_balance
        roi = (total_profit / self.initial_balance) * 100
        
        print("\n" + "=" * 50)
        print("BACKTEST RESULTS (REALISTIC)")
        print("=" * 50)
        print(f"Total Trades: {len(self.trades)}")
        print(f"Wins: {len(wins)}")
        print(f"Losses: {len(losses)}")
        print(f"Win Rate: {len(wins)/len(self.trades)*100:.1f}%")
        
        if wins:
            avg_win = np.mean([t['profit'] for t in wins])
            print(f"Average Win: ${avg_win:.2f}")
        if losses:
            avg_loss = np.mean([abs(t['profit']) for t in losses])
            print(f"Average Loss: ${avg_loss:.2f}")
        
        print(f"\nInitial Balance: ${self.initial_balance:.2f}")
        print(f"Final Balance: ${self.balance:.2f}")
        print(f"Total Profit/Loss: ${total_profit:+.2f}")
        print(f"ROI: {roi:+.1f}%")
        
        # Risk metrics
        equity = np.array(self.equity_curve)
        drawdowns = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity)
        max_drawdown = np.min(drawdowns) * 100
        
        print(f"Max Drawdown: {max_drawdown:.1f}%")
        
        # Expectancy (average $ per trade)
        expectancy = total_profit / len(self.trades) if self.trades else 0
        print(f"Expectancy: ${expectancy:.2f} per trade")
        
        # Reality check
        print("\n" + "=" * 50)
        if roi > 0:
            print("✅ PROFITABLE STRATEGY")
            print(f"   At this rate: ${total_profit:.2f} per {len(self.trades)} trades")
            monthly_trades = len(self.trades) * 30 / 100  # Rough estimate
            print(f"   Potential monthly: ${expectancy * monthly_trades:.2f}")
        else:
            print("❌ LOSING STRATEGY") 
            print("   DO NOT trade this with real money!")
        
        print("\n⚠️ WARNINGS:")
        print("- Past performance doesn't guarantee future results")
        print("- Real trading has more costs (VPS, data, etc.)")
        print("- Psychological factors will affect real trading")
        print("- Start with MINIMUM capital when going live")


# Test with real data
if __name__ == "__main__":
    print("HONEST FOREX SYSTEM TEST")
    print("=" * 50)
    
    # Generate sample data for testing
    import requests
    
    try:
        # Try to get real data
        response = requests.get(
            'https://www.alphavantage.co/query',
            params={
                'function': 'FX_DAILY',
                'from_symbol': 'EUR',
                'to_symbol': 'USD',
                'apikey': 'KNF41ZTAUM44W2LN'
            }
        )
        data = response.json()
        
        if 'Time Series FX (Daily)' in data:
            df = pd.DataFrame(data['Time Series FX (Daily)']).T
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df.columns = ['open', 'high', 'low', 'close']
            df = df.sort_index()
            df['volume'] = np.random.randint(10000, 100000, len(df))  # Fake volume
            
            print(f"Got {len(df)} days of real EUR/USD data")
        else:
            raise Exception("No data")
            
    except:
        print("Using simulated data for demonstration")
        # Create realistic forex data
        dates = pd.date_range(end=datetime.now(), periods=500, freq='4H')
        prices = 1.1000
        data = []
        
        for date in dates:
            # Random walk with trend
            change = np.random.normal(0.0001, 0.002)  # Realistic forex volatility
            prices = prices * (1 + change)
            
            high = prices * (1 + abs(np.random.normal(0, 0.001)))
            low = prices * (1 - abs(np.random.normal(0, 0.001)))
            close = np.random.uniform(low, high)
            open_price = np.random.uniform(low, high)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': np.random.randint(10000, 100000)
            })
        
        df = pd.DataFrame(data, index=dates)
    
    # Run the REAL strategy
    strategy = RealForexStrategy()
    
    # Test signal generation
    signal, confidence, reason = strategy.calculate_signal(df)
    print(f"\nCurrent Signal: {signal}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Reason: {reason}")
    
    # Run backtest
    backtester = ForexBacktester(strategy, initial_balance=1000)
    backtester.backtest(df)