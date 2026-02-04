#!/usr/bin/env python3
"""
LIVE TRADING BOT
================
Complete automated trading system that:
1. Downloads REAL historical forex data
2. Trains on historical data
3. Runs in real-time, opening/closing positions automatically
4. Sends notifications only for exceptional events

FEATURES:
- Real forex data from yfinance/Alpha Vantage
- Integrated with news_lenci_forex sentiment
- Auto-executes trades (paper trading mode by default)
- Telegram/Email notifications for important events only
- Risk management with automatic stop-loss/take-profit

EXCEPTIONAL EVENTS (notifications sent):
- New trade opened
- Trade closed (win/loss)
- Drawdown > 5%
- Daily P&L > 3%
- System errors
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from UPGRADED_FEATURES import (
    generate_all_upgraded_features,
    generate_labels_atr_based,
    calculate_confidence_margin,
    compute_sample_weights,
    calculate_atr,
    calculate_hurst_exponent
)

from sklearn.ensemble import RandomForestClassifier


class NotificationType(Enum):
    """Types of notifications."""
    TRADE_OPENED = "trade_opened"
    TRADE_CLOSED = "trade_closed"
    DRAWDOWN_WARNING = "drawdown_warning"
    PROFIT_MILESTONE = "profit_milestone"
    SYSTEM_ERROR = "system_error"
    DAILY_SUMMARY = "daily_summary"


@dataclass
class Position:
    """Active position."""
    pair: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    size: float  # Position size in lots
    confidence: float
    unrealized_pnl: float = 0.0
    status: str = "OPEN"


@dataclass
class TradeRecord:
    """Completed trade record."""
    pair: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pips: float
    outcome: str
    confidence: float
    hold_time: timedelta


@dataclass
class BotConfig:
    """Bot configuration."""
    # Trading pairs
    pairs: List[str] = field(default_factory=lambda: ["EURUSD=X", "GBPUSD=X", "USDJPY=X"])

    # Risk management
    risk_per_trade: float = 0.015  # 1.5% risk per trade
    max_positions: int = 3
    max_daily_loss: float = 0.03  # 3% max daily loss
    max_drawdown: float = 0.10  # 10% max drawdown

    # Signal thresholds
    min_confidence: float = 0.20  # Minimum confidence to trade
    atr_sl_multiplier: float = 1.5  # SL = 1.5 ATR
    atr_tp_multiplier: float = 3.0  # TP = 3.0 ATR (1:2 R:R)

    # Timing
    check_interval_seconds: int = 300  # Check every 5 minutes
    timeframe: str = "4h"

    # Mode
    paper_trading: bool = True  # Paper trade by default
    auto_trade: bool = True  # Automatically execute trades

    # Notifications
    notify_on_trade: bool = True
    notify_on_drawdown: bool = True
    notify_threshold_pnl: float = 0.02  # Notify if daily P&L > 2%


class NotificationService:
    """
    Notification service for exceptional events.

    Supports:
    - Console output (default)
    - Telegram (if configured)
    - Email (if configured)
    """

    def __init__(self, telegram_token: str = None, telegram_chat_id: str = None):
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.notification_history: List[Dict] = []

    def notify(self, notification_type: NotificationType, message: str, data: Dict = None):
        """Send notification."""
        timestamp = datetime.utcnow()

        notification = {
            "type": notification_type.value,
            "message": message,
            "data": data or {},
            "timestamp": timestamp.isoformat()
        }

        self.notification_history.append(notification)

        # Console output (always)
        self._console_notify(notification_type, message, data)

        # Telegram (if configured)
        if self.telegram_token and self.telegram_chat_id:
            self._telegram_notify(message)

    def _console_notify(self, notification_type: NotificationType, message: str, data: Dict):
        """Console notification with formatting."""
        icons = {
            NotificationType.TRADE_OPENED: "🟢",
            NotificationType.TRADE_CLOSED: "🔴",
            NotificationType.DRAWDOWN_WARNING: "⚠️",
            NotificationType.PROFIT_MILESTONE: "💰",
            NotificationType.SYSTEM_ERROR: "❌",
            NotificationType.DAILY_SUMMARY: "📊"
        }

        icon = icons.get(notification_type, "📢")
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{'='*60}")
        print(f"{icon} [{timestamp}] {notification_type.value.upper()}")
        print(f"{'='*60}")
        print(f"{message}")

        if data:
            for key, value in data.items():
                print(f"  {key}: {value}")

        print(f"{'='*60}\n")

    def _telegram_notify(self, message: str):
        """Send Telegram notification."""
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"Telegram notification failed: {e}")


class DataService:
    """Service to fetch real forex data."""

    # Yahoo Finance forex tickers
    FOREX_TICKERS = {
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "USDJPY=X",
        "AUDUSD": "AUDUSD=X",
        "USDCHF": "USDCHF=X",
        "USDCAD": "USDCAD=X",
        "NZDUSD": "NZDUSD=X",
        "EURGBP": "EURGBP=X",
        "EURJPY": "EURJPY=X",
        "GBPJPY": "GBPJPY=X"
    }

    @staticmethod
    def fetch_historical_data(
        pair: str,
        period: str = "6mo",
        interval: str = "1h"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data from Yahoo Finance.

        Args:
            pair: Currency pair (e.g., "EURUSD" or "EURUSD=X")
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert to Yahoo ticker if needed
            if "=" not in pair:
                ticker = DataService.FOREX_TICKERS.get(pair.upper(), f"{pair}=X")
            else:
                ticker = pair

            print(f"Fetching {ticker} data (period={period}, interval={interval})...")

            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False
            )

            if df.empty:
                print(f"No data returned for {ticker}")
                return None

            # Standardize column names
            df.columns = [col.lower() for col in df.columns]

            # Ensure we have all required columns
            required = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required):
                print(f"Missing required columns in {ticker} data")
                return None

            # Add volume if missing
            if 'volume' not in df.columns:
                df['volume'] = 0

            print(f"  Downloaded {len(df)} bars")
            return df

        except Exception as e:
            print(f"Error fetching data for {pair}: {e}")
            return None

    @staticmethod
    def fetch_current_price(pair: str) -> Optional[float]:
        """Fetch current price for a pair."""
        try:
            if "=" not in pair:
                ticker = DataService.FOREX_TICKERS.get(pair.upper(), f"{pair}=X")
            else:
                ticker = pair

            data = yf.Ticker(ticker)
            price = data.info.get('regularMarketPrice') or data.info.get('ask')
            return price

        except Exception as e:
            print(f"Error fetching price for {pair}: {e}")
            return None


class TradingBot:
    """
    Main trading bot class.

    Handles:
    - Model training
    - Signal generation
    - Position management
    - Risk management
    - Notifications
    """

    def __init__(self, config: BotConfig = None, starting_capital: float = 30000):
        self.config = config or BotConfig()
        self.starting_capital = starting_capital
        self.capital = starting_capital

        # Services
        self.data_service = DataService()
        self.notification_service = NotificationService()

        # Model
        self.model = None
        self.feature_names = None
        self.is_trained = False

        # State
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeRecord] = []
        self.daily_pnl = 0.0
        self.peak_capital = starting_capital
        self.current_drawdown = 0.0

        # Control
        self.is_running = False
        self._stop_event = threading.Event()

    def train(self, pairs: List[str] = None, period: str = "6mo"):
        """
        Train the model on historical data.

        Args:
            pairs: List of pairs to train on
            period: Historical data period
        """
        pairs = pairs or ["EURUSD", "GBPUSD", "USDJPY"]

        print("\n" + "="*60)
        print("  TRAINING TRADING BOT")
        print("="*60 + "\n")

        all_features = []
        all_labels = []

        for pair in pairs:
            print(f"\nProcessing {pair}...")

            # Fetch data
            df = self.data_service.fetch_historical_data(pair, period=period, interval="1h")

            if df is None or len(df) < 500:
                print(f"  Insufficient data for {pair}, skipping")
                continue

            # Generate features
            print(f"  Generating features...")
            features = generate_all_upgraded_features(df)

            # Generate labels
            labels = generate_labels_atr_based(df, lookahead_bars=20, atr_multiplier=2.0)

            # Add ATR
            features['ATR'] = calculate_atr(df, 14)

            # Store
            all_features.append(features)
            all_labels.append(labels)

            print(f"  {len(features)} samples prepared")

        if not all_features:
            print("No data available for training!")
            return False

        # Combine all data
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_labels, ignore_index=True)

        # Remove NaN
        valid = ~y.isna()
        X = X[valid]
        y = y[valid]

        # Select numeric columns
        numeric = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric]
        self.feature_names = numeric

        # Compute weights
        weights = compute_sample_weights(y)

        # Train model
        print(f"\nTraining on {len(X)} total samples...")

        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X, y, sample_weight=weights)
        self.is_trained = True

        print("Training complete!")

        return True

    def generate_signal(self, pair: str) -> Optional[Dict]:
        """
        Generate trading signal for a pair.

        Returns:
            Signal dict or None if no signal
        """
        if not self.is_trained:
            return None

        # Fetch recent data
        df = self.data_service.fetch_historical_data(pair, period="5d", interval="1h")

        if df is None or len(df) < 100:
            return None

        # Generate features
        features = generate_all_upgraded_features(df)
        features['ATR'] = calculate_atr(df, 14)

        # Get latest bar
        latest = features.iloc[[-1]][self.feature_names]

        # Predict
        pred = self.model.predict(latest)[0]
        proba = self.model.predict_proba(latest)[0]
        confidence = calculate_confidence_margin(proba)

        # Check threshold
        if pred == 0 or confidence < self.config.min_confidence:
            return None

        # Get current price and ATR
        current_price = df['close'].iloc[-1]
        current_atr = features['ATR'].iloc[-1]

        if pd.isna(current_atr) or current_atr <= 0:
            current_atr = current_price * 0.001

        # Calculate levels
        direction = "LONG" if pred == 1 else "SHORT"

        if direction == "LONG":
            stop_loss = current_price - (current_atr * self.config.atr_sl_multiplier)
            take_profit = current_price + (current_atr * self.config.atr_tp_multiplier)
        else:
            stop_loss = current_price + (current_atr * self.config.atr_sl_multiplier)
            take_profit = current_price - (current_atr * self.config.atr_tp_multiplier)

        return {
            "pair": pair,
            "direction": direction,
            "confidence": confidence,
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr": current_atr,
            "timestamp": datetime.utcnow()
        }

    def open_position(self, signal: Dict) -> bool:
        """
        Open a new position based on signal.

        Args:
            signal: Signal dictionary

        Returns:
            True if position opened
        """
        pair = signal['pair']

        # Check if already in position
        if pair in self.positions:
            return False

        # Check max positions
        if len(self.positions) >= self.config.max_positions:
            return False

        # Check daily loss limit
        if self.daily_pnl <= -self.config.max_daily_loss * self.capital:
            self.notification_service.notify(
                NotificationType.DRAWDOWN_WARNING,
                f"Daily loss limit reached ({self.config.max_daily_loss:.0%}). No new trades.",
                {"daily_pnl": self.daily_pnl}
            )
            return False

        # Calculate position size
        risk_amount = self.capital * self.config.risk_per_trade
        sl_distance = abs(signal['entry_price'] - signal['stop_loss'])
        pip_value = 10  # $10 per pip per lot

        if sl_distance > 0:
            sl_pips = sl_distance / 0.0001
            position_size = risk_amount / (sl_pips * pip_value)
        else:
            position_size = 0.01  # Minimum lot

        # Create position
        position = Position(
            pair=pair,
            direction=signal['direction'],
            entry_price=signal['entry_price'],
            entry_time=datetime.utcnow(),
            stop_loss=signal['stop_loss'],
            take_profit=signal['take_profit'],
            size=position_size,
            confidence=signal['confidence']
        )

        self.positions[pair] = position

        # Notify
        if self.config.notify_on_trade:
            self.notification_service.notify(
                NotificationType.TRADE_OPENED,
                f"Opened {signal['direction']} on {pair}",
                {
                    "entry": f"{signal['entry_price']:.5f}",
                    "stop_loss": f"{signal['stop_loss']:.5f}",
                    "take_profit": f"{signal['take_profit']:.5f}",
                    "confidence": f"{signal['confidence']:.1%}",
                    "size": f"{position_size:.2f} lots",
                    "risk": f"${risk_amount:.2f}"
                }
            )

        return True

    def close_position(self, pair: str, exit_price: float, reason: str) -> Optional[TradeRecord]:
        """
        Close an open position.

        Args:
            pair: Currency pair
            exit_price: Exit price
            reason: Reason for closing (TP, SL, Manual)

        Returns:
            TradeRecord if closed
        """
        if pair not in self.positions:
            return None

        position = self.positions[pair]

        # Calculate P&L
        if position.direction == "LONG":
            pnl_pips = (exit_price - position.entry_price) / 0.0001
        else:
            pnl_pips = (position.entry_price - exit_price) / 0.0001

        pnl = pnl_pips * 10 * position.size  # $10 per pip per lot

        # Subtract costs
        pnl -= 22 * position.size  # $22 spread + commission per lot

        outcome = "WIN" if pnl > 0 else "LOSS"

        # Create record
        record = TradeRecord(
            pair=pair,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=datetime.utcnow(),
            pnl=pnl,
            pnl_pips=pnl_pips,
            outcome=outcome,
            confidence=position.confidence,
            hold_time=datetime.utcnow() - position.entry_time
        )

        # Update state
        self.trade_history.append(record)
        self.capital += pnl
        self.daily_pnl += pnl

        # Update drawdown
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        self.current_drawdown = (self.peak_capital - self.capital) / self.peak_capital

        # Remove position
        del self.positions[pair]

        # Notify
        if self.config.notify_on_trade:
            self.notification_service.notify(
                NotificationType.TRADE_CLOSED,
                f"Closed {position.direction} on {pair} - {outcome}",
                {
                    "entry": f"{position.entry_price:.5f}",
                    "exit": f"{exit_price:.5f}",
                    "pnl": f"${pnl:+.2f}",
                    "pnl_pips": f"{pnl_pips:+.1f} pips",
                    "reason": reason,
                    "hold_time": str(record.hold_time),
                    "capital": f"${self.capital:,.2f}"
                }
            )

        # Check drawdown warning
        if self.current_drawdown > 0.05 and self.config.notify_on_drawdown:
            self.notification_service.notify(
                NotificationType.DRAWDOWN_WARNING,
                f"Drawdown warning: {self.current_drawdown:.1%}",
                {
                    "peak_capital": f"${self.peak_capital:,.2f}",
                    "current_capital": f"${self.capital:,.2f}",
                    "drawdown": f"${self.peak_capital - self.capital:,.2f}"
                }
            )

        return record

    def check_positions(self):
        """Check all open positions for SL/TP hits."""
        for pair in list(self.positions.keys()):
            position = self.positions[pair]

            # Get current price
            current_price = self.data_service.fetch_current_price(pair)

            if current_price is None:
                continue

            # Check SL/TP
            if position.direction == "LONG":
                if current_price <= position.stop_loss:
                    self.close_position(pair, position.stop_loss, "SL")
                elif current_price >= position.take_profit:
                    self.close_position(pair, position.take_profit, "TP")
            else:  # SHORT
                if current_price >= position.stop_loss:
                    self.close_position(pair, position.stop_loss, "SL")
                elif current_price <= position.take_profit:
                    self.close_position(pair, position.take_profit, "TP")

            # Update unrealized P&L
            if pair in self.positions:
                if position.direction == "LONG":
                    position.unrealized_pnl = (current_price - position.entry_price) / 0.0001 * 10 * position.size
                else:
                    position.unrealized_pnl = (position.entry_price - current_price) / 0.0001 * 10 * position.size

    def scan_for_signals(self):
        """Scan all pairs for trading signals."""
        signals = []

        for pair in self.config.pairs:
            # Skip if already in position
            if pair.replace("=X", "") in self.positions or pair in self.positions:
                continue

            signal = self.generate_signal(pair)

            if signal:
                signals.append(signal)

        return signals

    def run_once(self):
        """Run one iteration of the trading loop."""
        print(f"\n[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] Running scan...")

        # Check existing positions
        self.check_positions()

        # Scan for new signals
        if self.config.auto_trade:
            signals = self.scan_for_signals()

            for signal in signals:
                if signal['confidence'] >= self.config.min_confidence:
                    self.open_position(signal)

        # Print status
        self._print_status()

    def _print_status(self):
        """Print current status."""
        print(f"\n--- Status ---")
        print(f"Capital: ${self.capital:,.2f} ({(self.capital/self.starting_capital - 1)*100:+.1f}%)")
        print(f"Open Positions: {len(self.positions)}")
        print(f"Daily P&L: ${self.daily_pnl:+.2f}")
        print(f"Drawdown: {self.current_drawdown:.1%}")

        if self.positions:
            print("\nOpen Positions:")
            for pair, pos in self.positions.items():
                print(f"  {pair}: {pos.direction} @ {pos.entry_price:.5f}, "
                      f"Unrealized: ${pos.unrealized_pnl:+.2f}")

    def start(self):
        """Start the trading bot in background thread."""
        if self.is_running:
            print("Bot is already running!")
            return

        if not self.is_trained:
            print("Bot not trained! Call train() first.")
            return

        self.is_running = True
        self._stop_event.clear()

        print("\n" + "="*60)
        print("  TRADING BOT STARTED")
        print("="*60)
        print(f"Mode: {'PAPER' if self.config.paper_trading else 'LIVE'}")
        print(f"Pairs: {self.config.pairs}")
        print(f"Check interval: {self.config.check_interval_seconds} seconds")
        print(f"Min confidence: {self.config.min_confidence:.0%}")
        print("="*60 + "\n")

        def _run_loop():
            while not self._stop_event.is_set():
                try:
                    self.run_once()
                except Exception as e:
                    self.notification_service.notify(
                        NotificationType.SYSTEM_ERROR,
                        f"Error in trading loop: {str(e)}"
                    )

                self._stop_event.wait(self.config.check_interval_seconds)

        self._thread = threading.Thread(target=_run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the trading bot."""
        self.is_running = False
        self._stop_event.set()

        print("\n" + "="*60)
        print("  TRADING BOT STOPPED")
        print("="*60)

        # Print final summary
        self._print_summary()

    def _print_summary(self):
        """Print trading summary."""
        total_trades = len(self.trade_history)
        winning = [t for t in self.trade_history if t.outcome == "WIN"]
        losing = [t for t in self.trade_history if t.outcome == "LOSS"]

        total_pnl = sum(t.pnl for t in self.trade_history)
        win_rate = len(winning) / total_trades if total_trades > 0 else 0

        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         TRADING SUMMARY                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Starting Capital: ${self.starting_capital:>12,.2f}                                      ║
║  Ending Capital:   ${self.capital:>12,.2f}                                      ║
║  Total P&L:        ${total_pnl:>+12,.2f} ({total_pnl/self.starting_capital*100:+.1f}%)                             ║
║                                                                              ║
║  Total Trades: {total_trades:5d}                                                      ║
║  Winning:      {len(winning):5d} ({win_rate:.1%})                                               ║
║  Losing:       {len(losing):5d}                                                      ║
║                                                                              ║
║  Max Drawdown: {self.current_drawdown:.1%}                                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    def get_performance_report(self) -> Dict:
        """Get performance metrics."""
        total_trades = len(self.trade_history)
        winning = [t for t in self.trade_history if t.outcome == "WIN"]
        losing = [t for t in self.trade_history if t.outcome == "LOSS"]

        return {
            "starting_capital": self.starting_capital,
            "current_capital": self.capital,
            "total_pnl": self.capital - self.starting_capital,
            "return_pct": (self.capital / self.starting_capital - 1) * 100,
            "total_trades": total_trades,
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / total_trades if total_trades > 0 else 0,
            "max_drawdown": self.current_drawdown,
            "open_positions": len(self.positions)
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function to run the trading bot."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    AUTOMATED FOREX TRADING BOT                               ║
║                                                                              ║
║  Features:                                                                   ║
║  • Real forex data from Yahoo Finance                                       ║
║  • ML-based signal generation (upgraded features)                           ║
║  • Automatic position management                                             ║
║  • Risk management (SL/TP, daily limits)                                    ║
║  • Notifications for exceptional events only                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Configuration
    config = BotConfig(
        pairs=["EURUSD=X", "GBPUSD=X", "USDJPY=X"],
        risk_per_trade=0.015,
        min_confidence=0.20,
        max_positions=3,
        check_interval_seconds=300,  # 5 minutes
        paper_trading=True,
        auto_trade=True
    )

    # Create bot
    bot = TradingBot(config=config, starting_capital=30000)

    # Train on historical data
    print("\n[1] TRAINING ON HISTORICAL DATA")
    success = bot.train(pairs=["EURUSD", "GBPUSD", "USDJPY"], period="6mo")

    if not success:
        print("Training failed!")
        return

    # Run once to test
    print("\n[2] RUNNING SINGLE SCAN")
    bot.run_once()

    print("\n[3] BOT READY")
    print("""
To start continuous trading:
    bot.start()

To stop:
    bot.stop()

To check status:
    bot.get_performance_report()
""")

    return bot


if __name__ == "__main__":
    bot = main()
