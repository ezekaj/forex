"""
Game Engine
===========
The main trading simulation "game" that trains the bot.

Each "round" is a trading decision:
1. Bot sees past data only
2. Bot makes decision
3. We verify outcome
4. Bot learns from result
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_DIR,
    RESULTS_DIR,
    SIM_STARTING_CAPITAL,
    MAX_RISK_PER_TRADE,
    MIN_CONFIDENCE,
    SL_ATR_MULTIPLIER,
    TP_ATR_MULTIPLIER,
    PRIMARY_PAIRS
)
from simulation.time_machine import TimeMachine, SimulationState

logger = logging.getLogger(__name__)


class Trade:
    """Represents a single trade."""

    def __init__(
        self,
        trade_id: str,
        pair: str,
        direction: str,
        entry_price: float,
        entry_time: datetime,
        stop_loss: float,
        take_profit: float,
        size: float,
        confidence: float,
        reasoning: str
    ):
        self.trade_id = trade_id
        self.pair = pair
        self.direction = direction  # "BUY" or "SELL"
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.size = size
        self.confidence = confidence
        self.reasoning = reasoning

        # Filled when trade closes
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[datetime] = None
        self.pnl: Optional[float] = None
        self.pnl_pips: Optional[float] = None
        self.outcome: Optional[str] = None  # "WIN" or "LOSS"
        self.exit_reason: Optional[str] = None  # "TP", "SL", "MANUAL"
        self.lesson: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "trade_id": self.trade_id,
            "pair": self.pair,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "size": self.size,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "pnl": self.pnl,
            "pnl_pips": self.pnl_pips,
            "outcome": self.outcome,
            "exit_reason": self.exit_reason,
            "lesson": self.lesson
        }


class GameEngine:
    """
    Main simulation engine for training the trading bot.

    The "game":
    1. Present market state (past data only)
    2. Bot decides: BUY / SELL / HOLD
    3. If trade opened, verify against actual outcome
    4. Record result and extract lesson
    5. Advance time, repeat
    """

    def __init__(
        self,
        start_date: str = None,
        end_date: str = None,
        starting_capital: float = None,
        pairs: List[str] = None
    ):
        self.time_machine = TimeMachine(start_date, end_date)
        self.state = SimulationState(self.time_machine)

        self.capital = starting_capital or SIM_STARTING_CAPITAL
        self.state.start_capital = self.capital
        self.state.current_capital = self.capital

        self.pairs = pairs or PRIMARY_PAIRS

        self.open_positions: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []
        self.trade_counter = 0

        # Database for results
        self.db_path = str(DATA_DIR / "simulation_results.db")
        self._init_db()

        logger.info(f"GameEngine initialized: {self.time_machine.start_date} → {self.time_machine.end_date}")
        logger.info(f"Starting capital: ${self.capital:,.2f}")

    def _init_db(self):
        """Initialize results database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                pair TEXT,
                direction TEXT,
                entry_price REAL,
                entry_time TEXT,
                exit_price REAL,
                exit_time TEXT,
                stop_loss REAL,
                take_profit REAL,
                size REAL,
                confidence REAL,
                reasoning TEXT,
                pnl REAL,
                pnl_pips REAL,
                outcome TEXT,
                exit_reason TEXT,
                lesson TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                pair TEXT,
                action TEXT,
                confidence REAL,
                reasoning TEXT,
                indicators TEXT,
                news_context TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self.trade_counter += 1
        return f"SIM_{self.time_machine.now_date}_{self.trade_counter:04d}"

    def get_market_state(self, pair: str, price_data: pd.DataFrame) -> Dict:
        """
        Get current market state for the bot.

        IMPORTANT: Only includes data BEFORE current simulated time.

        Args:
            pair: Currency pair
            price_data: Full price DataFrame

        Returns:
            Dict with market state the bot is allowed to see
        """
        # Filter to only past data
        filtered_data = self.time_machine.filter_data_by_time(price_data)

        if filtered_data is None or len(filtered_data) < 50:
            return {"error": "Insufficient data", "pair": pair}

        # Get latest values
        latest = filtered_data.iloc[-1]
        prev = filtered_data.iloc[-2] if len(filtered_data) > 1 else latest

        return {
            "pair": pair,
            "current_time": self.time_machine.now_str,
            "current_price": float(latest['close']),
            "open": float(latest['open']),
            "high": float(latest['high']),
            "low": float(latest['low']),
            "previous_close": float(prev['close']),
            "change_pct": ((latest['close'] - prev['close']) / prev['close']) * 100,
            "bars_available": len(filtered_data),
        }

    def make_decision(
        self,
        pair: str,
        action: str,
        confidence: float,
        reasoning: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        indicators: Dict = None,
        news_context: str = None
    ) -> Optional[Trade]:
        """
        Process a trading decision from the bot.

        Args:
            pair: Currency pair
            action: "BUY", "SELL", or "HOLD"
            confidence: Confidence level (0-1)
            reasoning: Why the bot made this decision
            entry_price: Entry price
            stop_loss: Stop loss level
            take_profit: Take profit level
            indicators: Technical indicators used
            news_context: News that influenced decision

        Returns:
            Trade object if trade opened, None otherwise
        """
        # Record decision
        decision = {
            "pair": pair,
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
            "indicators": indicators,
            "news_context": news_context
        }
        self.state.record_decision(decision)
        self._save_decision(decision)

        # Check if should trade
        if action == "HOLD":
            logger.info(f"[{self.time_machine.now_str}] {pair}: HOLD (confidence: {confidence:.1%})")
            return None

        if confidence < MIN_CONFIDENCE:
            logger.info(f"[{self.time_machine.now_str}] {pair}: Confidence too low ({confidence:.1%} < {MIN_CONFIDENCE:.1%})")
            return None

        if pair in self.open_positions:
            logger.info(f"[{self.time_machine.now_str}] {pair}: Already have open position")
            return None

        # Calculate position size
        risk_amount = self.capital * MAX_RISK_PER_TRADE
        sl_distance = abs(entry_price - stop_loss)

        if sl_distance <= 0:
            logger.warning(f"Invalid stop loss distance for {pair}")
            return None

        # Simplified: assume $10 per pip per lot for major pairs
        pip_value = 0.0001 if "JPY" not in pair else 0.01
        sl_pips = sl_distance / pip_value
        size = risk_amount / (sl_pips * 10) if sl_pips > 0 else 0.01
        size = min(size, 10.0)  # Cap at 10 lots

        # Create trade
        trade = Trade(
            trade_id=self._generate_trade_id(),
            pair=pair,
            direction=action,
            entry_price=entry_price,
            entry_time=self.time_machine.now,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size=size,
            confidence=confidence,
            reasoning=reasoning
        )

        self.open_positions[pair] = trade

        logger.info(f"[{self.time_machine.now_str}] OPENED {action} {pair} @ {entry_price:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f} | Conf: {confidence:.1%}")

        return trade

    def check_positions(self, price_data: Dict[str, pd.DataFrame]) -> List[Trade]:
        """
        Check all open positions against current prices.

        Uses FUTURE data (from our perspective) to determine if SL/TP hit.

        Args:
            price_data: Dict of pair -> full price DataFrame

        Returns:
            List of trades that closed this round
        """
        closed = []

        for pair, trade in list(self.open_positions.items()):
            if pair not in price_data:
                continue

            df = price_data[pair]

            # Get price data AFTER entry (the "future" for the trade)
            future_data = self.time_machine.get_future_data(df, lookahead_hours=24)

            if future_data is None or len(future_data) == 0:
                continue

            # Check each bar for SL/TP
            for idx, row in future_data.iterrows():
                if trade.direction == "BUY":
                    # Check stop loss (low hit SL)
                    if row['low'] <= trade.stop_loss:
                        self._close_trade(trade, trade.stop_loss, idx, "SL")
                        closed.append(trade)
                        break
                    # Check take profit (high hit TP)
                    elif row['high'] >= trade.take_profit:
                        self._close_trade(trade, trade.take_profit, idx, "TP")
                        closed.append(trade)
                        break

                else:  # SELL
                    # Check stop loss (high hit SL)
                    if row['high'] >= trade.stop_loss:
                        self._close_trade(trade, trade.stop_loss, idx, "SL")
                        closed.append(trade)
                        break
                    # Check take profit (low hit TP)
                    elif row['low'] <= trade.take_profit:
                        self._close_trade(trade, trade.take_profit, idx, "TP")
                        closed.append(trade)
                        break

        return closed

    def _close_trade(self, trade: Trade, exit_price: float, exit_time: datetime, reason: str):
        """Close a trade and calculate P&L."""
        trade.exit_price = exit_price
        trade.exit_time = exit_time if isinstance(exit_time, datetime) else datetime.fromisoformat(str(exit_time))
        trade.exit_reason = reason

        # Calculate P&L
        pip_value = 0.0001 if "JPY" not in trade.pair else 0.01

        if trade.direction == "BUY":
            trade.pnl_pips = (exit_price - trade.entry_price) / pip_value
        else:
            trade.pnl_pips = (trade.entry_price - exit_price) / pip_value

        # P&L in dollars (assuming $10 per pip per lot)
        trade.pnl = trade.pnl_pips * 10 * trade.size

        # Subtract trading costs (~$7 per lot round trip)
        trade.pnl -= 7 * trade.size

        trade.outcome = "WIN" if trade.pnl > 0 else "LOSS"

        # Update capital
        self.capital += trade.pnl
        self.state.current_capital = self.capital
        self.state.record_capital(self.capital)

        # Move to closed trades
        if trade.pair in self.open_positions:
            del self.open_positions[trade.pair]
        self.closed_trades.append(trade)
        self.state.record_trade(trade.to_dict())

        # Save to database
        self._save_trade(trade)

        logger.info(
            f"[{trade.exit_time}] CLOSED {trade.direction} {trade.pair} | "
            f"{trade.outcome} | P&L: ${trade.pnl:+.2f} ({trade.pnl_pips:+.1f} pips) | "
            f"Capital: ${self.capital:,.2f}"
        )

    def extract_lesson(self, trade: Trade, llm_client) -> str:
        """
        Ask LLM to extract a lesson from a completed trade.

        Args:
            trade: Completed trade
            llm_client: Ollama client for LLM

        Returns:
            Lesson learned
        """
        prompt = f"""
Analyze this completed trade and extract a brief lesson (2-3 sentences max):

Trade Details:
- Pair: {trade.pair}
- Direction: {trade.direction}
- Entry: {trade.entry_price:.5f} at {trade.entry_time}
- Exit: {trade.exit_price:.5f} at {trade.exit_time}
- Outcome: {trade.outcome}
- P&L: {trade.pnl_pips:+.1f} pips (${trade.pnl:+.2f})
- Exit Reason: {trade.exit_reason}
- Original Reasoning: {trade.reasoning}

What is the key lesson from this trade? Focus on what worked or what went wrong.
"""
        try:
            lesson = llm_client.generate(prompt, temperature=0.3, max_tokens=200)
            trade.lesson = lesson.strip()
            return trade.lesson
        except Exception as e:
            logger.error(f"Failed to extract lesson: {e}")
            return f"Trade {trade.outcome}: {trade.exit_reason}"

    def _save_trade(self, trade: Trade):
        """Save trade to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO trades
            (trade_id, pair, direction, entry_price, entry_time, exit_price, exit_time,
             stop_loss, take_profit, size, confidence, reasoning, pnl, pnl_pips,
             outcome, exit_reason, lesson)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.trade_id, trade.pair, trade.direction,
            trade.entry_price, trade.entry_time.isoformat() if trade.entry_time else None,
            trade.exit_price, trade.exit_time.isoformat() if trade.exit_time else None,
            trade.stop_loss, trade.take_profit, trade.size, trade.confidence,
            trade.reasoning, trade.pnl, trade.pnl_pips, trade.outcome,
            trade.exit_reason, trade.lesson
        ))

        conn.commit()
        conn.close()

    def _serialize_indicators(self, indicators: Dict) -> str:
        """Serialize indicators to JSON, filtering out non-serializable values."""
        if not indicators:
            return None
        # Filter out pandas Series (keys ending with _series) and convert numpy types
        serializable = {}
        for k, v in indicators.items():
            if k.endswith('_series'):
                continue
            if hasattr(v, 'item'):  # numpy scalar
                serializable[k] = float(v.item()) if v is not None else None
            elif isinstance(v, (int, float, str, bool, type(None))):
                serializable[k] = v
            elif isinstance(v, list):
                serializable[k] = v
        return json.dumps(serializable)

    def _save_decision(self, decision: Dict):
        """Save decision to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO decisions
            (timestamp, pair, action, confidence, reasoning, indicators, news_context)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            self.time_machine.now_str,
            decision.get("pair"),
            decision.get("action"),
            decision.get("confidence"),
            decision.get("reasoning"),
            self._serialize_indicators(decision.get("indicators")),
            decision.get("news_context")
        ))

        conn.commit()
        conn.close()

    def get_performance_summary(self) -> Dict:
        """Get current performance summary."""
        summary = self.state.get_summary()

        # Add detailed stats
        if self.closed_trades:
            pnls = [t.pnl for t in self.closed_trades]
            summary["avg_pnl"] = np.mean(pnls)
            summary["max_win"] = max(pnls)
            summary["max_loss"] = min(pnls)
            summary["profit_factor"] = (
                sum(p for p in pnls if p > 0) / abs(sum(p for p in pnls if p < 0))
                if any(p < 0 for p in pnls) else float('inf')
            )

        return summary

    def print_status(self):
        """Print current simulation status."""
        summary = self.get_performance_summary()

        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    SIMULATION STATUS                          ║
╠══════════════════════════════════════════════════════════════╣
║  Time: {self.time_machine.now_str[:19]:<41} ║
║  Progress: {summary['progress']:<47} ║
╠══════════════════════════════════════════════════════════════╣
║  Capital: ${summary['current_capital']:>10,.2f}  (Return: {summary['return_pct']:>+6.2f}%)       ║
║  Trades: {summary['total_trades']:>3}  (W: {summary['wins']:>3} / L: {summary['losses']:>3})  Win Rate: {summary['win_rate']*100:>5.1f}%    ║
║  Open Positions: {len(self.open_positions):>3}                                       ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test game engine
    engine = GameEngine(
        start_date="2025-01-01",
        end_date="2025-01-10",
        starting_capital=30000
    )

    print("Game Engine initialized!")
    engine.print_status()

    # Simulate a few rounds
    for i in range(3):
        print(f"\n--- Round {i+1} ---")
        print(f"Current time: {engine.time_machine.now_str}")

        # Simulate a decision
        if i == 1:
            trade = engine.make_decision(
                pair="EURUSD",
                action="BUY",
                confidence=0.65,
                reasoning="Test trade - RSI oversold with bullish divergence",
                entry_price=1.0850,
                stop_loss=1.0800,
                take_profit=1.0950
            )

        engine.time_machine.advance()

    engine.print_status()
