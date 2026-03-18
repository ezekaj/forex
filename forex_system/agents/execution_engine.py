"""Execution Engine — paper trading with P&L tracking."""

import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from forex_system.agents.fund_manager import Decision
from forex_system.training.config import TrainingConfig
from forex_system.training.data.price_loader import UniversalPriceLoader

log = logging.getLogger(__name__)


@dataclass
class Trade:
    trade_id: str
    symbol: str
    direction: str        # "BUY" or "SELL"
    entry_price: float
    entry_date: str
    size_pct: float       # position size as % of portfolio
    stop_loss: float      # price level
    take_profit: float    # price level
    max_hold_date: str    # exit by this date
    status: str = "open"  # open, closed_stop, closed_target, closed_timeout, closed_manual
    exit_price: float = 0
    exit_date: str = ""
    pnl_pct: float = 0
    confidence: int = 0
    reasoning: str = ""


class ExecutionEngine:
    """
    Paper trading engine. Tracks positions, P&L, win rate.
    Uses SQLite for persistent storage.
    """

    def __init__(self, db_path: str = None):
        config = TrainingConfig()
        self.db_path = db_path or config.PAPER_TRADES_DB_PATH
        self.price_loader = UniversalPriceLoader(config)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_date TEXT NOT NULL,
                    size_pct REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    max_hold_date TEXT,
                    status TEXT DEFAULT 'open',
                    exit_price REAL DEFAULT 0,
                    exit_date TEXT DEFAULT '',
                    pnl_pct REAL DEFAULT 0,
                    confidence INTEGER DEFAULT 0,
                    reasoning TEXT DEFAULT '',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    total_value REAL,
                    open_positions INTEGER,
                    daily_pnl REAL,
                    cumulative_pnl REAL,
                    win_rate REAL,
                    total_trades INTEGER
                )
            """)

    def place_order(
        self,
        decision: Decision,
        entry_price: float,
        size_pct: float = 0.03,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.06,
        max_hold_days: int = 7,
    ) -> str:
        """Open a paper trade. Returns trade_id."""
        trade_id = str(uuid.uuid4())[:8]

        if decision.direction == "BUY":
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
        else:  # SELL
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)

        entry_dt = datetime.now()
        max_hold_date = (entry_dt + timedelta(days=max_hold_days)).strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trades (trade_id, symbol, direction, entry_price, entry_date,
                    size_pct, stop_loss, take_profit, max_hold_date, confidence, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id, decision.symbol, decision.direction, entry_price,
                entry_dt.strftime("%Y-%m-%d"), size_pct, stop_loss, take_profit,
                max_hold_date, decision.confidence, decision.reasoning[:500],
            ))

        log.info(f"TRADE OPENED: {trade_id} | {decision.symbol} {decision.direction} @ {entry_price:.2f} "
                 f"| SL={stop_loss:.2f} TP={take_profit:.2f}")
        return trade_id

    def check_positions(self, current_date: str = None) -> list[Trade]:
        """
        Check all open positions against current prices.
        Close positions that hit stop, target, or timeout.
        Returns list of closed trades.
        """
        current_date = current_date or datetime.now().strftime("%Y-%m-%d")
        closed = []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            open_trades = conn.execute(
                "SELECT * FROM trades WHERE status = 'open'"
            ).fetchall()

        for row in open_trades:
            trade = Trade(**{k: row[k] for k in row.keys()})

            # Get current price
            try:
                df = self.price_loader.load_ohlcv(trade.symbol, "1d")
                if df.empty:
                    continue

                # Find price on or before current_date
                target = datetime.strptime(current_date, "%Y-%m-%d")
                valid = df[df.index <= target]
                if valid.empty:
                    continue
                current_price = float(valid["close"].iloc[-1])
            except Exception as e:
                log.warning(f"Cannot get price for {trade.symbol}: {e}")
                continue

            # Check exit conditions
            exit_reason = None
            if trade.direction == "BUY":
                if current_price <= trade.stop_loss:
                    exit_reason = "closed_stop"
                elif current_price >= trade.take_profit:
                    exit_reason = "closed_target"
                pnl_pct = (current_price / trade.entry_price - 1) * 100
            else:  # SELL
                if current_price >= trade.stop_loss:
                    exit_reason = "closed_stop"
                elif current_price <= trade.take_profit:
                    exit_reason = "closed_target"
                pnl_pct = (trade.entry_price / current_price - 1) * 100

            # Check timeout
            if current_date >= trade.max_hold_date and exit_reason is None:
                exit_reason = "closed_timeout"

            if exit_reason:
                trade.status = exit_reason
                trade.exit_price = current_price
                trade.exit_date = current_date
                trade.pnl_pct = round(pnl_pct, 4)

                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE trades SET status=?, exit_price=?, exit_date=?, pnl_pct=?
                        WHERE trade_id=?
                    """, (exit_reason, current_price, current_date, trade.pnl_pct, trade.trade_id))

                log.info(f"TRADE CLOSED: {trade.trade_id} | {trade.symbol} {exit_reason} "
                         f"| P&L: {trade.pnl_pct:+.2f}%")
                closed.append(trade)

        return closed

    def get_portfolio_state(self) -> dict:
        """Get current portfolio state."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            open_trades = conn.execute("SELECT * FROM trades WHERE status = 'open'").fetchall()
            all_closed = conn.execute("SELECT * FROM trades WHERE status != 'open'").fetchall()

            total_trades = len(all_closed)
            wins = sum(1 for t in all_closed if t["pnl_pct"] > 0)
            total_pnl = sum(t["pnl_pct"] for t in all_closed)

            positions = {}
            total_exposure = 0
            for t in open_trades:
                from forex_system.training.config import get_asset
                asset = get_asset(t["symbol"])
                positions[t["symbol"]] = {
                    "direction": t["direction"],
                    "entry_price": t["entry_price"],
                    "size_pct": t["size_pct"],
                    "sector": asset.sector,
                    "asset_class": asset.asset_class,
                }
                total_exposure += t["size_pct"]

        return {
            "positions": positions,
            "total_exposure": total_exposure,
            "open_count": len(open_trades),
            "total_trades": total_trades,
            "wins": wins,
            "win_rate": wins / max(total_trades, 1),
            "cumulative_pnl": total_pnl,
            "daily_pnl": 0,  # TODO: compute from today's position changes
        }

    def get_trade_history(self) -> list[dict]:
        """Get all closed trades."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades WHERE status != 'open' ORDER BY exit_date DESC"
            ).fetchall()
            return [dict(r) for r in rows]
