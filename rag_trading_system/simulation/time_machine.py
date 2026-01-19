"""
Time Machine
============
Controls simulated time for backtesting.
Ensures the bot NEVER sees future data.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, List, Generator
import pandas as pd
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SIM_START_DATE, SIM_END_DATE, DECISION_INTERVAL_HOURS

logger = logging.getLogger(__name__)


class TimeMachine:
    """
    Controls simulated time for backtesting.

    The bot can only access data BEFORE the current simulated time.
    We (the testers) can see everything to verify outcomes.
    """

    def __init__(
        self,
        start_date: str = None,
        end_date: str = None,
        interval_hours: int = None
    ):
        """
        Initialize time machine.

        Args:
            start_date: Simulation start (YYYY-MM-DD)
            end_date: Simulation end (YYYY-MM-DD)
            interval_hours: Hours between decision points
        """
        self.start_date = datetime.fromisoformat(start_date or SIM_START_DATE)
        self.end_date = datetime.fromisoformat(end_date or SIM_END_DATE)
        self.interval = timedelta(hours=interval_hours or DECISION_INTERVAL_HOURS)

        self.current_time = self.start_date
        self._paused = False

        logger.info(f"TimeMachine initialized: {self.start_date} → {self.end_date}")

    @property
    def now(self) -> datetime:
        """Current simulated time (what the bot sees as 'now')."""
        return self.current_time

    @property
    def now_str(self) -> str:
        """Current time as ISO string."""
        return self.current_time.isoformat()

    @property
    def now_date(self) -> str:
        """Current date as YYYY-MM-DD string."""
        return self.current_time.strftime("%Y-%m-%d")

    def advance(self, hours: int = None) -> datetime:
        """
        Advance time by specified hours or default interval.

        Returns:
            New current time
        """
        delta = timedelta(hours=hours) if hours else self.interval
        self.current_time = min(self.current_time + delta, self.end_date)
        return self.current_time

    def jump_to(self, target_date: str) -> datetime:
        """
        Jump to a specific date.

        Args:
            target_date: Target date (YYYY-MM-DD or ISO format)

        Returns:
            New current time
        """
        target = datetime.fromisoformat(target_date)

        if target < self.start_date:
            logger.warning(f"Cannot jump before start date. Staying at {self.start_date}")
            target = self.start_date
        elif target > self.end_date:
            logger.warning(f"Cannot jump past end date. Jumping to {self.end_date}")
            target = self.end_date

        self.current_time = target
        return self.current_time

    def reset(self):
        """Reset to start date."""
        self.current_time = self.start_date
        logger.info(f"TimeMachine reset to {self.start_date}")

    def is_finished(self) -> bool:
        """Check if simulation has reached end date."""
        return self.current_time >= self.end_date

    def progress(self) -> float:
        """Get simulation progress as percentage."""
        total = (self.end_date - self.start_date).total_seconds()
        current = (self.current_time - self.start_date).total_seconds()
        return (current / total) * 100 if total > 0 else 100

    def remaining_steps(self) -> int:
        """Estimate remaining decision steps."""
        remaining = self.end_date - self.current_time
        return int(remaining / self.interval) if self.interval.total_seconds() > 0 else 0

    def iterate(self) -> Generator[datetime, None, None]:
        """
        Iterate through all time steps.

        Yields:
            Current time at each step
        """
        self.reset()
        while not self.is_finished():
            yield self.current_time
            self.advance()

    def filter_data_by_time(
        self,
        df: pd.DataFrame,
        date_column: str = None
    ) -> pd.DataFrame:
        """
        Filter DataFrame to only include data BEFORE current simulated time.

        This is CRITICAL for preventing look-ahead bias.

        Args:
            df: DataFrame with datetime index or date column
            date_column: Name of date column (if not using index)

        Returns:
            Filtered DataFrame with only past data
        """
        if df is None or len(df) == 0:
            return df

        df = df.copy()

        if date_column:
            # Filter by column
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column])
                return df[df[date_column] < self.current_time]
        else:
            # Filter by index
            if isinstance(df.index, pd.DatetimeIndex):
                return df[df.index < self.current_time]
            else:
                try:
                    df.index = pd.to_datetime(df.index)
                    return df[df.index < self.current_time]
                except:
                    logger.warning("Could not filter by time - index not datetime")
                    return df

        return df

    def can_see(self, target_date: str) -> bool:
        """
        Check if the bot can see data from a specific date.

        The bot can only see data BEFORE current simulated time.

        Args:
            target_date: Date to check (ISO format)

        Returns:
            True if bot can see this date, False otherwise
        """
        target = datetime.fromisoformat(target_date)
        return target < self.current_time

    def get_future_data(
        self,
        df: pd.DataFrame,
        lookahead_hours: int = 24,
        date_column: str = None
    ) -> pd.DataFrame:
        """
        Get FUTURE data for outcome verification.

        This is for US (testers) to verify outcomes - NOT for the bot!

        Args:
            df: Full DataFrame with all data
            lookahead_hours: How far ahead to look
            date_column: Name of date column (if not using index)

        Returns:
            DataFrame with future data
        """
        if df is None or len(df) == 0:
            return df

        df = df.copy()
        future_cutoff = self.current_time + timedelta(hours=lookahead_hours)

        if date_column:
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column])
                return df[(df[date_column] >= self.current_time) & (df[date_column] <= future_cutoff)]
        else:
            if isinstance(df.index, pd.DatetimeIndex):
                return df[(df.index >= self.current_time) & (df.index <= future_cutoff)]
            else:
                try:
                    df.index = pd.to_datetime(df.index)
                    return df[(df.index >= self.current_time) & (df.index <= future_cutoff)]
                except:
                    return df

        return df

    def get_context(self) -> Dict:
        """
        Get current time context for the bot.

        Returns:
            Dict with time information the bot is allowed to know
        """
        return {
            "current_time": self.now_str,
            "current_date": self.now_date,
            "day_of_week": self.current_time.strftime("%A"),
            "hour": self.current_time.hour,
            "is_market_hours": self._is_market_hours(),
            "is_weekend": self.current_time.weekday() >= 5,
        }

    def _is_market_hours(self) -> bool:
        """Check if within typical forex market hours."""
        # Forex markets are open 24/5 (Sunday 5pm EST to Friday 5pm EST)
        day = self.current_time.weekday()
        hour = self.current_time.hour

        if day == 5:  # Saturday
            return False
        if day == 6:  # Sunday
            return hour >= 17  # Opens 5pm
        if day == 4:  # Friday
            return hour < 17  # Closes 5pm
        return True


class SimulationState:
    """Tracks the full state of a simulation run."""

    def __init__(self, time_machine: TimeMachine):
        self.time_machine = time_machine
        self.trades: List[Dict] = []
        self.decisions: List[Dict] = []
        self.capital_history: List[Dict] = []
        self.start_capital: float = 0
        self.current_capital: float = 0

    def record_decision(self, decision: Dict):
        """Record a trading decision."""
        decision["timestamp"] = self.time_machine.now_str
        self.decisions.append(decision)

    def record_trade(self, trade: Dict):
        """Record a completed trade."""
        trade["recorded_at"] = self.time_machine.now_str
        self.trades.append(trade)

    def record_capital(self, capital: float):
        """Record capital at current time."""
        self.capital_history.append({
            "timestamp": self.time_machine.now_str,
            "capital": capital
        })
        self.current_capital = capital

    def get_summary(self) -> Dict:
        """Get simulation summary."""
        wins = len([t for t in self.trades if t.get("outcome") == "WIN"])
        losses = len([t for t in self.trades if t.get("outcome") == "LOSS"])
        total_pnl = sum(t.get("pnl", 0) for t in self.trades)

        return {
            "start_date": self.time_machine.start_date.isoformat(),
            "end_date": self.time_machine.end_date.isoformat(),
            "current_date": self.time_machine.now_str,
            "progress": f"{self.time_machine.progress():.1f}%",
            "total_decisions": len(self.decisions),
            "total_trades": len(self.trades),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(self.trades) if self.trades else 0,
            "total_pnl": total_pnl,
            "start_capital": self.start_capital,
            "current_capital": self.current_capital,
            "return_pct": (self.current_capital / self.start_capital - 1) * 100 if self.start_capital > 0 else 0
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test time machine
    tm = TimeMachine(
        start_date="2025-01-01",
        end_date="2025-01-10",
        interval_hours=4
    )

    print(f"Start: {tm.start_date}")
    print(f"End: {tm.end_date}")
    print(f"Current: {tm.now}")
    print(f"Progress: {tm.progress():.1f}%")
    print(f"Remaining steps: {tm.remaining_steps()}")

    print("\n--- Advancing through time ---")
    for i, time_point in enumerate(tm.iterate()):
        if i > 5:
            print("...")
            break
        print(f"Step {i}: {time_point} | Context: {tm.get_context()}")

    print(f"\nFinal: {tm.now}")
    print(f"Progress: {tm.progress():.1f}%")
