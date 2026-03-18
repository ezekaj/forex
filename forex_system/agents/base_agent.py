"""Base agent class for the Zeo Investment AI multi-agent system."""

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from forex_system.models.llm_client import LLMClient


@dataclass
class AgentResult:
    agent: str
    asset: str
    timestamp: str
    data: dict
    processing_time_ms: float = 0


class BaseAgent:
    """
    Base class for all agents in the investment system.

    Provides:
    - LLM access via vLLM API
    - SQLite database access for persistent state
    - Logging with agent name prefix
    - Performance tracking
    """

    def __init__(
        self,
        name: str,
        llm_endpoint: str = None,
        db_path: str = None,
    ):
        self.name = name
        self.log = logging.getLogger(f"agent.{name}")

        # LLM client (optional — some agents are code-only)
        self.llm = None
        if llm_endpoint:
            self.llm = LLMClient(base_url=llm_endpoint)

        # Database
        self.db_path = db_path or self._default_db_path()

        # Performance tracking
        self._cycle_count = 0
        self._total_time_ms = 0
        self._errors = 0

    def _default_db_path(self) -> str:
        from forex_system.training.config import TrainingConfig
        return TrainingConfig().ALPHA_MEMORY_DB_PATH

    def get_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def log_action(self, action: str, details: str = ""):
        """Log an agent action to the database."""
        self.log.info(f"{action}: {details}")
        try:
            with self.get_db() as conn:
                conn.execute(
                    """INSERT INTO agent_logs (agent_name, action, details, timestamp)
                       VALUES (?, ?, ?, ?)""",
                    (self.name, action, details, datetime.now().isoformat()),
                )
        except sqlite3.OperationalError:
            pass  # Table might not exist yet

    async def run_timed(self, func, *args, **kwargs):
        """Run a function and track execution time."""
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._cycle_count += 1
            self._total_time_ms += elapsed_ms
            return result
        except Exception as e:
            self._errors += 1
            self.log.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise

    def get_health(self) -> dict:
        """Return agent health metrics."""
        avg_time = self._total_time_ms / max(self._cycle_count, 1)
        return {
            "agent": self.name,
            "cycles": self._cycle_count,
            "errors": self._errors,
            "avg_time_ms": round(avg_time, 1),
            "status": "healthy" if self._errors < self._cycle_count * 0.1 else "degraded",
        }

    def get_learned_rules(self, asset: str = None, min_confidence: float = 0.5) -> list[dict]:
        """Load learned rules from alpha_memory."""
        query = "SELECT * FROM learned_rules WHERE confidence >= ?"
        params = [min_confidence]
        if asset:
            query += " AND condition LIKE ?"
            params.append(f"%{asset}%")
        query += " ORDER BY confidence DESC LIMIT 20"

        try:
            with self.get_db() as conn:
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            return []

    def save_learned_rule(self, rule: dict):
        """Save a learned rule to alpha_memory."""
        with self.get_db() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO learned_rules
                (condition, prediction, confidence, samples, correct, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                rule["condition"],
                rule["prediction"],
                rule["confidence"],
                rule.get("samples", 1),
                rule.get("correct", 1 if rule.get("was_correct") else 0),
                rule.get("created_at", datetime.now().isoformat()),
                datetime.now().isoformat(),
            ))
