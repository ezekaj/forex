"""SQLite-backed experiment tracking for training runs."""

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import pandas as pd

from forex_system.training.config import TrainingConfig


@dataclass
class ExperimentResult:
    experiment_id: str
    phase: str  # "news", "price", "merge"
    timestamp: str
    model_type: str
    symbols: list[str]
    timeframe: str
    hyperparams: dict
    train_metrics: dict
    val_metrics: dict
    backtest_metrics: dict
    data_hash: str
    model_path: Optional[str] = None
    notes: str = ""

    @staticmethod
    def create(
        phase: str,
        model_type: str,
        symbols: list[str],
        timeframe: str,
        hyperparams: dict,
        train_metrics: dict,
        val_metrics: dict,
        backtest_metrics: dict,
        data_hash: str = "",
        model_path: Optional[str] = None,
        notes: str = "",
    ) -> "ExperimentResult":
        return ExperimentResult(
            experiment_id=str(uuid.uuid4())[:8],
            phase=phase,
            timestamp=datetime.now().isoformat(),
            model_type=model_type,
            symbols=symbols,
            timeframe=timeframe,
            hyperparams=hyperparams,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            backtest_metrics=backtest_metrics,
            data_hash=data_hash,
            model_path=model_path,
            notes=notes,
        )


class ExperimentTracker:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    phase TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    symbols TEXT NOT NULL,
                    timeframe TEXT,
                    hyperparams TEXT NOT NULL,
                    train_metrics TEXT NOT NULL,
                    val_metrics TEXT NOT NULL,
                    backtest_metrics TEXT NOT NULL,
                    data_hash TEXT,
                    model_path TEXT,
                    notes TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_phase
                ON experiments(phase)
            """)

    def log_experiment(self, result: ExperimentResult) -> str:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    result.experiment_id,
                    result.phase,
                    result.timestamp,
                    result.model_type,
                    json.dumps(result.symbols),
                    result.timeframe,
                    json.dumps(result.hyperparams),
                    json.dumps(result.train_metrics),
                    json.dumps(result.val_metrics),
                    json.dumps(result.backtest_metrics),
                    result.data_hash,
                    result.model_path,
                    result.notes,
                ),
            )
        return result.experiment_id

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_result(row)

    def get_best_experiment(
        self,
        phase: str,
        metric: str = "win_rate",
        min_trades: int = 50,
    ) -> Optional[ExperimentResult]:
        experiments = self.get_history(phase=phase)
        if not experiments:
            return None

        valid = []
        for exp in experiments:
            bt = exp.backtest_metrics
            if bt.get("total_trades", 0) >= min_trades:
                score = bt.get(metric, exp.val_metrics.get(metric, 0))
                valid.append((score, exp))

        if not valid:
            return None
        valid.sort(key=lambda x: x[0], reverse=True)
        return valid[0][1]

    def get_history(
        self,
        phase: Optional[str] = None,
        limit: int = 50,
    ) -> list[ExperimentResult]:
        query = "SELECT * FROM experiments"
        params: list = []
        if phase:
            query += " WHERE phase = ?"
            params.append(phase)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_result(row) for row in rows]

    def compare_experiments(self, experiment_ids: list[str]) -> pd.DataFrame:
        experiments = [self.get_experiment(eid) for eid in experiment_ids]
        experiments = [e for e in experiments if e is not None]

        rows = []
        for exp in experiments:
            row = {
                "id": exp.experiment_id,
                "phase": exp.phase,
                "model": exp.model_type,
                "symbols": ",".join(exp.symbols),
                **{f"val_{k}": v for k, v in exp.val_metrics.items()},
                **{f"bt_{k}": v for k, v in exp.backtest_metrics.items()},
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def passes_gate(self, result: ExperimentResult, config: TrainingConfig = None) -> bool:
        config = config or TrainingConfig()
        bt = result.backtest_metrics

        win_rate = bt.get("win_rate", 0)
        sharpe = bt.get("sharpe_ratio", 0)
        profit_factor = bt.get("profit_factor", 0)
        max_dd = bt.get("max_drawdown", 1.0)
        total_trades = bt.get("total_trades", 0)

        return (
            win_rate >= config.MIN_WIN_RATE
            and sharpe >= config.MIN_SHARPE
            and profit_factor >= config.MIN_PROFIT_FACTOR
            and max_dd <= config.MAX_DRAWDOWN_PCT
            and total_trades >= config.MIN_TRADES_PER_FOLD
        )

    @staticmethod
    def _row_to_result(row) -> ExperimentResult:
        return ExperimentResult(
            experiment_id=row[0],
            phase=row[1],
            timestamp=row[2],
            model_type=row[3],
            symbols=json.loads(row[4]),
            timeframe=row[5],
            hyperparams=json.loads(row[6]),
            train_metrics=json.loads(row[7]),
            val_metrics=json.loads(row[8]),
            backtest_metrics=json.loads(row[9]),
            data_hash=row[10],
            model_path=row[11],
            notes=row[12],
        )

    @staticmethod
    def compute_data_hash(df: pd.DataFrame) -> str:
        """Compute a hash of the training data for reproducibility."""
        s = f"{len(df)}_{df.columns.tolist()}_{df.iloc[0].values.tolist() if len(df) > 0 else ''}"
        return hashlib.md5(s.encode()).hexdigest()[:12]
