"""Coordinator — the main orchestrator that runs the full investment cycle."""

import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Optional

from forex_system.agents.news_analyst import NewsAnalyst
from forex_system.agents.technical_analyst import TechnicalAnalyst
from forex_system.agents.researchers import BullResearcher, BearResearcher
from forex_system.agents.fund_manager import FundManager, Decision
from forex_system.agents.meta_learner import MetaLearner
from forex_system.training.config import TrainingConfig, ASSET_REGISTRY, get_symbols_by_class

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-20s] %(levelname)s %(message)s",
)
log = logging.getLogger("coordinator")

MIN_SIGNAL_CONFIDENCE = 15  # minimum confidence from news/tech to trigger analysis
MIN_DECISION_CONFIDENCE = 30  # minimum confidence from Fund Manager to act (low = small position)


class Coordinator:
    """
    Runs the full predict→check→reason→learn cycle.

    Modes:
    - historical: runs on past dates to train the system
    - live: runs on current data for paper/live trading
    """

    def __init__(
        self,
        llm_endpoint: str = "http://localhost:8000",
        symbols: list[str] = None,
        db_path: str = None,
    ):
        self.config = TrainingConfig()

        # Agents
        self.news_analyst = NewsAnalyst()
        self.tech_analyst = TechnicalAnalyst()
        self.bull = BullResearcher(llm_endpoint)
        self.bear = BearResearcher(llm_endpoint)
        self.fund_manager = FundManager(llm_endpoint)
        # Predictions database
        self.predictions_db = db_path or self.config.EXPERIMENT_DB_PATH.replace(
            "experiments.db", "coordinator_predictions.db"
        )

        # Meta-learner uses alpha_memory for learned rules
        self.meta_learner = MetaLearner(llm_endpoint, self.config.ALPHA_MEMORY_DB_PATH)

        # Symbol universe
        self.symbols = symbols or list(ASSET_REGISTRY.keys())
        self._init_predictions_db()

    def _init_predictions_db(self):
        with sqlite3.connect(self.predictions_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence INTEGER,
                    reasoning TEXT,
                    risk_factors TEXT,
                    news_signal TEXT,
                    tech_signal TEXT,
                    bull_argument TEXT,
                    bear_argument TEXT,
                    price_at_prediction REAL,
                    actual_change REAL,
                    was_correct INTEGER,
                    reflection TEXT,
                    learned_rule TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
            """)

    async def run_historical_day(self, date: str, symbols: list[str] = None):
        """
        Run one full cycle on a historical date.
        This is the core training loop: collect → predict → (check + reason happen later)
        """
        symbols = symbols or self.symbols
        cycle_start = time.perf_counter()

        log.info(f"{'='*60}")
        log.info(f"CYCLE: {date} | {len(symbols)} assets")
        log.info(f"{'='*60}")

        # STEP 1: Collect — News and Technical analysis for all assets
        log.info("Step 1: Collecting signals...")
        news_results = await self.news_analyst.analyze_all(symbols, date)
        tech_results = await self.tech_analyst.analyze_all(symbols, date)

        # STEP 2: Filter — only analyze assets with meaningful signals
        candidates = []
        for symbol in symbols:
            news = news_results.get(symbol)
            tech = tech_results.get(symbol)

            if not news or not tech:
                continue

            news_conf = news.data.get("confidence", 0)
            tech_conf = tech.data.get("confidence", 0)

            if news_conf >= MIN_SIGNAL_CONFIDENCE or tech_conf >= MIN_SIGNAL_CONFIDENCE:
                candidates.append((symbol, news.data, tech.data))

        log.info(f"Step 2: {len(candidates)} assets with signals (of {len(symbols)} total)")

        if not candidates:
            log.info("No signals. Skipping.")
            return

        # STEP 3: For each candidate — Bull/Bear debate → Fund Manager decision
        decisions = []
        for symbol, news_data, tech_data in candidates:
            try:
                # Get learned rules for this asset
                rules = self.meta_learner.get_learned_rules(symbol)

                # Bull/Bear debate
                bull_result = await self.bull.argue(symbol, news_data, tech_data, rules)
                bear_result = await self.bear.argue(symbol, news_data, tech_data, rules)

                # Fund Manager decision
                decision = await self.fund_manager.decide(
                    symbol, date, news_data, tech_data,
                    bull_result.data["argument"],
                    bear_result.data["argument"],
                    rules,
                )

                # If Fund Manager says HOLD but technicals have a clear direction, override
                if decision.direction == "HOLD" and tech_data.get("direction") in ("bullish", "bearish"):
                    override_dir = "BUY" if tech_data["direction"] == "bullish" else "SELL"
                    decision = Decision(
                        symbol=symbol,
                        direction=override_dir,
                        confidence=35,
                        reasoning=f"LLM inconclusive, following technical signal: {tech_data['direction']}",
                        risk_factors=decision.risk_factors,
                        time_horizon_days=3,
                        bull_strength=decision.bull_strength,
                        bear_strength=decision.bear_strength,
                    )
                    log.info(f"  {symbol}: HOLD→{override_dir} (tech override, conf=35)")

                if decision.confidence >= MIN_DECISION_CONFIDENCE and decision.direction != "HOLD":
                    decisions.append((decision, news_data, tech_data,
                                     bull_result.data["argument"], bear_result.data["argument"]))
                    log.info(f"  {symbol}: {decision.direction} (conf={decision.confidence})")
                else:
                    log.info(f"  {symbol}: HOLD (conf={decision.confidence})")

            except Exception as e:
                log.error(f"  {symbol}: ERROR — {e}")

        # STEP 4: Save predictions
        for decision, news_data, tech_data, bull_arg, bear_arg in decisions:
            self._save_prediction(decision, date, news_data, tech_data, bull_arg, bear_arg)

        elapsed = (time.perf_counter() - cycle_start)
        log.info(f"Cycle complete: {len(decisions)} predictions in {elapsed:.1f}s")

    async def check_and_reflect(self, date: str, horizon_days: int = 3):
        """
        Check outcomes for predictions made `horizon_days` ago.
        Then reflect on each outcome to extract learned rules.
        """
        check_date = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=horizon_days)).strftime("%Y-%m-%d")

        with sqlite3.connect(self.predictions_db) as conn:
            conn.row_factory = sqlite3.Row
            preds = conn.execute(
                "SELECT * FROM predictions WHERE date = ? AND was_correct IS NULL",
                (check_date,)
            ).fetchall()

        if not preds:
            return

        log.info(f"Checking {len(preds)} predictions from {check_date}...")

        for pred in preds:
            symbol = pred["symbol"]
            price_at_pred = pred["price_at_prediction"]

            if not price_at_pred:
                continue

            # Get current price
            try:
                df = self.tech_analyst.price_loader.load_ohlcv(symbol, "1d")
                if df.empty:
                    continue

                # Find price on check date
                target_date = datetime.strptime(date, "%Y-%m-%d")
                valid = df[df.index <= target_date]
                if valid.empty:
                    continue

                current_price = float(valid["close"].iloc[-1])
                actual_change = (current_price / price_at_pred - 1) * 100

                direction = pred["direction"]
                was_correct = (
                    (direction == "BUY" and actual_change > 0.5) or
                    (direction == "SELL" and actual_change < -0.5)
                )

                # Update prediction
                with sqlite3.connect(self.predictions_db) as conn:
                    conn.execute(
                        "UPDATE predictions SET actual_change=?, was_correct=? WHERE id=?",
                        (round(actual_change, 4), 1 if was_correct else 0, pred["id"])
                    )

                # REFLECT — ask the 235B model WHY
                reflection = await self.meta_learner.reflect(
                    symbol=symbol,
                    date=check_date,
                    prediction_direction=direction,
                    prediction_confidence=pred["confidence"] or 50,
                    prediction_reasoning=pred["reasoning"] or "",
                    actual_change_pct=actual_change,
                    news_data=json.loads(pred["news_signal"]) if pred["news_signal"] else None,
                    tech_data=json.loads(pred["tech_signal"]) if pred["tech_signal"] else None,
                )

                # Update event weights
                news_data = json.loads(pred["news_signal"]) if pred["news_signal"] else {}
                for event_type in news_data.get("event_types", []):
                    self.meta_learner.update_event_weights(event_type, was_correct)

                status = "CORRECT" if was_correct else "WRONG"
                log.info(f"  {symbol}: {status} (pred={direction}, actual={actual_change:+.2f}%)")

            except Exception as e:
                log.error(f"  {symbol}: check failed — {e}")

    async def run_historical_training(
        self,
        start_date: str,
        end_date: str,
        symbols: list[str] = None,
    ):
        """
        Run the full training loop over a date range.
        For each day: predict, then check outcomes from 3 days ago.
        """
        symbols = symbols or self.symbols
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        day_count = 0
        while current <= end_dt:
            date_str = current.strftime("%Y-%m-%d")

            # Skip weekends
            if current.weekday() < 5:
                # Check outcomes from 3 days ago
                await self.check_and_reflect(date_str, horizon_days=3)

                # Run today's predictions
                await self.run_historical_day(date_str, symbols)

                day_count += 1

                # Log progress every 5 days
                if day_count % 5 == 0:
                    stats = self.meta_learner.get_rule_stats()
                    log.info(f"Progress: {day_count} days | Rules: {stats}")

            current += timedelta(days=1)

        log.info(f"Training complete: {day_count} trading days processed")
        log.info(f"Final rule stats: {self.meta_learner.get_rule_stats()}")

    def _save_prediction(self, decision: Decision, date: str, news_data: dict,
                          tech_data: dict, bull_arg: str, bear_arg: str):
        """Save a prediction to the database."""
        price = tech_data.get("indicators", {}).get("last_close", 0)

        with sqlite3.connect(self.predictions_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO predictions
                (symbol, date, direction, confidence, reasoning, risk_factors,
                 news_signal, tech_signal, bull_argument, bear_argument, price_at_prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision.symbol, date, decision.direction, decision.confidence,
                decision.reasoning, json.dumps(decision.risk_factors),
                json.dumps(news_data), json.dumps(tech_data),
                bull_arg[:2000], bear_arg[:2000], price,
            ))

    def get_prediction_stats(self) -> dict:
        """Get overall prediction statistics."""
        with sqlite3.connect(self.predictions_db) as conn:
            total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            validated = conn.execute("SELECT COUNT(*) FROM predictions WHERE was_correct IS NOT NULL").fetchone()[0]
            correct = conn.execute("SELECT COUNT(*) FROM predictions WHERE was_correct = 1").fetchone()[0]

        return {
            "total_predictions": total,
            "validated": validated,
            "correct": correct,
            "win_rate": round(correct / max(validated, 1) * 100, 1),
        }


async def main():
    """CLI entry point for running the coordinator."""
    import sys

    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "NVDA", "TSLA", "BTC-USD", "EURUSD"]

    coordinator = Coordinator(
        llm_endpoint="http://localhost:8000",
        symbols=symbols,
    )

    # Run on historical data
    await coordinator.run_historical_training(
        start_date="2025-06-01",
        end_date="2025-12-31",
        symbols=symbols,
    )

    # Print final stats
    stats = coordinator.get_prediction_stats()
    log.info(f"Final stats: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
