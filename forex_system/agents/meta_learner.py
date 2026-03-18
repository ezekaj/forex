"""Meta-Learner Agent — reflects on outcomes, extracts rules, updates weights."""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Optional

from forex_system.agents.base_agent import BaseAgent, AgentResult

log = logging.getLogger(__name__)

REFLECTION_SYSTEM = """You are a meta-analyst reviewing a trading prediction outcome.

Your job: extract ONE simple, GENERAL rule that applies to MANY future situations.

BAD rules (too specific, will never match again):
- "when RSI > 72.3 AND MACD hist = -2.21 AND AAPL news about iPhone sales in India"
- "when price is 3.17% above SMA20 on a Tuesday after earnings"

GOOD rules (general, reusable):
- "when RSI > 70 and trend is bearish → expect reversal down"
- "when strong positive earnings news and price above SMA20 → expect 1-3% gain over 3 days"
- "when all news is negative and price below SMA20 → expect continued decline"
- "when crypto RSI < 25 → oversold bounce is NOT reliable in downtrends"

Keep conditions to MAX 2-3 factors. The rule must apply to ANY asset, not just the one being analyzed.

Respond in JSON format."""


class MetaLearner(BaseAgent):
    """
    Reflects on prediction outcomes and extracts learned rules.
    Uses the 235B reasoning model for deep analysis of WHY trades succeeded or failed.
    Implements the SEP framework (Summarize-Explain-Predict) from arxiv 2402.03659.
    """

    def __init__(self, llm_endpoint: str = "http://localhost:8000", db_path: str = None):
        super().__init__(name="meta_learner", llm_endpoint=llm_endpoint, db_path=db_path)
        self._ensure_tables()

    def _ensure_tables(self):
        """Create learned_rules and reflections tables if they don't exist."""
        with self.get_db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learned_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    condition TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    samples INTEGER DEFAULT 0,
                    correct INTEGER DEFAULT 0,
                    asset_class TEXT,
                    event_type TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    UNIQUE(condition)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS event_weights (
                    event_type TEXT PRIMARY KEY,
                    weight REAL DEFAULT 0.0,
                    samples INTEGER DEFAULT 0,
                    updated_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reflections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    prediction_direction TEXT,
                    actual_direction TEXT,
                    was_correct INTEGER,
                    reasoning TEXT,
                    learned_rule TEXT,
                    confidence_calibration TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

    async def reflect(
        self,
        symbol: str,
        date: str,
        prediction_direction: str,
        prediction_confidence: int,
        prediction_reasoning: str,
        actual_change_pct: float,
        news_data: dict = None,
        tech_data: dict = None,
    ) -> dict:
        """
        Reflect on a prediction outcome. Ask the 235B model WHY it was right or wrong.
        Extract a learned rule for future use.
        """
        was_correct = (
            (prediction_direction == "BUY" and actual_change_pct > 0.5) or
            (prediction_direction == "SELL" and actual_change_pct < -0.5)
        )

        prompt = self._build_reflection_prompt(
            symbol, date, prediction_direction, prediction_confidence,
            prediction_reasoning, actual_change_pct, was_correct,
            news_data, tech_data,
        )

        response = await self.llm.ask_json(prompt, system=REFLECTION_SYSTEM, max_tokens=1500)

        # Save reflection
        with self.get_db() as conn:
            conn.execute("""
                INSERT INTO reflections (symbol, date, prediction_direction, actual_direction,
                    was_correct, reasoning, learned_rule, confidence_calibration, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, date, prediction_direction,
                "UP" if actual_change_pct > 0 else "DOWN",
                1 if was_correct else 0,
                response.get("primary_driver", ""),
                json.dumps(response.get("learned_rule", {})),
                response.get("confidence_calibration", ""),
                datetime.now().isoformat(),
            ))

        # Save learned rule if one was extracted
        rule = response.get("learned_rule", {})
        if rule and rule.get("condition"):
            self._save_or_update_rule(rule, was_correct)

        self.log_action(
            "reflection",
            f"{symbol} {date}: {'CORRECT' if was_correct else 'WRONG'} "
            f"(pred={prediction_direction}, actual={actual_change_pct:+.2f}%)"
        )

        return response

    def _save_or_update_rule(self, rule: dict, was_correct: bool):
        """Save a new rule or update existing one with new sample."""
        condition = rule.get("condition", "")
        prediction = rule.get("prediction", "")
        confidence = rule.get("confidence", 50)

        with self.get_db() as conn:
            existing = conn.execute(
                "SELECT id, samples, correct, confidence FROM learned_rules WHERE condition = ?",
                (condition,)
            ).fetchone()

            if existing:
                new_samples = existing[1] + 1
                new_correct = existing[2] + (1 if was_correct else 0)
                new_confidence = new_correct / new_samples * 100
                conn.execute(
                    "UPDATE learned_rules SET samples=?, correct=?, confidence=?, updated_at=? WHERE id=?",
                    (new_samples, new_correct, new_confidence, datetime.now().isoformat(), existing[0])
                )
            else:
                conn.execute("""
                    INSERT INTO learned_rules (condition, prediction, confidence, samples, correct,
                        created_at, updated_at)
                    VALUES (?, ?, ?, 1, ?, ?, ?)
                """, (
                    condition, prediction, confidence,
                    1 if was_correct else 0,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                ))

    def update_event_weights(self, event_type: str, was_correct: bool, learning_rate: float = 0.01):
        """Update event type weights based on outcome (gradient descent)."""
        with self.get_db() as conn:
            row = conn.execute(
                "SELECT weight FROM event_weights WHERE event_type = ?",
                (event_type,)
            ).fetchone()

            if row:
                old_weight = row[0]
                adjustment = learning_rate * (1.0 if was_correct else -1.0)
                new_weight = max(-0.5, min(0.5, old_weight + adjustment))
                conn.execute(
                    "UPDATE event_weights SET weight=?, samples=samples+1, updated_at=? WHERE event_type=?",
                    (new_weight, datetime.now().isoformat(), event_type)
                )
            else:
                weight = 0.05 if was_correct else -0.05
                conn.execute(
                    "INSERT INTO event_weights VALUES (?, ?, 1, ?)",
                    (event_type, weight, datetime.now().isoformat())
                )

    def get_rule_stats(self) -> dict:
        """Get summary of learned rules."""
        with self.get_db() as conn:
            try:
                total = conn.execute("SELECT COUNT(*) FROM learned_rules").fetchone()[0]
                high_conf = conn.execute(
                    "SELECT COUNT(*) FROM learned_rules WHERE confidence > 70 AND samples >= 5"
                ).fetchone()[0]
                avg_conf = conn.execute(
                    "SELECT AVG(confidence) FROM learned_rules WHERE samples >= 3"
                ).fetchone()[0] or 0
                return {"total_rules": total, "high_confidence": high_conf, "avg_confidence": round(avg_conf, 1)}
            except sqlite3.OperationalError:
                return {"total_rules": 0, "high_confidence": 0, "avg_confidence": 0}

    def _build_reflection_prompt(self, symbol, date, pred_dir, pred_conf, pred_reasoning,
                                  actual_change, was_correct, news, tech):
        result = "CORRECT" if was_correct else "WRONG"
        actual_dir = "UP" if actual_change > 0 else "DOWN"

        parts = [
            f"PREDICTION REVIEW for {symbol} on {date}",
            f"",
            f"PREDICTION: {pred_dir} with {pred_conf}% confidence",
            f"ACTUAL: {actual_dir} {actual_change:+.2f}%",
            f"RESULT: {result}",
            f"",
            f"ORIGINAL REASONING: {pred_reasoning[:500]}",
            f"",
        ]

        if news:
            parts.append(f"NEWS CONTEXT: {news.get('article_count', 0)} articles, "
                        f"avg sentiment={news.get('avg_sentiment', 0):.3f}, "
                        f"events={news.get('event_types', [])}")

        if tech:
            ind = tech.get("indicators", {})
            parts.append(f"TECHNICAL CONTEXT: RSI={ind.get('rsi', '?')}, "
                        f"trend={tech.get('trend', '?')}, "
                        f"5d return={ind.get('ret_5d', '?')}%")

        parts.extend([
            "",
            "ANALYZE:",
            "1. Why was the prediction " + result + "?",
            "2. Was confidence well-calibrated?",
            "3. What RULE can we extract for future use?",
            "",
            "Respond in JSON:",
            json.dumps({
                "primary_driver": "why the prediction was right/wrong",
                "confidence_calibration": "too_low / correct / too_high",
                "ideal_confidence": "0-100",
                "risk_assessment": "did risk factors materialize?",
                "learned_rule": {
                    "condition": "when [specific condition]",
                    "prediction": "expect [specific outcome]",
                    "confidence": "0-100",
                    "explanation": "because [reasoning]"
                }
            }, indent=2)
        ])

        return "\n".join(parts)
