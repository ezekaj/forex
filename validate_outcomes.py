#!/usr/bin/env python3
"""Backfill prediction outcomes against actual Yahoo Finance prices."""

import sqlite3
import yfinance as yf
from datetime import datetime, timedelta

DB = "/workspace/investment-monitor/data/predictions.db"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

preds = conn.execute(
    "SELECT id, ticker, predicted_date, predicted_direction, confidence FROM predictions"
).fetchall()
print("Validating", len(preds), "predictions...")

validated = 0
correct = 0

for pred in preds:
    pred_id = pred[0]
    ticker = pred[1]
    pred_date = pred[2]
    direction = pred[3]

    try:
        start = datetime.strptime(pred_date, "%Y-%m-%d")
        end = start + timedelta(days=10)

        t = yf.Ticker(ticker)
        data = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))

        if len(data) < 2:
            print(f"  {ticker}: insufficient data")
            continue

        entry_price = float(data["Close"].iloc[0])
        best_idx = min(3, len(data) - 1)
        exit_price = float(data["Close"].iloc[best_idx])
        actual_change = (exit_price / entry_price - 1) * 100

        if direction == "UP":
            was_correct = 1 if actual_change > 0.5 else 0
        else:
            was_correct = 1 if actual_change < -0.5 else 0

        conn.execute(
            "UPDATE predictions SET actual_change = ?, was_correct = ? WHERE id = ?",
            (round(actual_change, 4), was_correct, pred_id),
        )

        validated += 1
        if was_correct:
            correct += 1

        status = "CORRECT" if was_correct else "WRONG"
        print(f"  {ticker:12s} | pred={direction:4s} | actual={actual_change:+.2f}% | {status}")

    except Exception as e:
        print(f"  {ticker}: ERROR - {e}")

conn.commit()
conn.close()

print()
print("=" * 50)
print(f"Validated: {validated}/{len(preds)}")
if validated > 0:
    pct = correct / validated * 100
    print(f"Correct: {correct}/{validated} ({pct:.1f}%)")
print("=" * 50)
