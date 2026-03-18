"""Financial sentiment scoring using DistilRoBERTa (98.23% accuracy, 82M params)."""

import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MODEL_ID = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"


def _default_cache_path() -> str:
    if Path("/data/ssd").exists():
        return "/workspace/forex/forex_system/training/artifacts/sentiment_cache_v2.db"
    return "/home/user/forex/forex_system/training/artifacts/sentiment_cache_v2.db"


class SentimentScorer:
    """
    Scores financial text as positive/negative/neutral using DistilRoBERTa.
    Results cached permanently in SQLite.
    """

    def __init__(self, cache_db: str = None, device: str = "auto", batch_size: int = 128):
        self.cache_db = cache_db or _default_cache_path()
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._tokenizer = None
        self._init_cache()

    def _init_cache(self):
        Path(self.cache_db).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scores (
                    text_hash TEXT PRIMARY KEY,
                    positive REAL,
                    negative REAL,
                    neutral REAL,
                    label TEXT,
                    score REAL
                )
            """)

    def _load_model(self):
        if self._model is not None:
            return
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self._model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self.device)
        self._model.eval()
        log.info(f"SentimentScorer loaded on {self.device}")

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()

    def score_batch(self, texts: list[str]) -> list[dict]:
        """Score a batch of texts. Uses cache for already-scored texts."""
        import torch

        self._load_model()

        hashes = [self._hash(t) for t in texts]

        # Check cache
        cached = {}
        with sqlite3.connect(self.cache_db) as conn:
            for h in hashes:
                row = conn.execute("SELECT * FROM scores WHERE text_hash = ?", (h,)).fetchone()
                if row:
                    cached[h] = {
                        "positive": row[1], "negative": row[2], "neutral": row[3],
                        "label": row[4], "score": row[5],
                    }

        # Score uncached
        uncached_idx = [i for i, h in enumerate(hashes) if h not in cached]
        if uncached_idx:
            uncached_texts = [texts[i] for i in uncached_idx]
            new_scores = {}

            for batch_start in range(0, len(uncached_texts), self.batch_size):
                batch = uncached_texts[batch_start:batch_start + self.batch_size]
                batch_hashes = [hashes[uncached_idx[batch_start + j]] for j in range(len(batch))]

                inputs = self._tokenizer(
                    batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self._model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

                # DistilRoBERTa sentiment: 0=negative, 1=neutral, 2=positive
                for j, (h, prob) in enumerate(zip(batch_hashes, probs)):
                    label_idx = int(np.argmax(prob))
                    labels = ["negative", "neutral", "positive"]
                    result = {
                        "negative": float(prob[0]),
                        "neutral": float(prob[1]),
                        "positive": float(prob[2]),
                        "label": labels[label_idx],
                        "score": float(prob[2] - prob[0]),
                    }
                    new_scores[h] = result

            # Save to cache
            with sqlite3.connect(self.cache_db) as conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO scores VALUES (?,?,?,?,?,?)",
                    [(h, r["positive"], r["negative"], r["neutral"], r["label"], r["score"])
                     for h, r in new_scores.items()],
                )
            cached.update(new_scores)

        # Return in order
        default = {"positive": 0, "negative": 0, "neutral": 1, "label": "neutral", "score": 0}
        return [cached.get(h, default) for h in hashes]

    def score_dataframe(self, df: pd.DataFrame, text_col: str = "title") -> pd.DataFrame:
        """Score a DataFrame and add sentiment columns."""
        texts = df[text_col].fillna("").tolist()
        scores = self.score_batch(texts)
        result = df.copy()
        result["sent_positive"] = [s["positive"] for s in scores]
        result["sent_negative"] = [s["negative"] for s in scores]
        result["sent_neutral"] = [s["neutral"] for s in scores]
        result["sent_label"] = [s["label"] for s in scores]
        result["sent_score"] = [s["score"] for s in scores]
        return result

    def get_cache_stats(self) -> dict:
        with sqlite3.connect(self.cache_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        return {"cached": count, "db": self.cache_db}
