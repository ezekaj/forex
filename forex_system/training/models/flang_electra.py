"""Financial sentiment scorer using ProsusAI/finbert (3-class: positive/negative/neutral)."""

import sqlite3
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


import os as _os
if _os.path.exists("/data/ssd"):
    SENTIMENT_CACHE_DB = "/workspace/forex/forex_system/training/artifacts/sentiment_cache.db"
else:
    SENTIMENT_CACHE_DB = "/home/user/forex/forex_system/training/artifacts/sentiment_cache.db"


class FlangElectraSentiment:
    """
    Wraps SALT-NLP/FLANG-ELECTRA for financial sentiment analysis.
    Scores: positive, negative, neutral probabilities + composite score.
    Results cached to SQLite so we only score each article once.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        cache_db: str = SENTIMENT_CACHE_DB,
        device: str = "auto",
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.cache_db = cache_db
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._tokenizer = None
        self._init_cache()

    def _init_cache(self):
        Path(self.cache_db).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_scores (
                    text_hash TEXT PRIMARY KEY,
                    positive REAL,
                    negative REAL,
                    neutral REAL,
                    score REAL
                )
            """)

    def _load_model(self):
        if self._model is not None:
            return

        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self.device)
        self._model.eval()

    def _text_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()

    def _get_cached(self, text_hashes: list[str]) -> dict[str, dict]:
        if not text_hashes:
            return {}
        with sqlite3.connect(self.cache_db) as conn:
            placeholders = ",".join(["?"] * len(text_hashes))
            cursor = conn.execute(
                f"SELECT text_hash, positive, negative, neutral, score FROM sentiment_scores WHERE text_hash IN ({placeholders})",
                text_hashes,
            )
            return {
                row[0]: {"positive": row[1], "negative": row[2], "neutral": row[3], "score": row[4]}
                for row in cursor.fetchall()
            }

    def _save_cached(self, results: dict[str, dict]):
        if not results:
            return
        with sqlite3.connect(self.cache_db) as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO sentiment_scores VALUES (?,?,?,?,?)",
                [
                    (h, r["positive"], r["negative"], r["neutral"], r["score"])
                    for h, r in results.items()
                ],
            )

    def score_texts(self, texts: list[str]) -> pd.DataFrame:
        """
        Score a list of texts for financial sentiment.
        Returns DataFrame with columns: positive, negative, neutral, score
        """
        import torch

        self._load_model()

        hashes = [self._text_hash(t) for t in texts]
        cached = self._get_cached(hashes)

        uncached_indices = [i for i, h in enumerate(hashes) if h not in cached]
        uncached_texts = [texts[i] for i in uncached_indices]

        new_results = {}
        if uncached_texts:
            for batch_start in range(0, len(uncached_texts), self.batch_size):
                batch = uncached_texts[batch_start:batch_start + self.batch_size]
                batch_hashes = [hashes[uncached_indices[batch_start + j]] for j in range(len(batch))]

                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    outputs = self._model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

                for j, (h, prob) in enumerate(zip(batch_hashes, probs)):
                    # ProsusAI/finbert: 0=positive, 1=negative, 2=neutral
                    result = {
                        "positive": float(prob[0]),
                        "negative": float(prob[1]),
                        "neutral": float(prob[2]),
                        "score": float(prob[0] - prob[1]),
                    }
                    new_results[h] = result

            self._save_cached(new_results)

        all_results = []
        for h in hashes:
            if h in cached:
                all_results.append(cached[h])
            elif h in new_results:
                all_results.append(new_results[h])
            else:
                all_results.append({"positive": 0.0, "negative": 0.0, "neutral": 1.0, "score": 0.0})

        return pd.DataFrame(all_results)

    def score_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "title",
        prefix: str = "sent_",
    ) -> pd.DataFrame:
        """Score a DataFrame's text column and add sentiment columns."""
        texts = df[text_column].fillna("").tolist()
        scores = self.score_texts(texts)
        result = df.copy()
        for col in scores.columns:
            result[f"{prefix}{col}"] = scores[col].values
        return result

    def get_cache_stats(self) -> dict:
        with sqlite3.connect(self.cache_db) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM sentiment_scores")
            count = cursor.fetchone()[0]
        return {"cached_scores": count, "cache_db": self.cache_db}
