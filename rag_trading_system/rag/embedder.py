"""
Embedder Module
===============
Creates and manages vector embeddings for RAG.
Uses Ollama's nomic-embed-text model.
"""

import json
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_DIR,
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_BATCH_SIZE
)

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Simple vector store using SQLite + numpy.
    Stores embeddings and performs similarity search.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(VECTOR_STORE_DIR / "vectors.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # News embeddings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id TEXT UNIQUE,
                title TEXT,
                content TEXT,
                source TEXT,
                published_date TEXT,
                embedding BLOB,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Trade memory embeddings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                pair TEXT,
                direction TEXT,
                outcome TEXT,
                description TEXT,
                lesson TEXT,
                trade_date TEXT,
                embedding BLOB,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Pattern embeddings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT UNIQUE,
                pattern_type TEXT,
                description TEXT,
                win_rate REAL,
                sample_count INTEGER,
                embedding BLOB,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_date
            ON news_embeddings(published_date)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trade_date
            ON trade_embeddings(trade_date)
        """)

        conn.commit()
        conn.close()

    def _serialize_embedding(self, embedding: List[float]) -> bytes:
        """Convert embedding to bytes for storage."""
        return np.array(embedding, dtype=np.float32).tobytes()

    def _deserialize_embedding(self, blob: bytes) -> np.ndarray:
        """Convert bytes back to numpy array."""
        return np.frombuffer(blob, dtype=np.float32)

    def add_news(
        self,
        article_id: str,
        title: str,
        content: str,
        source: str,
        published_date: str,
        embedding: List[float],
        metadata: Dict = None
    ):
        """Add news article embedding."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO news_embeddings
                (article_id, title, content, source, published_date, embedding, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                article_id,
                title,
                content[:5000],  # Truncate long content
                source,
                published_date,
                self._serialize_embedding(embedding),
                json.dumps(metadata or {})
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to add news: {e}")
        finally:
            conn.close()

    def add_trade(
        self,
        trade_id: str,
        pair: str,
        direction: str,
        outcome: str,
        description: str,
        lesson: str,
        trade_date: str,
        embedding: List[float],
        metadata: Dict = None
    ):
        """Add trade memory embedding."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO trade_embeddings
                (trade_id, pair, direction, outcome, description, lesson, trade_date, embedding, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id,
                pair,
                direction,
                outcome,
                description,
                lesson,
                trade_date,
                self._serialize_embedding(embedding),
                json.dumps(metadata or {})
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to add trade: {e}")
        finally:
            conn.close()

    def add_pattern(
        self,
        pattern_id: str,
        pattern_type: str,
        description: str,
        win_rate: float,
        sample_count: int,
        embedding: List[float],
        metadata: Dict = None
    ):
        """Add pattern embedding."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO pattern_embeddings
                (pattern_id, pattern_type, description, win_rate, sample_count, embedding, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern_id,
                pattern_type,
                description,
                win_rate,
                sample_count,
                self._serialize_embedding(embedding),
                json.dumps(metadata or {})
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to add pattern: {e}")
        finally:
            conn.close()

    def search_news(
        self,
        query_embedding: List[float],
        max_date: str = None,
        top_k: int = 10,
        min_similarity: float = 0.5
    ) -> List[Dict]:
        """
        Search for similar news articles.

        Args:
            query_embedding: Query vector
            max_date: Only return articles before this date (YYYY-MM-DD)
            top_k: Number of results to return
            min_similarity: Minimum cosine similarity threshold

        Returns:
            List of matching articles with similarity scores
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all embeddings (with date filter if specified)
        if max_date:
            cursor.execute("""
                SELECT article_id, title, content, source, published_date, embedding, metadata
                FROM news_embeddings
                WHERE published_date <= ?
            """, (max_date,))
        else:
            cursor.execute("""
                SELECT article_id, title, content, source, published_date, embedding, metadata
                FROM news_embeddings
            """)

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        # Calculate similarities
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)

        results = []
        for row in rows:
            article_id, title, content, source, pub_date, emb_blob, metadata = row
            emb_vec = self._deserialize_embedding(emb_blob)

            # Cosine similarity
            similarity = np.dot(query_vec, emb_vec) / (query_norm * np.linalg.norm(emb_vec))

            if similarity >= min_similarity:
                results.append({
                    "article_id": article_id,
                    "title": title,
                    "content": content,
                    "source": source,
                    "published_date": pub_date,
                    "similarity": float(similarity),
                    "metadata": json.loads(metadata) if metadata else {}
                })

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def search_trades(
        self,
        query_embedding: List[float],
        max_date: str = None,
        pair: str = None,
        top_k: int = 10
    ) -> List[Dict]:
        """Search for similar past trades."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT trade_id, pair, direction, outcome, description, lesson, trade_date, embedding, metadata FROM trade_embeddings WHERE 1=1"
        params = []

        if max_date:
            query += " AND trade_date <= ?"
            params.append(max_date)

        if pair:
            query += " AND pair = ?"
            params.append(pair)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)

        results = []
        for row in rows:
            trade_id, pair, direction, outcome, desc, lesson, trade_date, emb_blob, metadata = row
            emb_vec = self._deserialize_embedding(emb_blob)
            similarity = np.dot(query_vec, emb_vec) / (query_norm * np.linalg.norm(emb_vec))

            results.append({
                "trade_id": trade_id,
                "pair": pair,
                "direction": direction,
                "outcome": outcome,
                "description": desc,
                "lesson": lesson,
                "trade_date": trade_date,
                "similarity": float(similarity),
                "metadata": json.loads(metadata) if metadata else {}
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def search_patterns(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict]:
        """Search for similar patterns."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT pattern_id, pattern_type, description, win_rate, sample_count, embedding, metadata
            FROM pattern_embeddings
        """)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)

        results = []
        for row in rows:
            pattern_id, pattern_type, desc, win_rate, sample_count, emb_blob, metadata = row
            emb_vec = self._deserialize_embedding(emb_blob)
            similarity = np.dot(query_vec, emb_vec) / (query_norm * np.linalg.norm(emb_vec))

            results.append({
                "pattern_id": pattern_id,
                "pattern_type": pattern_type,
                "description": desc,
                "win_rate": win_rate,
                "sample_count": sample_count,
                "similarity": float(similarity),
                "metadata": json.loads(metadata) if metadata else {}
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def get_stats(self) -> Dict:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(*) FROM news_embeddings")
        stats["news_count"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM trade_embeddings")
        stats["trade_count"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM pattern_embeddings")
        stats["pattern_count"] = cursor.fetchone()[0]

        conn.close()
        return stats


class Embedder:
    """
    Creates embeddings using Ollama.
    """

    def __init__(self):
        # Import here to avoid circular imports
        from ensemble.ollama_client import get_client
        self.client = get_client()
        self.vector_store = VectorStore()

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        return self.client.embed(text)

    def embed_news_article(
        self,
        article_id: str,
        title: str,
        content: str,
        source: str,
        published_date: str,
        metadata: Dict = None
    ):
        """Embed and store a news article."""
        # Create embedding from title + content summary
        text_to_embed = f"{title}. {content[:1000]}"
        embedding = self.embed_text(text_to_embed)

        if embedding:
            self.vector_store.add_news(
                article_id=article_id,
                title=title,
                content=content,
                source=source,
                published_date=published_date,
                embedding=embedding,
                metadata=metadata
            )
            return True
        return False

    def embed_trade(
        self,
        trade_id: str,
        pair: str,
        direction: str,
        outcome: str,
        description: str,
        lesson: str,
        trade_date: str,
        metadata: Dict = None
    ):
        """Embed and store a trade."""
        text_to_embed = f"{pair} {direction} trade: {description}. Lesson: {lesson}"
        embedding = self.embed_text(text_to_embed)

        if embedding:
            self.vector_store.add_trade(
                trade_id=trade_id,
                pair=pair,
                direction=direction,
                outcome=outcome,
                description=description,
                lesson=lesson,
                trade_date=trade_date,
                embedding=embedding,
                metadata=metadata
            )
            return True
        return False

    def search_relevant_news(
        self,
        query: str,
        max_date: str = None,
        top_k: int = 10
    ) -> List[Dict]:
        """Search for relevant news articles."""
        query_embedding = self.embed_text(query)
        if not query_embedding:
            return []
        return self.vector_store.search_news(query_embedding, max_date, top_k)

    def search_similar_trades(
        self,
        query: str,
        max_date: str = None,
        pair: str = None,
        top_k: int = 10
    ) -> List[Dict]:
        """Search for similar past trades."""
        query_embedding = self.embed_text(query)
        if not query_embedding:
            return []
        return self.vector_store.search_trades(query_embedding, max_date, pair, top_k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test vector store
    store = VectorStore()
    print(f"Vector store initialized at: {store.db_path}")
    print(f"Stats: {store.get_stats()}")
