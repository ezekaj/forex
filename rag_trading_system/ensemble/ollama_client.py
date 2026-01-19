"""
Ollama Client
=============
Unified interface for all Ollama models.
"""

import json
import base64
import requests
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OLLAMA_HOST,
    OLLAMA_TIMEOUT,
    MAIN_MODEL,
    MAIN_MODEL_FALLBACK,
    VISION_MODEL,
    VISION_MODEL_FALLBACK,
    EMBEDDING_MODEL
)

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama models."""

    def __init__(self, host: str = None):
        self.host = host or OLLAMA_HOST
        self.available_models = self._get_available_models()

        # Select best available models
        self.main_model = self._select_model([MAIN_MODEL, MAIN_MODEL_FALLBACK])
        self.vision_model = self._select_model([VISION_MODEL, VISION_MODEL_FALLBACK])
        self.embedding_model = EMBEDDING_MODEL

        logger.info(f"Main model: {self.main_model}")
        logger.info(f"Vision model: {self.vision_model}")
        logger.info(f"Embedding model: {self.embedding_model}")

    def _get_available_models(self) -> List[str]:
        """Get list of available models."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
        return []

    def _select_model(self, preferences: List[str]) -> Optional[str]:
        """Select best available model from preferences."""
        for model in preferences:
            # Check exact match or partial match
            for available in self.available_models:
                if model in available or available.startswith(model.split(":")[0]):
                    return available
        return preferences[-1] if preferences else None

    def generate(
        self,
        prompt: str,
        model: str = None,
        system: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False
    ) -> str:
        """Generate text response from LLM."""
        model = model or self.main_model

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        if system:
            payload["system"] = system

        if json_mode:
            payload["format"] = "json"

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT
            )

            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Ollama error: {response.status_code} - {response.text}")
                return ""

        except requests.exceptions.Timeout:
            logger.error(f"Ollama timeout after {OLLAMA_TIMEOUT}s")
            return ""
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return ""

    def generate_with_image(
        self,
        prompt: str,
        image_path: str,
        model: str = None,
        temperature: float = 0.7
    ) -> str:
        """Generate response from vision model with image input."""
        model = model or self.vision_model

        if not model:
            logger.error("No vision model available")
            return ""

        # Read and encode image
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to read image: {e}")
            return ""

        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_data],
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT * 2  # Vision takes longer
            )

            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Ollama vision error: {response.status_code}")
                return ""

        except Exception as e:
            logger.error(f"Ollama vision error: {e}")
            return ""

    def embed(self, text: str, model: str = None) -> List[float]:
        """Generate embedding for text."""
        model = model or self.embedding_model

        payload = {
            "model": model,
            "prompt": text,
        }

        try:
            response = requests.post(
                f"{self.host}/api/embeddings",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                return response.json().get("embedding", [])
            else:
                logger.error(f"Embedding error: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return []

    def embed_batch(self, texts: List[str], model: str = None) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text, model) for text in texts]

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.7,
        json_mode: bool = False
    ) -> str:
        """Chat with the model using message history."""
        model = model or self.main_model

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }

        if json_mode:
            payload["format"] = "json"

        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=OLLAMA_TIMEOUT
            )

            if response.status_code == 200:
                return response.json().get("message", {}).get("content", "")
            else:
                logger.error(f"Chat error: {response.status_code}")
                return ""

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return ""

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


# Singleton instance
_client = None

def get_client() -> OllamaClient:
    """Get or create Ollama client instance."""
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client


if __name__ == "__main__":
    # Test the client
    logging.basicConfig(level=logging.INFO)

    client = get_client()
    print(f"\nAvailable models: {client.available_models}")
    print(f"Main model: {client.main_model}")
    print(f"Vision model: {client.vision_model}")

    if client.is_available():
        print("\n--- Testing text generation ---")
        response = client.generate(
            "What are the key factors that move EUR/USD? Be brief.",
            temperature=0.5
        )
        print(f"Response: {response[:500]}...")

        print("\n--- Testing embedding ---")
        embedding = client.embed("EUR/USD bullish momentum")
        print(f"Embedding dimension: {len(embedding)}")
    else:
        print("Ollama not available!")
