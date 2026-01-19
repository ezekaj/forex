"""
RAG Trading System Configuration
================================
All settings for the trading bot.
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "databases"
MODELS_DIR = BASE_DIR / "models"
CHARTS_DIR = BASE_DIR / "charts"
RESULTS_DIR = BASE_DIR / "results"
VECTOR_STORE_DIR = BASE_DIR / "2_rag" / "vector_store"

# News data from news_lenci_forex
NEWS_LENCI_DIR = Path.home() / "Desktop" / "news_lenci_forex"
NEWS_DB_PATH = NEWS_LENCI_DIR / "historical_news.db"

# =============================================================================
# MODELS (Ollama)
# =============================================================================

# Main reasoning model
MAIN_MODEL = "qwen2.5:14b"  # Can upgrade to 32b if RAM allows
MAIN_MODEL_FALLBACK = "qwen2.5:7b"

# Vision model for chart analysis
VISION_MODEL = "llava:13b"  # or "qwen2-vl" when available
VISION_MODEL_FALLBACK = "llava:7b"

# Embedding model for RAG
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSION = 768

# Ollama settings
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TIMEOUT = 120  # seconds

# =============================================================================
# TRADING PAIRS
# =============================================================================

FOREX_PAIRS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCHF",
    "USDCAD",
    "NZDUSD",
]

# Primary pairs to focus on
PRIMARY_PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]

# =============================================================================
# RAG SETTINGS
# =============================================================================

# How many articles to retrieve per query
RAG_TOP_K = 10

# How many past trades to retrieve
TRADE_MEMORY_K = 20

# How many patterns to retrieve
PATTERN_MEMORY_K = 10

# Embedding batch size
EMBEDDING_BATCH_SIZE = 100

# =============================================================================
# SIMULATION SETTINGS
# =============================================================================

# Simulation date range (must be within historical data: 2024-01-01 to 2026-01-16)
SIM_START_DATE = "2024-06-01"  # Start 6 months in to allow indicator warmup
SIM_END_DATE = "2025-06-01"    # 1 year simulation period

# How often to make decisions (in hours)
# Use 24 for daily data, 4 for 4H data
DECISION_INTERVAL_HOURS = 24

# Starting capital for simulation
SIM_STARTING_CAPITAL = 30000

# =============================================================================
# RISK MANAGEMENT
# =============================================================================

# Maximum risk per trade (% of capital)
MAX_RISK_PER_TRADE = 0.01  # 1% = $300 on $30K

# Maximum concurrent positions
MAX_POSITIONS = 3

# Minimum confidence to trade
MIN_CONFIDENCE = 0.60  # 60%

# Minimum ensemble agreement (out of 4 models)
MIN_ENSEMBLE_AGREEMENT = 3

# Stop loss multiplier (ATR)
SL_ATR_MULTIPLIER = 1.5

# Take profit multiplier (ATR)
TP_ATR_MULTIPLIER = 2.5

# Daily loss limit (% of capital)
DAILY_LOSS_LIMIT = 0.02  # 2%

# Maximum drawdown before stopping
MAX_DRAWDOWN = 0.10  # 10%

# =============================================================================
# BROKER (OANDA)
# =============================================================================

OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")
OANDA_ACCESS_TOKEN = os.getenv("OANDA_ACCESS_TOKEN", "")
OANDA_ENVIRONMENT = "practice"  # "practice" or "live"

# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

INDICATORS = {
    "RSI": {"period": 14},
    "MACD": {"fast": 12, "slow": 26, "signal": 9},
    "ATR": {"period": 14},
    "BB": {"period": 20, "std": 2},
    "EMA": {"periods": [9, 21, 50, 200]},
    "ADX": {"period": 14},
    "STOCH": {"k_period": 14, "d_period": 3},
}

# =============================================================================
# CHART GENERATION
# =============================================================================

CHART_CANDLES = 100  # Number of candles to show
CHART_WIDTH = 1200
CHART_HEIGHT = 800
CHART_STYLE = "charles"  # mplfinance style

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = "INFO"
LOG_FILE = BASE_DIR / "trading.log"

# =============================================================================
# ENSEMBLE WEIGHTS
# =============================================================================

# How much each model's vote counts
ENSEMBLE_WEIGHTS = {
    "main_reasoning": 0.35,
    "chart_vision": 0.25,
    "pattern_detector": 0.20,
    "sentiment_analyzer": 0.20,
}
