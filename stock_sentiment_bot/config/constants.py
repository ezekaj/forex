"""
All magic numbers and constants centralized.
Based on Quarter-Kelly position sizing with circuit breakers.
"""

# =============================================================================
# POSITION SIZING (Quarter-Kelly)
# =============================================================================

# Base risk per trade (Quarter-Kelly optimal for 55% win rate, 1.5:1 R:R)
BASE_RISK_PERCENT = 0.04  # 4% of account per trade

# Risk bounds (never exceed these regardless of confidence)
MAX_RISK_PERCENT = 0.06   # 6% max even on highest conviction
MIN_RISK_PERCENT = 0.01   # 1% min to stay in the game

# Confidence-based scaling
CONFIDENCE_RISK_MAP = {
    # signal_score range: risk_multiplier
    (90, 100): 1.25,    # 5% risk (high conviction)
    (80, 89): 1.0,      # 4% risk (standard)
    (70, 79): 0.75,     # 3% risk (moderate)
    (60, 69): 0.50,     # 2% risk (low conviction)
}


# =============================================================================
# CIRCUIT BREAKERS (Hard limits - CANNOT be overridden)
# =============================================================================

# Daily limits
MAX_DAILY_LOSS_PERCENT = 0.10    # Stop trading at -10% daily loss
MAX_DAILY_TRADES = 5              # Prevent overtrading

# Drawdown limits
WARNING_DRAWDOWN = 0.15           # Reduce position size at -15%
HALT_DRAWDOWN = 0.25              # Full trading halt at -25%
REVIEW_DRAWDOWN = 0.30            # Requires manual review at -30%

# Consecutive loss limits
REDUCE_AFTER_LOSSES = 3           # Cut size 50% after 3 consecutive losses
HALT_AFTER_LOSSES = 7             # Stop trading after 7 consecutive losses

# Position limits
MAX_POSITIONS = 3                 # Maximum concurrent positions
MAX_SINGLE_POSITION_PERCENT = 0.35  # Max 35% of account in one stock
MAX_SECTOR_EXPOSURE_PERCENT = 0.50  # Max 50% in one sector


# =============================================================================
# DRAWDOWN RECOVERY PROTOCOL
# =============================================================================

DRAWDOWN_RISK_REDUCTION = {
    # drawdown_range: risk_multiplier
    (0.00, 0.10): 1.0,    # Normal trading
    (0.10, 0.15): 0.50,   # 50% risk reduction
    (0.15, 0.20): 0.25,   # 75% risk reduction
    (0.20, 0.25): 0.0,    # Trading halted
}


# =============================================================================
# SIGNAL SCORING THRESHOLDS
# =============================================================================

MIN_SIGNAL_SCORE = 60             # Don't trade below this score

# Sentiment thresholds
MENTION_SPIKE_MULTIPLIER = 3.0    # mentions_1h > 3x avg_24h = spike
BULLISH_SENTIMENT_THRESHOLD = 0.6  # Score > 0.6 = strong bullish
BEARISH_SENTIMENT_THRESHOLD = 0.4  # Score < 0.4 = strong bearish

# Quality filters
MIN_UNIQUE_AUTHORS = 20           # Avoid bot manipulation
MIN_ENGAGEMENT_SCORE = 10         # Minimum upvotes + comments

# Technical filters
RSI_OVERSOLD = 40                 # Consider oversold below this
RSI_OVERBOUGHT = 60               # Consider overbought above this
VOLUME_SPIKE_MULTIPLIER = 2.0     # volume > 2x avg = confirmation


# =============================================================================
# TRADE EXECUTION
# =============================================================================

# Reward:Risk targets
MIN_REWARD_RISK_RATIO = 1.5       # Don't take trades below 1.5:1
TARGET_REWARD_RISK_RATIO = 2.0    # Aim for 2:1

# Stop loss
STOP_LOSS_ATR_MULTIPLIER = 1.5    # 1.5x ATR below entry

# Take profit scaling
TAKE_PROFIT_LEVELS = {
    1.5: 0.50,    # Take 50% profit at 1.5R
    2.5: 0.30,    # Take 30% profit at 2.5R
    4.0: 0.20,    # Let 20% run to 4R (moon shots)
}


# =============================================================================
# DATA COLLECTION
# =============================================================================

# Reddit scraping
REDDIT_RATE_LIMIT_SECONDS = 2     # Minimum seconds between API calls
REDDIT_MAX_POSTS_PER_SUB = 100    # Posts to fetch per subreddit
REDDIT_CACHE_TTL_SECONDS = 300    # Cache data for 5 minutes

# News scraping
NEWS_CACHE_TTL_SECONDS = 600      # Cache news for 10 minutes
NEWS_MAX_ARTICLES = 50            # Max articles per ticker

# Price data
PRICE_CACHE_TTL_SECONDS = 60      # Cache prices for 1 minute
HISTORICAL_BARS_LIMIT = 100       # Default historical bars to fetch


# =============================================================================
# SCHEDULING
# =============================================================================

# Market hours (Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# Pre-market analysis
PRE_MARKET_ANALYSIS_HOUR = 9
PRE_MARKET_ANALYSIS_MINUTE = 0

# Scraping intervals (minutes)
REDDIT_SCRAPE_INTERVAL = 15
NEWS_SCRAPE_INTERVAL = 30
POSITION_CHECK_INTERVAL = 5


# =============================================================================
# WATCHLIST
# =============================================================================

# Default tickers to always monitor (high-liquidity meme stocks)
DEFAULT_WATCHLIST = [
    "GME", "AMC", "BBBY", "PLTR", "NVDA",
    "TSLA", "AAPL", "AMD", "SOFI", "SPY",
]

# Sectors for exposure tracking
SECTOR_MAP = {
    "tech": ["NVDA", "AMD", "AAPL", "TSLA", "PLTR"],
    "meme": ["GME", "AMC", "BBBY"],
    "finance": ["SOFI"],
    "etf": ["SPY"],
}


# =============================================================================
# LOGGING
# =============================================================================

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE_MAX_BYTES = 10_000_000   # 10MB
LOG_FILE_BACKUP_COUNT = 5
