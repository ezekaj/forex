"""
Asset model for multi-asset trading platform.
Supports crypto, forex, and stocks.
"""

from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime


class AssetType(str, Enum):
    """Asset type classification"""
    CRYPTO = "crypto"
    FOREX = "forex"
    STOCK = "stock"
    CFD = "cfd"


class Candle(BaseModel):
    """OHLCV candle data"""
    time: int = Field(..., description="Timestamp in milliseconds")
    open: float = Field(..., description="Open price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Close price")
    volume: float = Field(default=0, description="Volume")


class Asset(BaseModel):
    """
    Unified asset model supporting multiple asset types.
    Compatible with both AlphaThink frontend and broker systems.
    """
    symbol: str = Field(..., description="Asset symbol (e.g., BTCUSDT, EURUSD, AAPL)")
    name: Optional[str] = Field(None, description="Human-readable name")
    type: AssetType = Field(..., description="Asset type (crypto/forex/stock)")
    exchange: Optional[str] = Field(None, description="Exchange or broker")

    # Current market data
    price: float = Field(..., description="Current price")
    change: float = Field(default=0.0, description="Price change percentage (24h)")
    volume: float = Field(default=0.0, description="24h volume")

    # Historical data
    history: List[Candle] = Field(default_factory=list, description="Historical candle data")

    # Metadata
    decimals: int = Field(default=2, description="Price decimal places")
    min_size: float = Field(default=0.001, description="Minimum order size")
    max_size: Optional[float] = Field(None, description="Maximum order size")
    active: bool = Field(default=True, description="Whether asset is actively traded")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTCUSDT",
                "name": "Bitcoin",
                "type": "crypto",
                "exchange": "Binance",
                "price": 45000.0,
                "change": 2.5,
                "volume": 1000000.0,
                "history": [
                    {
                        "time": 1700000000000,
                        "open": 44900.0,
                        "high": 45200.0,
                        "low": 44800.0,
                        "close": 45000.0,
                        "volume": 500.0
                    }
                ],
                "decimals": 2,
                "min_size": 0.001,
                "active": True
            }
        }


class AssetIndicators(BaseModel):
    """Technical indicators for an asset"""
    symbol: str
    timestamp: datetime

    # Trend indicators
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None

    # Volatility indicators
    atr: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None

    # Momentum indicators
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None

    # Volume indicators
    obv: Optional[float] = None
    volume_sma: Optional[float] = None

    # Strength indicators
    adx: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None

    # Pivot points
    pivot: Optional[float] = None
    r1: Optional[float] = None
    r2: Optional[float] = None
    r3: Optional[float] = None
    s1: Optional[float] = None
    s2: Optional[float] = None
    s3: Optional[float] = None


class MarketRegime(str, Enum):
    """Market regime classification"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CONSOLIDATING = "consolidating"
    BREAKOUT = "breakout"
    UNKNOWN = "unknown"


class AssetAnalysis(BaseModel):
    """Comprehensive asset analysis"""
    symbol: str
    timestamp: datetime

    # Market data
    current_price: float
    price_change_24h: float

    # Indicators
    indicators: AssetIndicators

    # Market regime
    regime: MarketRegime
    regime_confidence: float = Field(ge=0, le=100)

    # Patterns detected
    patterns: List[str] = Field(default_factory=list)

    # Support and resistance levels
    support_levels: List[float] = Field(default_factory=list)
    resistance_levels: List[float] = Field(default_factory=list)

    # Signals
    buy_signals: List[str] = Field(default_factory=list)
    sell_signals: List[str] = Field(default_factory=list)

    # Overall sentiment
    sentiment: str = Field(default="neutral")  # bullish, bearish, neutral
    sentiment_score: float = Field(default=0.0, ge=-100, le=100)
