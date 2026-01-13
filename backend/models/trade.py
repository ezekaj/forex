"""
Trade model for the unified trading platform.
Compatible with AlphaThink and broker systems.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


class TradeAction(str, Enum):
    """Trade action type"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradeStatus(str, Enum):
    """Trade execution status"""
    PENDING = "pending"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TradeType(str, Enum):
    """Type of trade"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class Trade(BaseModel):
    """
    Trade execution record.
    Tracks individual buy/sell transactions.
    """
    id: str = Field(..., description="Unique trade ID")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Asset info
    symbol: str = Field(..., description="Asset symbol")

    # Trade details
    action: TradeAction = Field(..., description="Buy or Sell")
    type: TradeType = Field(default=TradeType.MARKET, description="Order type")
    status: TradeStatus = Field(default=TradeStatus.PENDING, description="Execution status")

    # Pricing
    price: float = Field(..., description="Execution price")
    amount: float = Field(..., description="Quantity traded")
    value: float = Field(..., description="Total value (price * amount)")

    # Fees
    fee: float = Field(default=0.0, description="Trading fee")
    fee_currency: str = Field(default="USD", description="Fee currency")

    # Risk parameters (if set)
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")

    # AI decision context
    reasoning: Optional[str] = Field(None, description="AI reasoning for trade")
    confidence: Optional[float] = Field(None, ge=0, le=100, description="AI confidence level")
    strategy_used: Optional[str] = Field(None, description="Trading strategy applied")
    market_phase: Optional[str] = Field(None, description="Market phase at trade time")

    # P&L (for sells)
    pnl: Optional[float] = Field(None, description="Profit/loss (for closing trades)")
    pnl_percent: Optional[float] = Field(None, description="P&L percentage")

    # Execution details
    execution_time: Optional[datetime] = Field(None, description="Actual execution timestamp")
    slippage: Optional[float] = Field(None, description="Price slippage")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "trade_123",
                "timestamp": "2024-01-14T12:00:00",
                "symbol": "BTCUSDT",
                "action": "BUY",
                "type": "market",
                "status": "executed",
                "price": 45000.0,
                "amount": 0.1,
                "value": 4500.0,
                "fee": 4.5,
                "fee_currency": "USD",
                "stop_loss": 43000.0,
                "take_profit": 48000.0,
                "reasoning": "Strong bullish momentum with RSI confirmation",
                "confidence": 85.5,
                "strategy_used": "Trend Following",
                "market_phase": "Trending Up"
            }
        }


class TradeDecision(BaseModel):
    """
    AI-generated trading decision.
    Compatible with AlphaThink's Gemini analysis.
    """
    # Core decision
    action: TradeAction = Field(..., description="Recommended action")
    confidence: float = Field(..., ge=0, le=100, description="Confidence level (0-100)")

    # Position sizing
    suggested_position_size: float = Field(
        ...,
        ge=0,
        le=1,
        description="Fraction of available balance (0.0 to 1.0)"
    )

    # Market context
    market_phase: str = Field(..., description="Current market regime")
    reasoning: str = Field(..., description="Detailed reasoning for decision")
    strategy_used: str = Field(..., description="Primary strategy applied")
    fallback_strategy: str = Field(..., description="Alternative strategy if primary fails")

    # Key factors
    key_factors: list[str] = Field(default_factory=list, description="Important decision factors")

    # Risk management
    stop_loss: float = Field(..., description="Suggested stop loss price")
    take_profit: float = Field(..., description="Suggested take profit price")

    # Additional metrics
    risk_reward_ratio: Optional[float] = Field(None, description="Risk/reward ratio")
    win_probability: Optional[float] = Field(None, ge=0, le=100, description="Estimated win probability")

    # Agent consensus (if using multi-agent)
    agent_votes: Optional[dict] = Field(None, description="Individual agent recommendations")
    consensus_strength: Optional[float] = Field(None, ge=0, le=100, description="Agreement level among agents")

    class Config:
        json_schema_extra = {
            "example": {
                "action": "BUY",
                "confidence": 85.5,
                "suggested_position_size": 0.3,
                "market_phase": "Trending Up",
                "reasoning": "Strong bullish momentum confirmed by multiple indicators. RSI shows strength without overbought conditions. Price broke resistance at $44,500 with high volume.",
                "strategy_used": "Trend Following + Breakout",
                "fallback_strategy": "Mean Reversion",
                "key_factors": [
                    "RSI: 65 (bullish momentum)",
                    "ADX: 32 (strong trend)",
                    "Volume: +45% above average",
                    "Price above 20 SMA with upward slope"
                ],
                "stop_loss": 43000.0,
                "take_profit": 48000.0,
                "risk_reward_ratio": 2.5,
                "win_probability": 68.0
            }
        }


class TradeLogEntry(BaseModel):
    """
    Trade log entry for history tracking.
    Used by frontend to display trade history.
    """
    id: str
    timestamp: int  # milliseconds
    symbol: str
    action: TradeAction
    price: float
    amount: Optional[float] = None
    reasoning: Optional[str] = None
    pnl: Optional[float] = None
    market_context: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "id": "log_123",
                "timestamp": 1700000000000,
                "symbol": "BTCUSDT",
                "action": "SELL",
                "price": 48000.0,
                "amount": 0.1,
                "reasoning": "Take profit target reached",
                "pnl": 300.0,
                "market_context": "Entry(Trending Up)->Exit(Trending Up) | Strat: Trend Following"
            }
        }
