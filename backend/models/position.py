"""
Position model for the unified trading platform.
Tracks open positions across all asset types.
"""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
from .trade import TradeAction


class Position(BaseModel):
    """
    Open position tracking.
    Compatible with AlphaThink frontend and broker systems.
    """
    # Asset info
    symbol: str = Field(..., description="Asset symbol")

    # Position details
    entry_price: float = Field(..., description="Average entry price")
    amount: float = Field(..., description="Position size (quantity)")
    side: TradeAction = Field(..., description="Long (BUY) or Short (SELL)")

    # Current value
    current_value: float = Field(..., description="Current position value")
    entry_value: float = Field(..., description="Initial position value")

    # P&L
    pnl: float = Field(default=0.0, description="Unrealized profit/loss")
    pnl_percent: float = Field(default=0.0, description="P&L percentage")

    # Risk management
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")

    # Context
    entry_market_phase: Optional[str] = Field(None, description="Market phase at entry")
    entry_reasoning: Optional[str] = Field(None, description="Reason for opening position")
    strategy: Optional[str] = Field(None, description="Strategy used")

    # Timestamps
    opened_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Position management
    max_pnl: float = Field(default=0.0, description="Maximum P&L achieved")
    max_drawdown: float = Field(default=0.0, description="Maximum drawdown from peak")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTCUSDT",
                "entry_price": 45000.0,
                "amount": 0.1,
                "side": "BUY",
                "current_value": 4650.0,
                "entry_value": 4500.0,
                "pnl": 150.0,
                "pnl_percent": 3.33,
                "stop_loss": 43000.0,
                "take_profit": 48000.0,
                "entry_market_phase": "Trending Up",
                "entry_reasoning": "Strong bullish momentum with RSI confirmation",
                "strategy": "Trend Following"
            }
        }


class Portfolio(BaseModel):
    """
    Overall portfolio status.
    Aggregates all positions and account balance.
    """
    # Account balance
    balance: float = Field(..., description="Available cash balance")
    initial_balance: float = Field(..., description="Starting balance")

    # Positions
    positions: list[Position] = Field(default_factory=list)
    position_count: int = Field(default=0)

    # Total metrics
    total_equity: float = Field(..., description="Balance + position values")
    total_pnl: float = Field(default=0.0, description="Total unrealized P&L")
    total_pnl_percent: float = Field(default=0.0)

    # Performance metrics
    win_rate: Optional[float] = Field(None, ge=0, le=100)
    profit_factor: Optional[float] = Field(None)
    sharpe_ratio: Optional[float] = Field(None)
    max_drawdown: Optional[float] = Field(None)

    # Risk metrics
    portfolio_heat: float = Field(default=0.0, description="Total risk exposure %")
    leverage: float = Field(default=1.0, description="Effective leverage")

    # Goals (from AlphaThink)
    start_capital: Optional[float] = Field(None)
    goal_capital: Optional[float] = Field(None)
    progress_to_goal: Optional[float] = Field(None, ge=0, le=100)

    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "balance": 5500.0,
                "initial_balance": 10000.0,
                "positions": [],
                "position_count": 1,
                "total_equity": 10150.0,
                "total_pnl": 150.0,
                "total_pnl_percent": 1.5,
                "win_rate": 65.0,
                "profit_factor": 2.1,
                "sharpe_ratio": 1.8,
                "max_drawdown": -5.2,
                "portfolio_heat": 12.5,
                "leverage": 1.0,
                "start_capital": 10000.0,
                "goal_capital": 100000.0,
                "progress_to_goal": 1.5
            }
        }
