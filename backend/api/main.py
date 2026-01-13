"""
Main FastAPI application for the unified trading platform.
Combines AlphaThink's UI capabilities with forex system's broker integrations.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import asyncio
import json
import logging

from ..models.asset import Asset, AssetType
from ..models.trade import Trade, TradeAction
from ..models.position import Position
from ..data.aggregator import DataAggregator
from ..ai.orchestrator import AIOrchestrator
from ..trading.paper_trader import PaperTrader
from ..trading.position_manager import PositionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unified Trading Platform",
    description="AlphaThink + Forex System - Professional algorithmic trading platform",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
data_aggregator = DataAggregator()
ai_orchestrator = AIOrchestrator()
position_manager = PositionManager()
paper_trader = PaperTrader(position_manager)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.active_connections.remove(conn)

manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Unified Trading Platform...")

    # Initialize data aggregator
    await data_aggregator.initialize()

    # Start real-time data streaming
    asyncio.create_task(stream_market_data())

    logger.info("Platform started successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Unified Trading Platform...")
    await data_aggregator.shutdown()
    logger.info("Platform shut down successfully!")


async def stream_market_data():
    """Stream real-time market data to all connected clients"""
    while True:
        try:
            # Get latest market data
            market_data = await data_aggregator.get_latest_data()

            # Broadcast to all connected clients
            if market_data:
                await manager.broadcast({
                    "type": "market_update",
                    "data": market_data
                })

            await asyncio.sleep(1)  # 1 second update interval

        except Exception as e:
            logger.error(f"Error streaming market data: {e}")
            await asyncio.sleep(5)


# REST API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "platform": "Unified Trading Platform",
        "version": "1.0.0"
    }


@app.get("/api/assets")
async def get_assets():
    """Get list of all available assets"""
    try:
        assets = await data_aggregator.get_assets()
        return {"success": True, "assets": assets}
    except Exception as e:
        logger.error(f"Error getting assets: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/assets/{symbol}/history")
async def get_asset_history(symbol: str, limit: int = 200):
    """Get historical price data for an asset"""
    try:
        history = await data_aggregator.get_history(symbol, limit)
        return {"success": True, "history": history}
    except Exception as e:
        logger.error(f"Error getting history for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/portfolio")
async def get_portfolio():
    """Get current portfolio status"""
    try:
        portfolio = {
            "balance": position_manager.get_balance(),
            "positions": position_manager.get_positions(),
            "total_equity": position_manager.get_total_equity(),
            "pnl": position_manager.get_total_pnl()
        }
        return {"success": True, "portfolio": portfolio}
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/analyze")
async def analyze_market(request: dict):
    """Analyze market and get trading decision"""
    try:
        symbol = request.get("symbol")
        context = request.get("context", {})

        # Get latest asset data
        asset = await data_aggregator.get_asset(symbol)
        if not asset:
            return {"success": False, "error": f"Asset {symbol} not found"}

        # Get AI decision
        decision = await ai_orchestrator.analyze(asset, context)

        return {"success": True, "decision": decision}

    except Exception as e:
        logger.error(f"Error analyzing market: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/trade")
async def execute_trade(request: dict):
    """Execute a trade"""
    try:
        action = request.get("action")
        symbol = request.get("symbol")
        amount = request.get("amount")
        price = request.get("price")

        # Execute trade
        result = await paper_trader.execute_trade(
            action=TradeAction(action),
            symbol=symbol,
            amount=amount,
            price=price
        )

        return {"success": True, "result": result}

    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/trades")
async def get_trade_history(limit: int = 50):
    """Get trade history"""
    try:
        trades = position_manager.get_trade_history(limit)
        return {"success": True, "trades": trades}
    except Exception as e:
        logger.error(f"Error getting trade history: {e}")
        return {"success": False, "error": str(e)}


# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data streaming"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle client messages (subscriptions, commands, etc.)
            await handle_client_message(websocket, message)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def handle_client_message(websocket: WebSocket, message: dict):
    """Handle messages from WebSocket clients"""
    msg_type = message.get("type")

    if msg_type == "subscribe":
        # Handle subscription to specific assets
        symbol = message.get("symbol")
        logger.info(f"Client subscribed to {symbol}")
        # Subscription logic here

    elif msg_type == "unsubscribe":
        # Handle unsubscription
        symbol = message.get("symbol")
        logger.info(f"Client unsubscribed from {symbol}")
        # Unsubscription logic here

    elif msg_type == "ping":
        # Respond to ping
        await websocket.send_json({"type": "pong"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
