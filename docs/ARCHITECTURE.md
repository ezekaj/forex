# Unified Trading Platform - Architecture

## Overview

This document describes the architecture of the Unified Trading Platform, which combines AlphaThink's real-time UI with a production-grade Python trading backend.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            React Frontend (TypeScript)                  │   │
│  │                                                         │   │
│  │  • Real-time Charts (Recharts)                         │   │
│  │  • Portfolio Dashboard                                 │   │
│  │  • AI Decision Console                                 │   │
│  │  • Trading Controls                                    │   │
│  └──────────────┬──────────────────────────────────────────┘   │
└─────────────────┼──────────────────────────────────────────────┘
                  │ HTTP REST + WebSocket
┌─────────────────▼──────────────────────────────────────────────┐
│                        API LAYER                                │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              FastAPI Application                        │   │
│  │                                                         │   │
│  │  • REST API Endpoints                                  │   │
│  │  • WebSocket Server                                    │   │
│  │  • Request Validation (Pydantic)                       │   │
│  │  • Authentication & Authorization                      │   │
│  └──────────────┬──────────────────────────────────────────┘   │
└─────────────────┼──────────────────────────────────────────────┘
                  │
┌─────────────────▼──────────────────────────────────────────────┐
│                     BUSINESS LOGIC LAYER                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            AI Decision Engine                           │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │ Gemini 3.0 Pro (Deep Reasoning)                │   │   │
│  │  │ • 1024-token thinking budget                  │   │   │
│  │  │ • Market regime classification                │   │   │
│  │  │ • Strategy selection                          │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │ Multi-Agent System                            │   │   │
│  │  │ • Chart Analysis Agent                        │   │   │
│  │  │ • News Sentiment Agent                        │   │   │
│  │  │ • Risk Management Agent                       │   │   │
│  │  │ • Trend Identification Agent                  │   │   │
│  │  │ • Economic Factors Agent                      │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │ AI Orchestrator (Consensus Engine)            │   │   │
│  │  │ • Collect agent recommendations               │   │   │
│  │  │ • Weight by confidence                        │   │   │
│  │  │ • Generate final decision                     │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            Trading Engine                               │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │ Position Manager                              │   │   │
│  │  │ • Track open positions                        │   │   │
│  │  │ • Calculate P&L                               │   │   │
│  │  │ • Monitor stop loss / take profit             │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │ Risk Manager                                  │   │   │
│  │  │ • Kelly Criterion sizing                      │   │   │
│  │  │ • Portfolio heat monitoring                   │   │   │
│  │  │ • Correlation limits                          │   │   │
│  │  │ • Max drawdown protection                     │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │ Execution Engine                              │   │   │
│  │  │ • Paper Trader (simulation)                   │   │   │
│  │  │ • Demo Trader (broker demo)                   │   │   │
│  │  │ • Live Trader (real money)                    │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            Data Aggregation Service                     │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │ Market Data Feeds                             │   │   │
│  │  │ • Binance WebSocket (crypto)                  │   │   │
│  │  │ • MT5 Feed (forex/CFDs)                       │   │   │
│  │  │ • OANDA Feed (forex)                          │   │   │
│  │  │ • Yahoo Finance (stocks)                      │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │ Data Normalizer                               │   │   │
│  │  │ • Unified OHLCV format                        │   │   │
│  │  │ • Timezone conversion                         │   │   │
│  │  │ • Symbol mapping                              │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │ Indicator Calculator                          │   │   │
│  │  │ • ATR, RSI, Bollinger Bands                   │   │   │
│  │  │ • ADX, MACD, Pivot Points                     │   │   │
│  │  │ • 90+ ta-lib indicators                       │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────┬───────────────────┬─────────────────────────┘
                  │                   │
┌─────────────────▼──────────┐  ┌─────▼──────────────────────────┐
│      DATA LAYER            │  │     CACHE LAYER                │
│                            │  │                                │
│  ┌─────────────────────┐   │  │  ┌──────────────────────────┐  │
│  │   PostgreSQL        │   │  │  │      Redis               │  │
│  │                     │   │  │  │                          │  │
│  │  • Account data     │   │  │  │  • Real-time prices      │  │
│  │  • Trade history    │   │  │  │  • Market data cache     │  │
│  │  • Positions        │   │  │  │  • Session state         │  │
│  │  • Performance      │   │  │  │  • Pub/Sub for WS        │  │
│  └─────────────────────┘   │  │  └──────────────────────────┘  │
└────────────────────────────┘  └─────────────────────────────────┘
```

## Component Details

### Frontend (TypeScript/React)

**Technology Stack:**
- React 19.2 with Hooks
- TypeScript 5.8
- Vite 6.2 (build tool)
- Recharts 3.6 (charting)
- Lucide React (icons)
- Google GenAI SDK

**Key Components:**
1. **App.tsx**: Main application container
2. **PriceChart.tsx**: Real-time candlestick charts with markers
3. **Portfolio.tsx**: Position display and P&L tracking
4. **BotConsole.tsx**: AI decision transparency
5. **MultiAssetDashboard.tsx**: Asset selector and overview

**State Management:**
- React Context for global state
- Local state with useState for component state
- WebSocket updates trigger re-renders

### Backend (Python/FastAPI)

**Technology Stack:**
- Python 3.11+
- FastAPI 0.109 (async web framework)
- SQLAlchemy 2.0 (ORM)
- Pydantic 2.5 (validation)
- asyncio (concurrency)

**Key Modules:**

1. **API Layer** (`/api`)
   - REST endpoints for CRUD operations
   - WebSocket server for real-time updates
   - Request validation and error handling

2. **AI Decision Engine** (`/ai`)
   - Gemini service integration
   - Multi-agent orchestrator
   - Consensus algorithm

3. **Trading Engine** (`/trading`)
   - Position manager
   - Risk manager
   - Order execution (paper/demo/live)

4. **Data Services** (`/data`)
   - Market data aggregator
   - Broker connectors
   - Cache management

5. **Indicators** (`/indicators`)
   - Technical analysis library
   - Pattern recognition
   - Custom indicators

6. **Models** (`/models`)
   - Pydantic schemas
   - SQLAlchemy ORM models
   - Type definitions

## Data Flow

### Market Data Flow

```
Binance/MT5/OANDA → WebSocket → Data Aggregator → Redis Cache
                                        ↓
                                 Indicator Calculator
                                        ↓
                                  WebSocket Server → Frontend
```

### Trading Decision Flow

```
Market Data → Indicators → AI Agents → Orchestrator → Decision
                                                          ↓
                                             Risk Manager (validate)
                                                          ↓
                                              Execution Engine
                                                          ↓
                                             Broker/Paper Trader
                                                          ↓
                                            Position Manager (update)
                                                          ↓
                                              WebSocket → Frontend
```

### AI Ensemble Decision Flow

```
Market Context
      ├─→ Gemini 3.0 Pro (deep reasoning) ──→ Decision A (weight: 40%)
      ├─→ Chart Analysis Agent ──→ Decision B (weight: 15%)
      ├─→ News Sentiment Agent ──→ Decision C (weight: 10%)
      ├─→ Risk Management Agent ──→ Decision D (weight: 15%)
      ├─→ Trend Agent ──→ Decision E (weight: 10%)
      └─→ Economic Agent ──→ Decision F (weight: 10%)
                                    ↓
                            Orchestrator (weighted consensus)
                                    ↓
                              Final Decision
                              (confidence score)
```

## Database Schema

### Key Tables

**accounts**
- id (PK)
- user_id
- balance
- initial_balance
- created_at
- updated_at

**positions**
- id (PK)
- account_id (FK)
- symbol
- entry_price
- amount
- side (BUY/SELL)
- stop_loss
- take_profit
- opened_at
- status

**trades**
- id (PK)
- account_id (FK)
- symbol
- action (BUY/SELL)
- price
- amount
- fee
- pnl
- reasoning (AI)
- confidence (AI)
- executed_at

**performance_metrics**
- id (PK)
- account_id (FK)
- date
- total_pnl
- win_rate
- sharpe_ratio
- max_drawdown

## API Endpoints

### REST API

**Assets**
- GET `/api/assets` - List all available assets
- GET `/api/assets/{symbol}` - Get asset details
- GET `/api/assets/{symbol}/history` - Get historical data

**Portfolio**
- GET `/api/portfolio` - Get current portfolio
- GET `/api/portfolio/metrics` - Get performance metrics

**Trading**
- POST `/api/analyze` - Analyze market (get AI decision)
- POST `/api/trade` - Execute trade
- GET `/api/trades` - Get trade history

**Configuration**
- GET `/api/config` - Get system configuration
- PUT `/api/config` - Update configuration

### WebSocket API

**Connection:** `ws://localhost:8000/ws`

**Client → Server Messages:**
```json
{
  "type": "subscribe",
  "symbol": "BTCUSDT"
}
```

**Server → Client Messages:**
```json
{
  "type": "market_update",
  "data": {
    "symbol": "BTCUSDT",
    "price": 45000.0,
    "change": 2.5,
    "timestamp": 1700000000000
  }
}
```

## Security Considerations

1. **API Security**
   - Rate limiting
   - API key authentication
   - Request validation
   - CORS configuration

2. **Data Security**
   - Encrypted API keys
   - No credentials in logs
   - Secure WebSocket connections
   - Database connection pooling

3. **Trading Security**
   - Position size limits
   - Max drawdown protection
   - Emergency stop mechanism
   - Audit trail

## Performance Optimization

1. **Caching Strategy**
   - Redis for real-time data
   - In-memory indicator cache
   - HTTP response caching

2. **Database Optimization**
   - Indexed queries
   - Connection pooling
   - Async operations
   - Batch inserts

3. **WebSocket Optimization**
   - Message compression
   - Connection pooling
   - Heartbeat mechanism
   - Automatic reconnection

## Scalability

### Horizontal Scaling

1. **Backend**
   - Stateless API servers
   - Load balancer distribution
   - Redis for shared state

2. **Database**
   - Read replicas
   - Connection pooling
   - Query optimization

3. **WebSocket**
   - Sticky sessions
   - Redis pub/sub for coordination

### Vertical Scaling

- Increase server resources
- Optimize algorithms
- Use faster indicators

## Monitoring & Observability

1. **Metrics**
   - Request latency
   - WebSocket connections
   - Trade execution time
   - AI decision time

2. **Logging**
   - Structured JSON logs
   - Log levels (DEBUG/INFO/WARN/ERROR)
   - Centralized log aggregation

3. **Alerts**
   - High error rate
   - Slow response time
   - Trading anomalies
   - System health checks

## Deployment Architecture

### Development

```
Developer Machine
└─ Docker Compose
   ├─ PostgreSQL
   ├─ Redis
   ├─ Backend
   └─ Frontend
```

### Production (Cloud)

```
AWS/GCP/Azure
├─ Load Balancer
├─ Backend (ECS/Cloud Run)
│  └─ Auto-scaling
├─ Frontend (S3 + CloudFront / Vercel)
├─ Database (RDS / Cloud SQL)
├─ Cache (ElastiCache / Memorystore)
└─ Monitoring (CloudWatch / Stackdriver)
```

## Technology Choices

### Why FastAPI?
- Async support (critical for real-time trading)
- Automatic API documentation
- Type safety with Pydantic
- High performance (comparable to Node.js)

### Why React?
- Component reusability
- Rich ecosystem
- Real-time updates (via hooks)
- Large community

### Why PostgreSQL?
- ACID compliance (critical for financial data)
- JSON support
- Strong typing
- Excellent Python support

### Why Redis?
- Sub-millisecond latency
- Pub/Sub for WebSocket coordination
- TTL for temporary data
- Atomic operations

## Future Architecture Enhancements

1. **Microservices**
   - Split into trading, analytics, data services
   - Event-driven architecture
   - Message queue (RabbitMQ/Kafka)

2. **Machine Learning Pipeline**
   - Model training service
   - Feature store
   - A/B testing framework

3. **Advanced Analytics**
   - Real-time dashboards
   - Historical analysis
   - Strategy backtesting

4. **Global Distribution**
   - Multi-region deployment
   - CDN for frontend
   - Database replication

---

**Document Version:** 1.0
**Last Updated:** January 2026
