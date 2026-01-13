export interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Asset {
  symbol: string;
  name: string;
  price: number;
  change: number;
  history: Candle[]; // Upgraded to Candles
  volatility: number; // ATR
  type: 'CRYPTO' | 'STOCK';
  trend: 'BULLISH' | 'BEARISH' | 'SIDEWAYS';
}

export enum TradeAction {
  BUY = 'BUY',
  SELL = 'SELL',
  HOLD = 'HOLD'
}

export interface TradeDecision {
  action: TradeAction;
  confidence: number;
  reasoning: string;
  strategyUsed: string;
  suggestedPositionSize: number; // New: 0.0 to 1.0 (Percentage of balance)
  stopLoss?: number;
  takeProfit?: number;
  keyFactors: string[];
  marketPhase: string;
  fallbackStrategy?: string;
}

export interface Position {
  symbol: string;
  entryPrice: number;
  amount: number;
  currentValue: number;
  pnl: number;
  pnlPercent: number;
  stopLoss?: number;
  takeProfit?: number;
  entryMarketPhase?: string;
}

export interface TradeLogEntry {
  id: string;
  timestamp: number;
  symbol: string;
  action: TradeAction;
  price: number;
  reasoning: string;
  pnl?: number;
  marketContext?: string;
}

export interface ChartMarker {
  id: string;
  time: number;
  price: number;
  type: 'BUY' | 'SELL';
}

export interface BotStatus {
  isAnalyzing: boolean;
  lastUpdate: number | null;
  mode: 'MANUAL' | 'AUTO';
}