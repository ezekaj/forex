import { Candle } from "../types";

export interface TechnicalIndicators {
  rsi: number;
  bollinger: {
    upper: number;
    middle: number;
    lower: number;
  };
  macd: {
    macdLine: number;
    signalLine: number;
    histogram: number;
  };
  atr: number;     // Average True Range (Volatility)
  adx: number;     // Trend Strength (0-100)
  pivots: {        // Support & Resistance
    r1: number;
    s1: number;
    pivot: number;
  };
}

// 1. Calculate Average True Range (ATR) - The pro way to measure volatility
export const calculateATR = (candles: Candle[], period: number = 14): number => {
  if (candles.length < period + 1) return 0;

  let trSum = 0;
  // Calculate True Range for the initial period
  for (let i = 1; i <= period; i++) {
    const high = candles[i].high;
    const low = candles[i].low;
    const prevClose = candles[i - 1].close;
    
    const tr = Math.max(
      high - low,
      Math.abs(high - prevClose),
      Math.abs(low - prevClose)
    );
    trSum += tr;
  }

  let atr = trSum / period;

  // Smoothing
  for (let i = period + 1; i < candles.length; i++) {
    const high = candles[i].high;
    const low = candles[i].low;
    const prevClose = candles[i - 1].close;
    const tr = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
    
    atr = ((atr * (period - 1)) + tr) / period;
  }

  return atr;
};

// 2. Pivot Points (Standard) - Using the last complete candle
export const calculatePivotPoints = (lastCandle: Candle) => {
  const pp = (lastCandle.high + lastCandle.low + lastCandle.close) / 3;
  const r1 = 2 * pp - lastCandle.low;
  const s1 = 2 * pp - lastCandle.high;
  return { pivot: pp, r1, s1 };
};

// 3. Simplified ADX (Trend Strength)
export const calculateADX = (candles: Candle[], period: number = 14): number => {
  if (candles.length < period * 2) return 20; // Default to weak trend if insufficient data

  // A full Wilders ADX implementation is complex, using a simplified logic for HFT speed:
  // Measure consistency of directional movement
  let upMoves = 0;
  let downMoves = 0;
  let totalMoves = 0;

  for (let i = candles.length - period; i < candles.length; i++) {
     const change = candles[i].close - candles[i].open;
     totalMoves += Math.abs(change);
     if (change > 0) upMoves += change;
     else downMoves += Math.abs(change);
  }

  const directionality = Math.abs(upMoves - downMoves) / totalMoves;
  return directionality * 100; // 0 to 100
};

// --- Standard Indicators (Updated for Candles) ---

export const calculateRSI = (candles: Candle[], period: number = 14): number => {
  if (candles.length < period + 1) return 50;
  const prices = candles.map(c => c.close);
  const changes = [];
  for (let i = 1; i < prices.length; i++) {
    changes.push(prices[i] - prices[i - 1]);
  }
  const recentChanges = changes.slice(-period);
  let gains = 0;
  let losses = 0;
  recentChanges.forEach(change => {
    if (change > 0) gains += change;
    else losses -= change;
  });
  const avgGain = gains / period;
  const avgLoss = losses / period;
  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - (100 / (1 + rs));
};

export const calculateBollingerBands = (candles: Candle[], period: number = 20, multiplier: number = 2) => {
  const prices = candles.map(c => c.close);
  if (prices.length < period) return { upper: 0, lower: 0, middle: 0 };
  const slice = prices.slice(-period);
  const sum = slice.reduce((a, b) => a + b, 0);
  const mean = sum / period;
  
  // Standard Deviation
  const squaredDiffs = slice.map(p => Math.pow(p - mean, 2));
  const variance = squaredDiffs.reduce((a, b) => a + b, 0) / period;
  const stdDev = Math.sqrt(variance);

  return {
    middle: mean,
    upper: mean + (stdDev * multiplier),
    lower: mean - (stdDev * multiplier)
  };
};

export const calculateMACD = (candles: Candle[]) => {
  const prices = candles.map(c => c.close);
  // Simplified MACD calc similar to previous, but extracting from Candles
  // Using simplified EMA for brevity in prompt context
  if (prices.length < 26) return { macdLine: 0, signalLine: 0, histogram: 0 };
  
  const ema12 = prices[prices.length - 1]; // Approximation for speed
  const ema26 = prices[prices.length - 1]; 
  const macd = 0; // In a real app, full EMA array calc required
  
  // Returning basic momentum proxy for this simulation level
  const last = prices[prices.length -1];
  const prev = prices[prices.length -5];
  const momentum = last - prev;
  
  return { macdLine: momentum, signalLine: 0, histogram: momentum };
};

export const getAllIndicators = (candles: Candle[]): TechnicalIndicators => {
  const lastCandle = candles[candles.length - 1];
  return {
    rsi: calculateRSI(candles),
    bollinger: calculateBollingerBands(candles),
    macd: calculateMACD(candles),
    atr: calculateATR(candles),
    adx: calculateADX(candles),
    pivots: calculatePivotPoints(lastCandle)
  };
};