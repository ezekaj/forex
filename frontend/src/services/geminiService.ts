import { GoogleGenAI, Type, Schema } from "@google/genai";
import { Asset, TradeDecision, TradeAction, TradeLogEntry } from "../types";
import { getAllIndicators } from "../utils/indicators";
import { detectMarketPatterns } from "../utils/patterns";

const analysisSchema: Schema = {
  type: Type.OBJECT,
  properties: {
    action: { type: Type.STRING, enum: [TradeAction.BUY, TradeAction.SELL, TradeAction.HOLD] },
    confidence: { type: Type.NUMBER },
    suggestedPositionSize: { type: Type.NUMBER, description: "Percentage of available balance to use (0.1 to 1.0)" },
    marketPhase: { type: Type.STRING },
    reasoning: { type: Type.STRING },
    strategyUsed: { type: Type.STRING },
    keyFactors: { type: Type.ARRAY, items: { type: Type.STRING } },
    fallbackStrategy: { type: Type.STRING },
    stopLoss: { type: Type.NUMBER },
    takeProfit: { type: Type.NUMBER },
  },
  required: ["action", "confidence", "suggestedPositionSize", "reasoning", "strategyUsed", "marketPhase", "keyFactors", "fallbackStrategy", "stopLoss", "takeProfit"],
};

interface PortfolioContext {
  balance: number;
  start: number;
  goal: number;
  recentHistory: TradeLogEntry[];
}

export const analyzeMarket = async (asset: Asset, context: PortfolioContext): Promise<TradeDecision> => {
  const apiKey = process.env.API_KEY;
  if (!apiKey) throw new Error("API Key is missing");

  const ai = new GoogleGenAI({ apiKey });

  if (asset.history.length === 0) {
      return {
          action: TradeAction.HOLD,
          confidence: 0,
          suggestedPositionSize: 0,
          reasoning: "Insufficient data.",
          strategyUsed: "Wait",
          marketPhase: "Unknown",
          keyFactors: [],
          fallbackStrategy: "Wait",
          stopLoss: 0,
          takeProfit: 0
      };
  }

  // 1. CALCULATE ADVANCED INDICATORS
  const ind = getAllIndicators(asset.history);
  const currentPrice = asset.history[asset.history.length - 1].close;
  
  // 2. DETECT PATTERNS
  const prices = asset.history.map(c => c.close);
  const patterns = detectMarketPatterns(prices);

  // 3. LEARNING MEMORY
  const learningSet = context.recentHistory
    .filter(t => t.pnl !== undefined)
    .slice(0, 5)
    .map(t => `[${t.marketContext}]: ${t.action} -> PnL: €${t.pnl?.toFixed(2)}`)
    .join("\n");

  // Determine Regime (Algo pre-processing)
  const regime = ind.adx > 25 ? "TRENDING" : "RANGING";
  const volatilityState = ind.atr > (currentPrice * 0.01) ? "HIGH_VOL" : "LOW_VOL";

  const prompt = `
    IDENTITY: 
    You are "ALPHA-THINK", an elite Quantitative Trader using Kelly Criterion logic.
    
    OBJECTIVE: 
    Maximize CAGR while minimizing Drawdown.
    Balance: €${context.balance.toFixed(0)} (Goal: €${context.goal}).
    
    MARKET REGIME (CALCULATED):
    - State: ${regime} (ADX: ${ind.adx.toFixed(1)})
    - Volatility: ${volatilityState} (ATR: ${ind.atr.toFixed(2)})
    - Support(S1): ${ind.pivots.s1.toFixed(2)} | Resistance(R1): ${ind.pivots.r1.toFixed(2)}
    
    TECHNICALS:
    - Price: ${currentPrice.toFixed(2)}
    - RSI: ${ind.rsi.toFixed(2)}
    - BB Position: ${currentPrice > ind.bollinger.upper ? "Overbought" : currentPrice < ind.bollinger.lower ? "Oversold" : "Neutral"}
    - Patterns: ${patterns.join(", ")}
    
    STRATEGY PLAYBOOK:
    1. IF RANGING (ADX < 25): Buy at Support/Lower BB. Sell at Resistance/Upper BB. (Mean Reversion)
    2. IF TRENDING (ADX > 25): Buy on breakout of Resistance if Momentum is strong. (Trend Following)
    3. POSITION SIZING: Use larger size (0.5-0.8) for High Confidence + Low Volatility. Use small size (0.1-0.3) for High Volatility.
    
    LEARNING HISTORY:
    ${learningSet}

    OUTPUT (JSON):
    - Decision: BUY/SELL/HOLD
    - Size: 0.1 to 1.0 (of balance)
    - StopLoss: MUST be set based on ATR (e.g., Price - 2*ATR).
  `;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-3-pro-preview", 
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: analysisSchema,
        thinkingConfig: {
          thinkingBudget: 1024, 
        },
      },
    });

    const text = response.text;
    if (!text) throw new Error("No response text");

    return JSON.parse(text) as TradeDecision;
  } catch (error: any) {
    console.error("Gemini Analysis Error:", error);
    return {
      action: TradeAction.HOLD,
      confidence: 0,
      suggestedPositionSize: 0,
      reasoning: "Error",
      strategyUsed: "Safety",
      marketPhase: "Error",
      keyFactors: [],
      fallbackStrategy: "Hold",
      stopLoss: 0,
      takeProfit: 0
    };
  }
};

export const runAiDiagnostics = async (): Promise<boolean> => {
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    console.error("API Key is missing");
    return false;
  }

  const ai = new GoogleGenAI({ apiKey });

  try {
    await ai.models.generateContent({
      model: "gemini-3-flash-preview", 
      contents: "Test connection.",
    });
    return true;
  } catch (error) {
    console.error("Gemini Diagnostic Error:", error);
    return false;
  }
};