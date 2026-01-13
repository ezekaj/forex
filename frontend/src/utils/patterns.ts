// Simple heuristic pattern recognition to assist the AI context
export const detectMarketPatterns = (prices: number[]): string[] => {
  if (prices.length < 10) return ["Insufficient Data"];

  const patterns: string[] = [];
  const recent = prices.slice(-10);
  const start = recent[0];
  const end = recent[recent.length - 1];
  const max = Math.max(...recent);
  const min = Math.min(...recent);

  // 1. Trend Direction
  const isUptrend = end > start && recent.every((p, i) => i === 0 || p >= recent[i - 1] * 0.99); // Loose uptrend
  const isDowntrend = end < start && recent.every((p, i) => i === 0 || p <= recent[i - 1] * 1.01); // Loose downtrend
  
  if (isUptrend) patterns.push("Short-term Uptrend");
  else if (isDowntrend) patterns.push("Short-term Downtrend");
  else patterns.push("Consolidation/Chop");

  // 2. Volatility State
  const rangePercent = ((max - min) / min) * 100;
  if (rangePercent < 0.2) patterns.push("Low Volatility (Squeeze Potential)");
  else if (rangePercent > 1.5) patterns.push("High Volatility (Expansion)");

  // 3. Support/Resistance Proximity
  if (Math.abs(end - max) / max < 0.001) patterns.push("Testing Recent Highs");
  if (Math.abs(end - min) / min < 0.001) patterns.push("Testing Recent Lows");

  return patterns;
};