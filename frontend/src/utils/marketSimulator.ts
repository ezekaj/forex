import { Asset, Candle } from "../types";

export const SYMBOLS = {
  'BTCUSDT': 'Bitcoin',
  'ETHUSDT': 'Ethereum',
  'SOLUSDT': 'Solana',
  'BNBUSDT': 'BNB',
  'DOGEUSDT': 'Dogecoin'
};

const INITIAL_ASSETS: Asset[] = Object.entries(SYMBOLS).map(([symbol, name]) => ({
  symbol,
  name,
  price: 0,
  change: 0,
  history: [],
  volatility: 0,
  type: 'CRYPTO',
  trend: 'SIDEWAYS'
}));

export const getInitialAssets = () => JSON.parse(JSON.stringify(INITIAL_ASSETS));

// Synthetic Data Generator (Generates Candles)
const generateSyntheticHistory = (basePrice: number): Candle[] => {
  const history: Candle[] = [];
  let currentPrice = basePrice;
  const now = Date.now();
  
  for (let i = 200; i > 0; i--) {
    const volatility = currentPrice * 0.002; // 0.2% volatility per minute
    const trendBias = (Math.random() - 0.5) * volatility; 
    
    const open = currentPrice;
    const close = currentPrice + trendBias + ((Math.random() - 0.5) * volatility);
    const high = Math.max(open, close) + (Math.random() * volatility * 0.5);
    const low = Math.min(open, close) - (Math.random() * volatility * 0.5);
    const volume = Math.random() * 1000 + 500;

    history.push({
      time: now - (i * 60000),
      open,
      high,
      low,
      close,
      volume
    });
    
    currentPrice = close;
  }
  return history;
};

export const fetchHistoricalData = async (symbol: string): Promise<Candle[]> => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000); 

    const response = await fetch(`https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=1m&limit=200`, {
        signal: controller.signal
    });
    clearTimeout(timeoutId);

    if (!response.ok) throw new Error('Network response was not ok');
    
    const data = await response.json();
    return data.map((d: any) => ({
      time: d[0],
      open: parseFloat(d[1]),
      high: parseFloat(d[2]),
      low: parseFloat(d[3]),
      close: parseFloat(d[4]),
      volume: parseFloat(d[5])
    }));
  } catch (e) {
    console.warn(`Failed to fetch history for ${symbol} (using synthetic data)`);
    const basePrices: Record<string, number> = {
        'BTCUSDT': 65000,
        'ETHUSDT': 3400,
        'SOLUSDT': 145,
        'BNBUSDT': 590,
        'DOGEUSDT': 0.12
    };
    return generateSyntheticHistory(basePrices[symbol] || 100);
  }
};