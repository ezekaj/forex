import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Zap, Play, Pause, TrendingUp, History, Wallet, LayoutDashboard, BrainCircuit, ArrowRight, ShieldCheck, Activity, BarChart3, Terminal } from 'lucide-react';
import { getInitialAssets, fetchHistoricalData, SYMBOLS } from './utils/marketSimulator';
import { analyzeMarket, runAiDiagnostics } from './services/geminiService';
import { Asset, Position, TradeAction, TradeDecision, TradeLogEntry, ChartMarker } from './types';
import { calculateATR } from './utils/indicators';
import PriceChart from './components/PriceChart';
import BotConsole from './components/BotConsole';
import Portfolio from './components/Portfolio';

const SetupScreen = ({ onStart, onTest }: { onStart: (start: number, goal: number) => void, onTest: () => void }) => {
  const [start, setStart] = useState("10000");
  const [goal, setGoal] = useState("100000");
  const [testing, setTesting] = useState(false);

  const handleTest = async () => {
    setTesting(true);
    const result = await runAiDiagnostics();
    setTesting(false);
    alert(result ? "System Diagnostic PASSED" : "System Diagnostic FAILED");
  };

  return (
    <div className="h-screen w-full bg-[#020617] relative flex flex-col items-center justify-center p-4 overflow-hidden">
      <div className="absolute inset-0 bg-[linear-gradient(rgba(30,41,59,0.2)_1px,transparent_1px),linear-gradient(90deg,rgba(30,41,59,0.2)_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none"></div>
      <div className="max-w-md w-full glass-panel p-8 rounded-2xl shadow-2xl relative z-10 animate-in fade-in zoom-in duration-500 neon-glow">
        <div className="mb-8 text-center">
          <div className="w-20 h-20 bg-indigo-500/10 rounded-full flex items-center justify-center mx-auto mb-6 border border-indigo-500/30 shadow-[0_0_30px_rgba(99,102,241,0.3)]">
            <Zap className="text-indigo-400 w-10 h-10" />
          </div>
          <h1 className="text-3xl font-bold text-white mb-2 tracking-tight">AlphaThink <span className="text-indigo-400">Pro</span></h1>
          <p className="text-slate-400 text-sm font-mono">Algorithmic Quant Terminal</p>
        </div>
        <div className="space-y-6">
          <div className="space-y-2">
            <label className="text-xs font-bold uppercase text-slate-500 tracking-wider">Capital (€)</label>
            <input type="number" value={start} onChange={(e) => setStart(e.target.value)} className="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-4 text-white font-mono outline-none" placeholder="10000" />
          </div>
          <div className="space-y-2">
            <label className="text-xs font-bold uppercase text-slate-500 tracking-wider">Target (€)</label>
            <input type="number" value={goal} onChange={(e) => setGoal(e.target.value)} className="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-4 text-emerald-400 font-mono outline-none" placeholder="100000" />
          </div>
          <div className="flex gap-3 pt-2">
              <button onClick={() => onStart(Number(start), Number(goal))} className="flex-1 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-bold py-4 rounded-xl shadow-lg transition-all flex items-center justify-center gap-2 group border border-white/10">Initialize <ArrowRight className="w-4 h-4 group-hover:translate-x-1" /></button>
              <button onClick={handleTest} disabled={testing} className="w-14 bg-slate-800/50 hover:bg-slate-700 text-slate-400 hover:text-white rounded-xl flex items-center justify-center border border-slate-700">{testing ? <span className="animate-spin">⟳</span> : <ShieldCheck size={20} />}</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default function App() {
  const [config, setConfig] = useState<{ start: number; goal: number; isSet: boolean }>({ start: 10000, goal: 100000, isSet: false });
  const [assets, setAssets] = useState<Asset[]>(getInitialAssets());
  const [selectedSymbol, setSelectedSymbol] = useState<string>('BTCUSDT');
  const [isAutoMode, setIsAutoMode] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activeTab, setActiveTab] = useState<'brain' | 'portfolio' | 'history'>('brain');
  const [balance, setBalance] = useState(0);
  const [positions, setPositions] = useState<Position[]>([]);
  const [tradeLog, setTradeLog] = useState<TradeLogEntry[]>([]);
  const [markers, setMarkers] = useState<ChartMarker[]>([]);
  const [lastDecision, setLastDecision] = useState<TradeDecision | null>(null);

  const botInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const selectedAsset = assets.find(a => a.symbol === selectedSymbol) || assets[0];
  const totalEquity = balance + positions.reduce((acc, pos) => acc + pos.currentValue, 0);
  const progressToGoal = config.isSet ? Math.min(100, Math.max(0, ((totalEquity - config.start) / (config.goal - config.start)) * 100)) : 0;

  useEffect(() => {
    const loadInitialData = async () => {
        const initialAssets = getInitialAssets();
        const updated = await Promise.all(initialAssets.map(async (asset: Asset) => {
            const history = await fetchHistoricalData(asset.symbol);
            if (history.length === 0) return asset;
            const lastPrice = history[history.length - 1].close;
            const startPrice = history[0].open;
            const change = ((lastPrice - startPrice) / startPrice) * 100;
            return { ...asset, price: lastPrice, history, change };
        }));
        setAssets(updated);
    };
    loadInitialData();
  }, []);

  // --- WebSocket & Simulation ---
  useEffect(() => {
    const streams = Object.keys(SYMBOLS).map(s => `${s.toLowerCase()}@kline_1m`).join('/');
    const wsUrl = `wss://stream.binance.com:9443/stream?streams=${streams}`;
    
    // Fallback: Generate Synthetic Candles
    const fallbackInterval = setInterval(() => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            setAssets(prev => prev.map(a => {
                const lastCandle = a.history[a.history.length - 1];
                const volatility = lastCandle.close * 0.001; 
                const bias = (Math.random() - 0.5) * volatility;
                
                // Create a realistic tick
                const newPrice = lastCandle.close + bias;
                const newCandle = {
                    time: Date.now(),
                    open: lastCandle.close,
                    high: Math.max(lastCandle.close, newPrice),
                    low: Math.min(lastCandle.close, newPrice),
                    close: newPrice,
                    volume: Math.random() * 100
                };

                // Simply append or update last candle logic (Simplified for simulation: just push new every tick is bad, so we replace last for "live" feel)
                // In a real app we'd aggregate ticks. Here we simulate "candle closing" every 3 seconds for speed.
                const newHistory = [...a.history, newCandle];
                if (newHistory.length > 200) newHistory.shift();

                return { ...a, price: newPrice, history: newHistory };
            }));
        }
    }, 1000);

    try {
        wsRef.current = new WebSocket(wsUrl);
        wsRef.current.onmessage = (event) => {
            const message = JSON.parse(event.data);
            const kline = message.data.k; 
            if (!kline) return;

            setAssets(prev => prev.map(asset => {
                if (asset.symbol === kline.s) {
                    const newCandle = {
                        time: kline.t,
                        open: parseFloat(kline.o),
                        high: parseFloat(kline.h),
                        low: parseFloat(kline.l),
                        close: parseFloat(kline.c),
                        volume: parseFloat(kline.v)
                    };
                    
                    const newHistory = [...asset.history];
                    // Update last candle if time matches, else push
                    if (newHistory.length > 0 && newHistory[newHistory.length - 1].time === newCandle.time) {
                        newHistory[newHistory.length - 1] = newCandle;
                    } else {
                        newHistory.push(newCandle);
                        if (newHistory.length > 200) newHistory.shift();
                    }

                    const change = newHistory.length > 0 ? ((newCandle.close - newHistory[0].open) / newHistory[0].open) * 100 : 0;

                    return { ...asset, price: newCandle.close, history: newHistory, change };
                }
                return asset;
            }));

            // Real-time PnL Update
            setPositions(currentPositions => currentPositions.map(pos => {
                if (pos.symbol !== kline.s) return pos;
                const price = parseFloat(kline.c);
                const val = pos.amount * price;
                return {
                    ...pos,
                    currentValue: val,
                    pnl: val - (pos.amount * pos.entryPrice),
                    pnlPercent: ((val - (pos.amount * pos.entryPrice)) / (pos.amount * pos.entryPrice)) * 100
                };
            }));
        };
    } catch (e) { console.warn("WS Failed"); }

    return () => { if (wsRef.current) wsRef.current.close(); clearInterval(fallbackInterval); };
  }, []);

  const executeTrade = useCallback((decision: TradeDecision, asset: Asset) => {
    if (decision.action === TradeAction.HOLD) return;

    if (decision.action === TradeAction.BUY) {
      if (balance < 10) return;
      
      // ALGO: Dynamic Position Sizing (Kelly Criterion Proxy)
      // AI suggests a size (e.g., 0.5). We cap max exposure per trade to avoid ruin.
      const safeSize = Math.min(decision.suggestedPositionSize, 0.4); // Max 40% of balance per trade
      const tradeAmountUSD = Math.max(50, balance * safeSize); 

      const quantity = tradeAmountUSD / asset.price;

      setBalance(prev => prev - tradeAmountUSD);
      setPositions(prev => {
        const existing = prev.find(p => p.symbol === asset.symbol);
        if (existing) {
          const totalCost = (existing.amount * existing.entryPrice) + tradeAmountUSD;
          const totalAmount = existing.amount + quantity;
          return prev.map(p => p.symbol === asset.symbol ? {
            ...p,
            amount: totalAmount,
            entryPrice: totalCost / totalAmount,
            currentValue: totalAmount * asset.price,
            pnl: (totalAmount * asset.price) - totalCost,
            pnlPercent: 0, 
            stopLoss: decision.stopLoss, // Update SL
            takeProfit: decision.takeProfit,
            entryMarketPhase: decision.marketPhase
          } : p);
        } else {
          setMarkers(m => [...m, { id: Date.now().toString(), time: Date.now(), price: asset.price, type: 'BUY' }]);
          return [...prev, {
            symbol: asset.symbol,
            entryPrice: asset.price,
            amount: quantity,
            currentValue: tradeAmountUSD,
            pnl: 0,
            pnlPercent: 0,
            stopLoss: decision.stopLoss,
            takeProfit: decision.takeProfit,
            entryMarketPhase: decision.marketPhase
          }];
        }
      });
      setActiveTab('portfolio');
    } else if (decision.action === TradeAction.SELL) {
      const pos = positions.find(p => p.symbol === asset.symbol);
      if (!pos) return;

      const saleValue = pos.amount * asset.price;
      const realizedPnL = saleValue - (pos.amount * pos.entryPrice);

      setBalance(prev => prev + saleValue);
      setPositions(prev => prev.filter(p => p.symbol !== asset.symbol));
      setMarkers(m => [...m, { id: Date.now().toString(), time: Date.now(), price: asset.price, type: 'SELL' }]);
      
      const context = `Entry(${pos.entryMarketPhase})->Exit(${decision.marketPhase}) | Strat: ${decision.strategyUsed}`;
      setTradeLog(prev => [{
        id: Date.now().toString(), timestamp: Date.now(), symbol: asset.symbol, action: TradeAction.SELL, price: asset.price,
        reasoning: decision.reasoning, pnl: realizedPnL, marketContext: context
      }, ...prev]);
      setActiveTab('history');
    }
  }, [balance, positions, selectedSymbol]);

  // SL/TP Monitor
  useEffect(() => {
    positions.forEach(pos => {
      const currentAsset = assets.find(a => a.symbol === pos.symbol);
      if (!currentAsset) return;

      if (pos.stopLoss && currentAsset.price <= pos.stopLoss) {
        executeTrade({ action: TradeAction.SELL, confidence: 100, suggestedPositionSize: 0, reasoning: "Stop Loss Hit", strategyUsed: "Risk", marketPhase: "Exiting", keyFactors: [], fallbackStrategy: "", stopLoss: 0, takeProfit: 0 }, currentAsset);
      } else if (pos.takeProfit && currentAsset.price >= pos.takeProfit) {
        executeTrade({ action: TradeAction.SELL, confidence: 100, suggestedPositionSize: 0, reasoning: "Take Profit Hit", strategyUsed: "Risk", marketPhase: "Exiting", keyFactors: [], fallbackStrategy: "", stopLoss: 0, takeProfit: 0 }, currentAsset);
      }
    });
  }, [assets, positions, executeTrade]);

  const triggerAnalysis = useCallback(async () => {
    if (isAnalyzing) return;
    setIsAnalyzing(true);
    if (!isAutoMode) setActiveTab('brain');
    
    const currentAsset = assets.find(a => a.symbol === selectedSymbol);
    if (!currentAsset) { setIsAnalyzing(false); return; }

    try {
      const totalEq = balance + positions.reduce((acc, pos) => acc + pos.currentValue, 0);
      const decision = await analyzeMarket(JSON.parse(JSON.stringify(currentAsset)), { balance: totalEq, start: config.start, goal: config.goal, recentHistory: tradeLog });
      
      setLastDecision(decision);
      if (decision.confidence >= 50) executeTrade(decision, currentAsset);
    } catch (e) { console.error(e); } finally { setIsAnalyzing(false); }
  }, [assets, selectedSymbol, isAnalyzing, executeTrade, balance, positions, config, tradeLog, isAutoMode]);

  useEffect(() => {
    if (isAutoMode) {
      botInterval.current = setInterval(triggerAnalysis, 4000);
    } else {
      if (botInterval.current) clearInterval(botInterval.current);
    }
    return () => { if (botInterval.current) clearInterval(botInterval.current); };
  }, [isAutoMode, triggerAnalysis]);

  const handleSetup = (start: number, goal: number) => { setConfig({ start, goal, isSet: true }); setBalance(start); setIsAutoMode(true); };

  if (!config.isSet) return <SetupScreen onStart={handleSetup} onTest={() => {}} />;

  return (
    <div className="h-screen bg-[#020617] text-slate-200 font-sans overflow-hidden flex flex-col">
      <header className="bg-slate-900/50 backdrop-blur-md border-b border-white/5 h-16 flex items-center justify-between px-6 shrink-0 z-20">
        <div className="flex items-center gap-4">
            <div className="bg-indigo-600/20 border border-indigo-500/50 p-2 rounded-lg"><Zap className="text-indigo-400 w-5 h-5" /></div>
            <div>
                <h1 className="text-lg font-bold text-white tracking-tight">AlphaThink <span className="text-indigo-400 font-mono">Quant</span></h1>
                <div className="flex items-center gap-2 mt-1"><span className="relative flex h-2 w-2"><span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span><span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span></span><span className="text-[10px] text-slate-400 font-mono tracking-wider">REGIME: {lastDecision?.marketPhase || 'ANALYZING'}</span></div>
            </div>
        </div>
        <div className="hidden md:flex flex-col w-1/3 max-w-md gap-1">
            <div className="flex justify-between text-[10px] font-bold text-slate-400 uppercase tracking-wider"><span className="font-mono">Start €{config.start.toLocaleString()}</span><span className="text-white font-mono">Target €{config.goal.toLocaleString()}</span></div>
            <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden relative"><div className="absolute top-0 left-0 h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-emerald-400 shadow-[0_0_10px_rgba(99,102,241,0.5)] transition-all duration-1000 ease-out" style={{ width: `${progressToGoal}%` }}></div></div>
        </div>
        <div className="flex items-center gap-6">
             <div className="text-right"><div className="text-[10px] text-slate-500 uppercase tracking-wider font-bold">Equity</div><div className="text-xl font-mono font-bold text-white tracking-tight">€{totalEquity.toLocaleString(undefined, { minimumFractionDigits: 0 })}</div></div>
             <button onClick={() => setIsAutoMode(!isAutoMode)} className={`flex items-center gap-2 px-4 py-2 rounded-lg font-bold text-xs uppercase tracking-wider transition-all border ${isAutoMode ? 'bg-red-500/10 text-red-400 border-red-500/50' : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/50'}`}>{isAutoMode ? <Pause size={14} /> : <Play size={14} />}{isAutoMode ? 'Stop' : 'Run'}</button>
          </div>
      </header>

      <div className="h-12 bg-[#020617] border-b border-white/5 flex items-center gap-1 overflow-x-auto px-2 shrink-0 scrollbar-hide">
        {assets.map(asset => (
            <button key={asset.symbol} onClick={() => setSelectedSymbol(asset.symbol)} className={`flex-shrink-0 px-3 py-1.5 rounded border transition-all flex items-center gap-3 min-w-[140px] ${selectedSymbol === asset.symbol ? 'bg-slate-800/80 border-indigo-500/50 text-white' : 'bg-transparent border-transparent hover:bg-slate-900 text-slate-400'}`}>
                <span className="font-bold text-xs">{asset.symbol}</span>
                <span className={`text-xs font-mono ${asset.change >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>{asset.change >= 0 ? '+' : ''}{asset.change.toFixed(2)}%</span>
            </button>
        ))}
      </div>

      <div className="flex-1 overflow-hidden grid grid-cols-12 gap-0">
        <div className="col-span-12 lg:col-span-8 flex flex-col border-r border-white/5 bg-[#0b1121] relative">
            <div className="flex-1 p-4">
                 <PriceChart asset={selectedAsset} markers={markers} stopLoss={lastDecision?.stopLoss} takeProfit={lastDecision?.takeProfit} marketPhase={lastDecision?.marketPhase} />
            </div>
            <div className="h-48 border-t border-white/5 p-4 grid grid-cols-2 gap-4 bg-[#020617]">
                 <div className="glass-panel rounded-xl p-4 flex flex-col">
                    <div className="flex items-center gap-2 mb-2 text-indigo-400"><Terminal size={14} /><span className="text-xs font-bold uppercase">Activity Log</span></div>
                    <div className="flex-1 overflow-y-auto font-mono text-[10px] text-slate-400 space-y-1 custom-scrollbar">
                        {tradeLog.slice(0, 5).map(log => (<div key={log.id} className="flex gap-2"><span className="text-slate-600">[{new Date(log.timestamp).toLocaleTimeString()}]</span><span className={log.action === 'BUY' ? 'text-emerald-500' : 'text-red-500'}>{log.action}</span><span>{log.symbol} {log.pnl ? `(PnL: ${log.pnl.toFixed(2)})` : ''}</span></div>))}
                    </div>
                 </div>
                 <div className="glass-panel rounded-xl p-4">
                     <div className="flex items-center gap-2 mb-2 text-indigo-400"><Activity size={14} /><span className="text-xs font-bold uppercase">Algorithmic State</span></div>
                     <div className="space-y-2">
                         <div className="flex justify-between text-xs"><span className="text-slate-500">Vol (ATR)</span><span className="text-white font-mono">Dynamic</span></div>
                         <div className="flex justify-between text-xs"><span className="text-slate-500">Trend (ADX)</span><span className="text-emerald-400 font-mono">Auto-Detect</span></div>
                         <div className="flex justify-between text-xs"><span className="text-slate-500">Status</span><span className="text-indigo-400 font-mono animate-pulse">{isAnalyzing ? "THINKING..." : "SCANNING"}</span></div>
                    </div>
                 </div>
            </div>
        </div>
        <div className="col-span-12 lg:col-span-4 bg-[#020617] flex flex-col border-l border-white/5">
            <div className="flex p-2 bg-[#020617] gap-1">
                <button onClick={() => setActiveTab('brain')} className={`flex-1 py-2 rounded-lg text-xs font-bold uppercase tracking-wider flex items-center justify-center gap-2 transition-all ${activeTab === 'brain' ? 'bg-indigo-600 text-white shadow-lg' : 'text-slate-500 hover:bg-slate-900'}`}><BrainCircuit size={14} /> AI Brain</button>
                <button onClick={() => setActiveTab('portfolio')} className={`flex-1 py-2 rounded-lg text-xs font-bold uppercase tracking-wider flex items-center justify-center gap-2 transition-all ${activeTab === 'portfolio' ? 'bg-indigo-600 text-white shadow-lg' : 'text-slate-500 hover:bg-slate-900'}`}><Wallet size={14} /> Assets</button>
            </div>
            <div className="flex-1 overflow-hidden relative">
                {activeTab === 'brain' && (
                    <div className="h-full flex flex-col p-4">
                        <div className="flex-1 glass-panel rounded-xl overflow-hidden border border-white/5 relative">
                            <div className="absolute inset-0 bg-gradient-to-b from-transparent via-indigo-500/5 to-transparent h-[20%] w-full animate-[scan_2s_ease-in-out_infinite] pointer-events-none"></div>
                            <BotConsole decision={lastDecision} isAnalyzing={isAnalyzing} symbol={selectedSymbol} />
                        </div>
                        {!isAutoMode && (
                             <button onClick={triggerAnalysis} disabled={isAnalyzing} className="mt-4 w-full py-3 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded-xl shadow-lg disabled:opacity-50 transition-all flex justify-center items-center gap-2 border border-white/10">{isAnalyzing ? <span className="animate-pulse">Reasoning...</span> : <> <BrainCircuit size={18} /> Manual Analysis </>}</button>
                        )}
                    </div>
                )}
                {activeTab === 'portfolio' && (
                    <div className="h-full overflow-y-auto p-4 custom-scrollbar"><Portfolio balance={balance} positions={positions} /></div>
                )}
            </div>
        </div>
      </div>
    </div>
  );
}