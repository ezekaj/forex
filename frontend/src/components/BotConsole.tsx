import React, { useEffect, useState } from 'react';
import { TradeDecision } from '../types';
import { Brain, Lock, Terminal, Zap } from 'lucide-react';

interface BotConsoleProps {
  decision: TradeDecision | null;
  isAnalyzing: boolean;
  symbol: string;
}

const THOUGHTS = [
  "Fetching order book...",
  "Calculating RSI/MACD divergence...",
  "Running Monte Carlo simulation...",
  "Checking support liquidity...",
  "Validating risk/reward ratio...",
  "Optimizing entry point..."
];

const BotConsole: React.FC<BotConsoleProps> = ({ decision, isAnalyzing, symbol }) => {
  const [currentThought, setCurrentThought] = useState("");

  useEffect(() => {
    if (isAnalyzing) {
      let i = 0;
      const interval = setInterval(() => {
        setCurrentThought(THOUGHTS[i % THOUGHTS.length]);
        i++;
      }, 500);
      return () => clearInterval(interval);
    } else {
        setCurrentThought("");
    }
  }, [isAnalyzing]);

  if (isAnalyzing) {
      return (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-6">
              <div className="relative">
                  <div className="w-24 h-24 rounded-full border-2 border-dashed border-indigo-500/30 animate-[spin_4s_linear_infinite] absolute inset-0"></div>
                  <div className="w-24 h-24 rounded-full border-2 border-indigo-500/10 absolute inset-0"></div>
                  <div className="w-24 h-24 rounded-full flex items-center justify-center relative z-10">
                      <Zap className="text-indigo-400 w-10 h-10 animate-pulse" />
                  </div>
              </div>
              <div className="space-y-2">
                  <h3 className="text-lg font-bold text-white tracking-widest uppercase">Processing</h3>
                  <div className="h-6 overflow-hidden">
                     <p className="text-xs text-indigo-400 font-mono animate-pulse">{currentThought}</p>
                  </div>
              </div>
          </div>
      );
  }

  if (!decision) {
      return (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 opacity-40">
              <Terminal className="text-slate-400 w-12 h-12 mb-4" />
              <p className="text-sm font-mono text-slate-400">WAITING FOR SIGNAL...</p>
          </div>
      );
  }

  const isBuy = decision.action === 'BUY';
  const isSell = decision.action === 'SELL';
  const themeColor = isBuy ? 'emerald' : isSell ? 'red' : 'slate';
  const textColor = isBuy ? 'text-emerald-400' : isSell ? 'text-red-400' : 'text-slate-400';
  const borderColor = isBuy ? 'border-emerald-500/30' : isSell ? 'border-red-500/30' : 'border-slate-500/30';
  const glow = isBuy ? 'shadow-[0_0_20px_rgba(16,185,129,0.1)]' : isSell ? 'shadow-[0_0_20px_rgba(239,68,68,0.1)]' : '';

  return (
    <div className="p-4 space-y-4 h-full overflow-y-auto custom-scrollbar">
      {/* Top Card: Decision */}
      <div className={`p-6 rounded-xl border ${borderColor} bg-slate-900/50 text-center ${glow} backdrop-blur-sm relative overflow-hidden group`}>
          <div className={`absolute top-0 left-0 w-1 h-full ${isBuy ? 'bg-emerald-500' : isSell ? 'bg-red-500' : 'bg-slate-500'}`}></div>
          <div className="text-[10px] uppercase tracking-[0.2em] text-slate-500 mb-2">Signal Generated</div>
          <div className={`text-5xl font-black ${textColor} tracking-tighter mb-2`}>{decision.action}</div>
          
          {/* Confidence Meter */}
          <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mt-4">
               <div 
                className={`h-full ${isBuy ? 'bg-emerald-500' : isSell ? 'bg-red-500' : 'bg-slate-500'}`} 
                style={{ width: `${decision.confidence}%` }}
               ></div>
          </div>
          <div className="flex justify-between mt-1 text-[10px] font-mono text-slate-400">
              <span>CONFIDENCE</span>
              <span>{decision.confidence}%</span>
          </div>
      </div>

      {/* Analysis Details */}
      <div className="space-y-3">
          <div className="bg-slate-900/30 p-4 rounded-lg border border-white/5">
              <h4 className="text-[10px] font-bold text-indigo-400 uppercase mb-2 flex items-center gap-2 tracking-wider">
                  <Brain size={12} /> Thesis
              </h4>
              <p className="text-xs text-slate-300 leading-relaxed font-mono">
                  {decision.reasoning}
              </p>
          </div>

          <div className="grid grid-cols-2 gap-3">
              <div className="bg-slate-900/30 p-3 rounded-lg border border-white/5">
                  <h4 className="text-[10px] font-bold text-slate-500 uppercase mb-1">Market Phase</h4>
                  <div className="text-xs font-bold text-white uppercase">{decision.marketPhase}</div>
              </div>
              <div className="bg-slate-900/30 p-3 rounded-lg border border-white/5">
                  <h4 className="text-[10px] font-bold text-slate-500 uppercase mb-1">Stop Loss</h4>
                  <div className="text-xs font-mono text-red-400">${decision.stopLoss?.toFixed(2) || 'N/A'}</div>
              </div>
          </div>

          <div className="bg-slate-900/30 p-3 rounded-lg border border-white/5">
               <h4 className="text-[10px] font-bold text-slate-500 uppercase mb-2">Signals Detected</h4>
               <div className="flex flex-wrap gap-2">
                   {decision.keyFactors?.map((f, i) => (
                       <span key={i} className="text-[10px] px-2 py-1 bg-indigo-500/10 border border-indigo-500/20 rounded text-indigo-200">
                           {f}
                       </span>
                   ))}
               </div>
          </div>

           {decision.fallbackStrategy && (
               <div className="bg-slate-900/30 p-3 rounded-lg border border-red-900/20">
                    <h4 className="text-[10px] font-bold text-red-400 uppercase mb-1">Fallback Plan</h4>
                    <p className="text-[10px] text-slate-400">{decision.fallbackStrategy}</p>
               </div>
           )}
      </div>
    </div>
  );
};

export default BotConsole;