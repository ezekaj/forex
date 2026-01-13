import React from 'react';
import { Position } from '../types';
import { TrendingUp, TrendingDown, PieChart, DollarSign, Activity } from 'lucide-react';

interface PortfolioProps {
  balance: number;
  positions: Position[];
}

const Portfolio: React.FC<PortfolioProps> = ({ balance, positions }) => {
  const totalEquity = balance + positions.reduce((acc, pos) => acc + pos.currentValue, 0);

  return (
    <div className="space-y-6">
      {/* Summary Stats Grid */}
      <div className="grid grid-cols-2 gap-3">
          <div className="bg-slate-800/50 p-3 rounded-xl border border-white/5">
              <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Cash Balance</div>
              <div className="text-sm font-mono text-white">€{balance.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
          </div>
          <div className="bg-slate-800/50 p-3 rounded-xl border border-white/5">
              <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Exposure</div>
              <div className="text-sm font-mono text-indigo-400">€{(totalEquity - balance).toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
          </div>
      </div>

      {/* Bar Visual */}
      <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden flex">
          <div className="h-full bg-slate-600" style={{ width: `${(balance / totalEquity) * 100}%` }} title="Cash"></div>
          <div className="h-full bg-indigo-500" style={{ width: `${((totalEquity - balance) / totalEquity) * 100}%` }} title="Invested"></div>
      </div>

      {/* Positions List */}
      <div>
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
            <Activity size={12} /> Active Positions ({positions.length})
        </h3>
        
        <div className="space-y-3">
            {positions.length === 0 ? (
                <div className="text-center py-10 border border-dashed border-slate-800 rounded-xl text-slate-600">
                    <p className="text-xs">No active trades.</p>
                    <p className="text-[10px] mt-1">AI is scanning the market...</p>
                </div>
            ) : (
                positions.map(pos => (
                    <div key={pos.symbol} className="bg-slate-800/30 p-4 rounded-xl border border-white/5 hover:border-indigo-500/30 transition-all group">
                        <div className="flex justify-between items-start mb-3">
                            <div>
                                <span className="font-bold text-white text-sm block">{pos.symbol}</span>
                                <span className="text-[10px] font-mono text-slate-500">{pos.amount.toFixed(4)} UNITS</span>
                            </div>
                            <div className="text-right">
                                <span className="block text-sm font-mono text-white">€{pos.currentValue.toFixed(2)}</span>
                                <span className="text-[10px] text-slate-500">Value</span>
                            </div>
                        </div>
                        
                        <div className={`flex items-center justify-between p-2 rounded-lg ${pos.pnl >= 0 ? 'bg-emerald-500/10' : 'bg-red-500/10'}`}>
                            <span className="text-[10px] text-slate-400 font-bold uppercase">PnL</span>
                            <div className={`flex items-center gap-1 text-xs font-bold font-mono ${pos.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                {pos.pnl >= 0 ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                                {pos.pnl >= 0 ? '+' : ''}{pos.pnl.toFixed(2)} ({pos.pnlPercent.toFixed(2)}%)
                            </div>
                        </div>
                    </div>
                ))
            )}
        </div>
      </div>
    </div>
  );
};

export default Portfolio;