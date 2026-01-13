import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Line, ReferenceDot } from 'recharts';
import { Asset, TradeAction, ChartMarker } from '../types';

interface PriceChartProps {
  asset: Asset;
  markers?: ChartMarker[];
  stopLoss?: number;
  takeProfit?: number;
  lastAction?: TradeAction;
  marketPhase?: string;
}

const calculateMA = (data: any[], period: number) => {
  return data.map((val, index, arr) => {
    if (index < period - 1) return { ...val, ma: null };
    const slice = arr.slice(index - period + 1, index + 1);
    const sum = slice.reduce((acc, curr) => acc + curr.price, 0);
    return { ...val, ma: sum / period };
  });
};

const PriceChart: React.FC<PriceChartProps> = ({ asset, markers, stopLoss, takeProfit, lastAction, marketPhase }) => {
  // Use 'close' price for the Area Chart representation
  const rawData = asset.history.map(point => ({
    time: point.time,
    price: point.close, // CHANGED: Using Close price from Candle
    formattedTime: new Date(point.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  }));

  const data = calculateMA(rawData, 10); 

  // Min/Max for Y-Axis scaling
  const minPrice = Math.min(...asset.history.map(h => h.low));
  const maxPrice = Math.max(...asset.history.map(h => h.high));
  const domainPadding = (maxPrice - minPrice) * 0.1;

  const displayPhase = marketPhase || (
    asset.trend === 'BULLISH' ? 'UPTREND' :
    asset.trend === 'BEARISH' ? 'DOWNTREND' :
    'RANGING'
  );

  const getPhaseStyles = (p: string) => {
    const lower = p.toLowerCase();
    if (lower.includes('trending') || lower.includes('bull')) return 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10 shadow-[0_0_15px_rgba(16,185,129,0.1)]';
    if (lower.includes('down') || lower.includes('bear')) return 'text-red-400 border-red-500/30 bg-red-500/10 shadow-[0_0_15px_rgba(239,68,68,0.1)]';
    if (lower.includes('ranging') || lower.includes('chop')) return 'text-yellow-400 border-yellow-500/30 bg-yellow-500/10 shadow-[0_0_15px_rgba(234,179,8,0.1)]';
    return 'text-indigo-400 border-indigo-500/30 bg-indigo-500/10';
  };

  return (
    <div className="h-64 w-full bg-slate-900 border border-slate-800 rounded-lg p-4 relative overflow-hidden">
      <div className="absolute inset-0 bg-[linear-gradient(rgba(30,41,59,0.3)_1px,transparent_1px),linear-gradient(90deg,rgba(30,41,59,0.3)_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none"></div>

      <div className="absolute top-16 right-4 z-20 flex flex-col items-end pointer-events-none">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest mb-1">Algorithmic Phase</span>
          <div className={`text-xs font-bold font-mono px-3 py-1.5 rounded border backdrop-blur-sm transition-all duration-500 ${getPhaseStyles(displayPhase)}`}>
              {displayPhase.toUpperCase()}
          </div>
      </div>

      <div className="flex justify-between items-center mb-2 relative z-10">
        <div className="flex items-center gap-3">
             <div className="bg-slate-800 p-2 rounded text-indigo-400 font-bold text-xl">{asset.symbol}</div>
             <div className="flex flex-col">
                <span className="text-slate-400 text-xs uppercase tracking-wider">{asset.name}</span>
                <span className="text-[10px] text-slate-500 font-mono">VOL: {(asset.history[asset.history.length-1]?.volume || 0).toFixed(0)}</span>
             </div>
        </div>
        <div className="text-right">
             <div className={`text-2xl font-mono font-bold tracking-tight ${asset.change >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                ${asset.price.toFixed(2)}
            </div>
            <div className={`text-xs ${asset.change >= 0 ? 'text-emerald-500' : 'text-red-500'} flex justify-end items-center gap-1`}>
                {asset.change >= 0 ? '▲' : '▼'} {Math.abs(asset.change).toFixed(2)}%
            </div>
        </div>
      </div>
      
      <ResponsiveContainer width="100%" height="80%">
        <AreaChart data={data}>
          <defs>
            <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={asset.change >= 0 ? "#10b981" : "#ef4444"} stopOpacity={0.2}/>
              <stop offset="95%" stopColor={asset.change >= 0 ? "#10b981" : "#ef4444"} stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
          <XAxis 
            dataKey="time" 
            type="number"
            domain={['dataMin', 'dataMax']}
            stroke="#475569" 
            fontSize={9} 
            tickMargin={10}
            minTickGap={40}
            tickFormatter={(unixTime) => new Date(unixTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          />
          <YAxis 
            domain={[minPrice - domainPadding, maxPrice + domainPadding]} 
            stroke="#475569" 
            fontSize={9} 
            tickFormatter={(val) => val.toFixed(1)}
            width={40}
            orientation="right"
          />
          <Tooltip 
            labelFormatter={(label) => new Date(label).toLocaleTimeString()}
            contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc', fontSize: '12px' }}
            itemStyle={{ color: '#f8fafc' }}
          />
          <Area 
            type="step" 
            dataKey="price" 
            stroke={asset.change >= 0 ? "#10b981" : "#ef4444"} 
            strokeWidth={2}
            fillOpacity={1} 
            fill="url(#colorPrice)" 
            isAnimationActive={false}
          />
          <Line 
            type="monotone" 
            dataKey="ma" 
            stroke="#fbbf24" 
            strokeWidth={1} 
            dot={false} 
            isAnimationActive={false} 
            strokeDasharray="5 5"
          />
          
          {markers && markers.map((marker) => (
             <ReferenceDot 
                key={marker.id} 
                x={marker.time} 
                y={marker.price} 
                r={4} 
                fill={marker.type === 'BUY' ? '#10b981' : '#ef4444'}
                stroke="#fff"
                strokeWidth={1}
            />
          ))}

          {stopLoss && (
             <ReferenceLine y={stopLoss} stroke="#ef4444" strokeDasharray="3 3" label={{ position: 'left',  value: 'STOP', fill: '#ef4444', fontSize: 9 }} />
          )}
          {takeProfit && (
             <ReferenceLine y={takeProfit} stroke="#10b981" strokeDasharray="3 3" label={{ position: 'left',  value: 'TARGET', fill: '#10b981', fontSize: 9 }} />
          )}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PriceChart;