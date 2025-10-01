"""
PROFIT OPTIMIZER - Dynamic Strategy Optimization for Maximum Returns
Adaptive risk adjustment, Pattern performance tracking, Strategy switching
Target: Maximize daily returns while managing risk
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from collections import deque
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ProfitOptimizer:
    """
    Continuously optimize trading parameters for maximum profit
    Based on Renaissance Technologies' adaptive approach
    """
    
    def __init__(self):
        # Strategy performance tracking
        self.strategy_performance = {
            'scalping': {'trades': 0, 'wins': 0, 'pnl': 0.0, 'score': 0.5},
            'momentum': {'trades': 0, 'wins': 0, 'pnl': 0.0, 'score': 0.5},
            'mean_reversion': {'trades': 0, 'wins': 0, 'pnl': 0.0, 'score': 0.5},
            'pattern': {'trades': 0, 'wins': 0, 'pnl': 0.0, 'score': 0.5},
            'arbitrage': {'trades': 0, 'wins': 0, 'pnl': 0.0, 'score': 0.5}
        }
        
        # Pattern performance tracking
        self.pattern_performance = {}
        
        # Market regime detection
        self.current_regime = 'neutral'
        self.regime_history = deque(maxlen=100)
        
        # Adaptive parameters
        self.risk_multiplier = 1.0
        self.confidence_threshold = 0.60
        self.position_scaler = 1.0
        
        # Performance window
        self.recent_trades = deque(maxlen=50)
        self.optimization_interval = 20  # Optimize every 20 trades
        
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        Detect current market regime for strategy selection
        """
        
        if len(df) < 50:
            return 'neutral'
            
        closes = df['close'].values[-50:]
        returns = pd.Series(closes).pct_change().dropna()
        
        # Calculate regime indicators
        volatility = returns.std()
        trend = np.polyfit(range(len(closes)), closes, 1)[0]
        mean_return = returns.mean()
        
        # Determine regime
        regime = 'neutral'
        
        if abs(trend) > 0.0001:
            if trend > 0:
                regime = 'trending_up'
            else:
                regime = 'trending_down'
        elif volatility > 0.002:
            regime = 'volatile'
        elif volatility < 0.0005:
            regime = 'ranging'
            
        self.current_regime = regime
        self.regime_history.append(regime)
        
        return regime
        
    def select_optimal_strategy(self, regime: str, edges: List[Dict]) -> str:
        """
        Select best strategy based on regime and performance
        """
        
        # Regime-specific strategy preferences
        regime_strategies = {
            'trending_up': ['momentum', 'scalping'],
            'trending_down': ['momentum', 'scalping'],
            'volatile': ['scalping', 'arbitrage'],
            'ranging': ['mean_reversion', 'pattern'],
            'neutral': ['pattern', 'scalping']
        }
        
        preferred = regime_strategies.get(regime, ['scalping'])
        
        # Select based on recent performance
        best_strategy = preferred[0]
        best_score = 0
        
        for strategy in preferred:
            if strategy in self.strategy_performance:
                score = self.calculate_strategy_score(strategy)
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
                    
        return best_strategy
        
    def calculate_strategy_score(self, strategy: str) -> float:
        """
        Calculate strategy score based on recent performance
        """
        
        perf = self.strategy_performance[strategy]
        
        if perf['trades'] == 0:
            return 0.5  # Neutral score for untested
            
        # Win rate component
        win_rate = perf['wins'] / perf['trades']
        
        # Profitability component
        avg_pnl = perf['pnl'] / perf['trades']
        
        # Recency bias (recent performance matters more)
        recency_factor = 1.0
        recent_strategy_trades = [t for t in self.recent_trades if t.get('strategy') == strategy]
        if len(recent_strategy_trades) > 5:
            recent_wins = sum(1 for t in recent_strategy_trades[-5:] if t['pnl'] > 0)
            recency_factor = 0.8 + (recent_wins / 5) * 0.4
            
        # Combined score
        score = (win_rate * 0.4 + min(avg_pnl / 10, 1.0) * 0.4 + recency_factor * 0.2)
        
        return score
        
    def optimize_parameters(self):
        """
        Optimize trading parameters based on recent performance
        """
        
        if len(self.recent_trades) < self.optimization_interval:
            return
            
        # Calculate recent metrics
        recent_wins = sum(1 for t in self.recent_trades if t['pnl'] > 0)
        recent_win_rate = recent_wins / len(self.recent_trades)
        recent_pnl = sum(t['pnl'] for t in self.recent_trades)
        
        # Adjust risk multiplier
        if recent_win_rate > 0.60 and recent_pnl > 0:
            # Increase risk when winning
            self.risk_multiplier = min(self.risk_multiplier * 1.1, 2.0)
        elif recent_win_rate < 0.45 or recent_pnl < 0:
            # Decrease risk when losing
            self.risk_multiplier = max(self.risk_multiplier * 0.9, 0.5)
            
        # Adjust confidence threshold
        if recent_win_rate > 0.55:
            # Lower threshold when performing well
            self.confidence_threshold = max(self.confidence_threshold - 0.01, 0.55)
        else:
            # Raise threshold when struggling
            self.confidence_threshold = min(self.confidence_threshold + 0.01, 0.70)
            
        # Adjust position scaling
        volatility = np.std([t['pnl'] for t in self.recent_trades])
        if volatility > 20:  # High volatility
            self.position_scaler = 0.8
        elif volatility < 5:  # Low volatility
            self.position_scaler = 1.2
        else:
            self.position_scaler = 1.0
            
    def update_performance(self, strategy: str, pattern: Optional[str], pnl: float):
        """
        Update strategy and pattern performance
        """
        
        # Update strategy performance
        if strategy in self.strategy_performance:
            self.strategy_performance[strategy]['trades'] += 1
            if pnl > 0:
                self.strategy_performance[strategy]['wins'] += 1
            self.strategy_performance[strategy]['pnl'] += pnl
            
        # Update pattern performance
        if pattern:
            if pattern not in self.pattern_performance:
                self.pattern_performance[pattern] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
                
            self.pattern_performance[pattern]['trades'] += 1
            if pnl > 0:
                self.pattern_performance[pattern]['wins'] += 1
            self.pattern_performance[pattern]['pnl'] += pnl
            
        # Add to recent trades
        self.recent_trades.append({
            'strategy': strategy,
            'pattern': pattern,
            'pnl': pnl,
            'timestamp': datetime.now()
        })
        
        # Optimize if needed
        if len(self.recent_trades) % self.optimization_interval == 0:
            self.optimize_parameters()
            
    def get_top_patterns(self, n: int = 5) -> List[str]:
        """
        Get top performing patterns
        """
        
        pattern_scores = {}
        
        for pattern, perf in self.pattern_performance.items():
            if perf['trades'] > 0:
                win_rate = perf['wins'] / perf['trades']
                avg_pnl = perf['pnl'] / perf['trades']
                score = win_rate * 0.6 + min(avg_pnl / 10, 1.0) * 0.4
                pattern_scores[pattern] = score
                
        # Sort by score
        sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [p[0] for p in sorted_patterns[:n]]
        
    def should_trade(self, confidence: float, expected_pnl: float) -> bool:
        """
        Decide if trade should be taken based on optimization
        """
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return False
            
        # Check expected value
        if expected_pnl < 0:
            return False
            
        # Check recent performance
        if len(self.recent_trades) > 10:
            recent_losses = sum(1 for t in self.recent_trades[-10:] if t['pnl'] < 0)
            if recent_losses > 7:  # 70% loss rate recently
                return False  # Take a break
                
        return True
        
    def calculate_optimal_position(self, base_position: float, confidence: float, 
                                  strategy: str) -> float:
        """
        Calculate optimal position size
        """
        
        # Start with base position
        position = base_position
        
        # Apply risk multiplier
        position *= self.risk_multiplier
        
        # Apply position scaler
        position *= self.position_scaler
        
        # Adjust for confidence
        confidence_factor = 0.5 + (confidence - 0.5)
        position *= confidence_factor
        
        # Adjust for strategy performance
        strategy_score = self.calculate_strategy_score(strategy)
        position *= (0.5 + strategy_score)
        
        # Safety caps
        position = max(0.01, min(position, base_position * 3))
        
        return round(position, 2)
        
    def get_optimization_report(self) -> Dict:
        """
        Get current optimization status
        """
        
        # Calculate overall metrics
        total_trades = sum(s['trades'] for s in self.strategy_performance.values())
        total_wins = sum(s['wins'] for s in self.strategy_performance.values())
        total_pnl = sum(s['pnl'] for s in self.strategy_performance.values())
        
        overall_win_rate = total_wins / total_trades if total_trades > 0 else 0
        
        # Best strategy
        best_strategy = max(self.strategy_performance.items(), 
                          key=lambda x: self.calculate_strategy_score(x[0]))
        
        # Best patterns
        top_patterns = self.get_top_patterns(3)
        
        return {
            'current_regime': self.current_regime,
            'risk_multiplier': self.risk_multiplier,
            'confidence_threshold': self.confidence_threshold,
            'position_scaler': self.position_scaler,
            'total_trades': total_trades,
            'overall_win_rate': overall_win_rate,
            'total_pnl': total_pnl,
            'best_strategy': best_strategy[0],
            'best_strategy_score': self.calculate_strategy_score(best_strategy[0]),
            'top_patterns': top_patterns,
            'strategy_performance': self.strategy_performance
        }
        
    def save_optimization_state(self, filepath: str = 'optimization_state.json'):
        """
        Save optimization state for persistence
        """
        
        state = {
            'strategy_performance': self.strategy_performance,
            'pattern_performance': self.pattern_performance,
            'risk_multiplier': self.risk_multiplier,
            'confidence_threshold': self.confidence_threshold,
            'position_scaler': self.position_scaler,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
        except:
            pass
            
    def load_optimization_state(self, filepath: str = 'optimization_state.json'):
        """
        Load previous optimization state
        """
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            self.strategy_performance = state.get('strategy_performance', self.strategy_performance)
            self.pattern_performance = state.get('pattern_performance', {})
            self.risk_multiplier = state.get('risk_multiplier', 1.0)
            self.confidence_threshold = state.get('confidence_threshold', 0.60)
            self.position_scaler = state.get('position_scaler', 1.0)
            
            return True
        except:
            return False


if __name__ == "__main__":
    # Test profit optimizer
    print("Testing Profit Optimizer...")
    
    optimizer = ProfitOptimizer()
    
    # Simulate trading session
    np.random.seed(42)
    
    for i in range(100):
        # Random strategy and outcome
        strategy = np.random.choice(['scalping', 'momentum', 'mean_reversion', 'pattern'])
        pattern = np.random.choice(['DOJI', 'HAMMER', 'ENGULFING', None])
        
        # Simulate PnL
        if np.random.random() < 0.5075:  # 50.75% win rate
            pnl = np.random.uniform(5, 20)
        else:
            pnl = np.random.uniform(-15, -5)
            
        # Update performance
        optimizer.update_performance(strategy, pattern, pnl)
        
        # Print progress
        if (i + 1) % 20 == 0:
            report = optimizer.get_optimization_report()
            print(f"\nAfter {i+1} trades:")
            print(f"  Best Strategy: {report['best_strategy']} (score: {report['best_strategy_score']:.2f})")
            print(f"  Risk Multiplier: {report['risk_multiplier']:.2f}")
            print(f"  Win Rate: {report['overall_win_rate']*100:.1f}%")
            print(f"  Total P&L: ${report['total_pnl']:.2f}")
            
    # Final report
    final_report = optimizer.get_optimization_report()
    print("\n" + "="*50)
    print("OPTIMIZATION FINAL REPORT")
    print("="*50)
    for key, value in final_report.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        elif isinstance(value, dict):
            continue  # Skip nested dicts
        else:
            print(f"{key}: {value}")
            
    # Save state
    optimizer.save_optimization_state()
    print("\nOptimization state saved.")