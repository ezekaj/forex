"""
RISK NUKE - High-Risk Position Sizing for Maximum Profit
Kelly Criterion aggressive mode
Position size: 20% of capital per trade
Martingale: 3-level recovery system
Daily stop: 50% of capital
"""

import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class RiskNuke:
    """
    Aggressive risk management for €10-100 capital
    Target: 50% daily returns
    Warning: HIGH RISK - Can lose everything
    """
    
    def __init__(self, initial_capital: float = 10.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Aggressive parameters
        self.base_risk = 0.20  # 20% per trade (VERY aggressive)
        self.kelly_multiplier = 2.0  # Double Kelly (ultra-aggressive)
        self.martingale_multiplier = 2.0  # Double after loss
        self.max_martingale_level = 3  # Maximum 3x scaling
        
        # Risk limits
        self.daily_loss_limit = 0.50  # 50% daily loss limit
        self.weekly_reset_threshold = 0.80  # Reset if lose 80%
        
        # Tracking
        self.martingale_level = 0
        self.daily_high = initial_capital
        self.daily_loss = 0
        self.trades_since_loss = 0
        
    def calculate_turbo_position(self, capital: float, n_edges: int, leverage: int = 500) -> float:
        """
        Calculate position size for turbo mode
        Distributes risk across multiple edges
        """
        
        # Base position per edge
        risk_per_edge = self.base_risk / max(n_edges, 1)
        
        # Adjust for martingale
        if self.martingale_level > 0:
            risk_per_edge *= (self.martingale_multiplier ** self.martingale_level)
            
        # Calculate position size
        position_size = (capital * risk_per_edge) / leverage
        
        # Minimum position
        min_position = 0.01  # Micro lot
        position_size = max(position_size, min_position)
        
        # Maximum position (safety cap)
        max_position = capital / (leverage * 10)  # Max 10% even with leverage
        position_size = min(position_size, max_position)
        
        return round(position_size, 2)
        
    def calculate_kelly_position(self, win_rate: float, avg_win: float, avg_loss: float, 
                               capital: float) -> float:
        """
        Kelly Criterion for optimal position sizing
        f* = (p*b - q) / b
        where p = win probability, q = loss probability, b = win/loss ratio
        """
        
        if avg_loss == 0 or win_rate == 0:
            return 0.01  # Minimum position
            
        # Kelly formula
        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss
        
        # Kelly fraction
        kelly = (p * b - q) / b
        
        # Apply multiplier (aggressive)
        kelly *= self.kelly_multiplier
        
        # Cap at maximum risk
        kelly = min(kelly, self.base_risk * 2)
        
        # Ensure positive
        kelly = max(kelly, 0.01)
        
        # Calculate position
        position = capital * kelly
        
        return round(position, 2)
        
    def activate_martingale(self):
        """
        Activate martingale recovery after loss
        WARNING: Very risky, can blow account
        """
        
        if self.martingale_level < self.max_martingale_level:
            self.martingale_level += 1
            return True
        else:
            # Max level reached, reset
            self.reset_martingale()
            return False
            
    def reset_martingale(self):
        """
        Reset martingale after win or max level
        """
        self.martingale_level = 0
        self.trades_since_loss = 0
        
    def update_after_trade(self, pnl: float):
        """
        Update risk parameters after trade
        """
        
        self.current_capital += pnl
        
        if pnl > 0:
            # Win - reset martingale
            self.reset_martingale()
            self.trades_since_loss += 1
            
            # Update daily high
            self.daily_high = max(self.daily_high, self.current_capital)
            
        else:
            # Loss - consider martingale
            self.daily_loss += abs(pnl)
            self.trades_since_loss = 0
            
            # Check if should activate martingale
            if self.should_use_martingale():
                self.activate_martingale()
                
    def should_use_martingale(self) -> bool:
        """
        Decide if martingale should be used
        """
        
        # Don't martingale if near daily limit
        if self.daily_loss > self.initial_capital * 0.40:
            return False
            
        # Don't martingale if capital too low
        if self.current_capital < self.initial_capital * 0.30:
            return False
            
        # Use martingale for recovery
        return True
        
    def check_risk_limits(self) -> Dict[str, bool]:
        """
        Check if any risk limits are hit
        """
        
        limits = {
            'can_trade': True,
            'daily_limit_hit': False,
            'capital_critical': False,
            'martingale_max': False
        }
        
        # Check daily loss limit
        daily_loss_pct = self.daily_loss / self.initial_capital
        if daily_loss_pct >= self.daily_loss_limit:
            limits['daily_limit_hit'] = True
            limits['can_trade'] = False
            
        # Check capital critical level
        if self.current_capital < self.initial_capital * 0.20:
            limits['capital_critical'] = True
            limits['can_trade'] = False
            
        # Check martingale max
        if self.martingale_level >= self.max_martingale_level:
            limits['martingale_max'] = True
            
        return limits
        
    def calculate_aggressive_size(self, confidence: float, volatility: float, 
                                 recent_performance: float) -> float:
        """
        Calculate ultra-aggressive position size
        """
        
        # Base size (20% of capital)
        size = self.current_capital * self.base_risk
        
        # Confidence adjustment (higher confidence = bigger size)
        if confidence > 0.70:
            size *= 1.5
        elif confidence > 0.60:
            size *= 1.2
            
        # Volatility adjustment
        if volatility < 0.0005:  # Low volatility
            size *= 1.3  # Increase size
        elif volatility > 0.002:  # High volatility
            size *= 0.7  # Decrease size
            
        # Performance adjustment
        if recent_performance > 0.60:  # Winning streak
            size *= 1.4
        elif recent_performance < 0.40:  # Losing streak
            size *= 0.6
            
        # Martingale adjustment
        if self.martingale_level > 0:
            size *= (self.martingale_multiplier ** self.martingale_level)
            
        # Convert to lots
        lot_size = size / 1000  # 1 lot = $1000
        
        # Minimum and maximum
        lot_size = max(0.01, min(lot_size, 1.0))
        
        return round(lot_size, 2)
        
    def get_risk_metrics(self) -> Dict:
        """
        Get current risk metrics
        """
        
        return {
            'current_capital': self.current_capital,
            'capital_at_risk': self.current_capital * self.base_risk,
            'martingale_level': self.martingale_level,
            'daily_loss': self.daily_loss,
            'daily_loss_pct': self.daily_loss / self.initial_capital,
            'capital_remaining_pct': self.current_capital / self.initial_capital,
            'can_trade': self.check_risk_limits()['can_trade'],
            'position_multiplier': self.martingale_multiplier ** self.martingale_level
        }
        
    def reset_daily(self):
        """
        Reset daily risk parameters
        """
        self.daily_loss = 0
        self.daily_high = self.current_capital
        self.reset_martingale()
        
    def emergency_stop(self):
        """
        Emergency stop - cease all trading
        """
        return {
            'stop_reason': 'EMERGENCY',
            'capital_lost': self.initial_capital - self.current_capital,
            'loss_percentage': (self.initial_capital - self.current_capital) / self.initial_capital,
            'recommendation': 'Stop trading and review strategy'
        }


if __name__ == "__main__":
    # Test Risk Nuke
    print("Testing Risk Nuke System...")
    print("WARNING: This is EXTREMELY HIGH RISK!")
    print("="*50)
    
    risk = RiskNuke(initial_capital=10.0)
    
    # Simulate trading session
    trades = []
    
    for i in range(20):
        # Check if can trade
        limits = risk.check_risk_limits()
        if not limits['can_trade']:
            print(f"\nRISK LIMIT HIT - Stopping trading")
            break
            
        # Calculate position
        confidence = np.random.uniform(0.5, 0.8)
        volatility = np.random.uniform(0.0001, 0.002)
        recent_perf = np.random.uniform(0.3, 0.7)
        
        position = risk.calculate_aggressive_size(confidence, volatility, recent_perf)
        
        # Simulate trade outcome
        win = np.random.random() < 0.5075  # 50.75% win rate
        if win:
            pnl = position * 20  # 2 pips profit
        else:
            pnl = -position * 10  # 1 pip loss
            
        # Update risk
        risk.update_after_trade(pnl)
        
        trades.append({
            'trade': i+1,
            'position': position,
            'pnl': pnl,
            'capital': risk.current_capital,
            'martingale': risk.martingale_level
        })
        
        if (i+1) % 5 == 0:
            metrics = risk.get_risk_metrics()
            print(f"\nAfter {i+1} trades:")
            print(f"  Capital: €{metrics['current_capital']:.2f}")
            print(f"  Daily Loss: €{metrics['daily_loss']:.2f} ({metrics['daily_loss_pct']*100:.1f}%)")
            print(f"  Martingale Level: {metrics['martingale_level']}")
            
    # Final report
    print("\n" + "="*50)
    print("RISK NUKE FINAL REPORT")
    print("="*50)
    print(f"Starting Capital: €{risk.initial_capital:.2f}")
    print(f"Final Capital: €{risk.current_capital:.2f}")
    print(f"Total P&L: €{risk.current_capital - risk.initial_capital:.2f}")
    print(f"Return: {((risk.current_capital - risk.initial_capital) / risk.initial_capital)*100:.1f}%")
    
    if risk.current_capital < risk.initial_capital * 0.5:
        print("\n⚠️ WARNING: Significant losses - Review strategy!")