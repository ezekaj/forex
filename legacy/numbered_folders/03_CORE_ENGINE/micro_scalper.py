"""
MICRO SCALPER - 1-Minute Chart Ultra-Fast Trading
Entry: Pattern break + ML confirmation
Exit: 2-pip profit or 1-pip stop loss
Frequency: 100+ trades per day
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MicroScalper:
    """
    High-frequency micro-scalping for maximum trade volume
    Renaissance principle: Many small edges compound
    """
    
    def __init__(self):
        # Scalping parameters
        self.target_pips = 2  # Take profit
        self.stop_pips = 1  # Stop loss (aggressive)
        self.pip_value = 0.0001  # For EUR/USD
        
        # Trade management
        self.open_trades = {}
        self.trade_history = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Performance metrics
        self.total_pips = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
        self.current_streak = 0
        
        # Speed optimization
        self.entry_speed_ms = 50  # Target entry speed
        self.exit_speed_ms = 30  # Target exit speed
        
    def execute_micro_trade(self, direction: str, confidence: float, 
                           position_size: float, leverage: int = 500) -> Dict:
        """
        Execute ultra-fast micro trade
        """
        self.total_trades += 1
        
        # Simulate entry price (would be real quote in production)
        entry_price = self.get_simulated_price()
        
        # Calculate exit levels
        if direction.upper() == 'BUY':
            take_profit = entry_price + (self.target_pips * self.pip_value)
            stop_loss = entry_price - (self.stop_pips * self.pip_value)
        else:  # SELL
            take_profit = entry_price - (self.target_pips * self.pip_value)
            stop_loss = entry_price + (self.stop_pips * self.pip_value)
            
        # Simulate trade execution
        exit_price, exit_reason = self.simulate_trade_execution(
            entry_price, take_profit, stop_loss, direction, confidence
        )
        
        # Calculate P&L
        if direction.upper() == 'BUY':
            pips = (exit_price - entry_price) / self.pip_value
        else:
            pips = (entry_price - exit_price) / self.pip_value
            
        # Calculate monetary P&L
        # With leverage: position_size * leverage * pips * pip_value
        pnl = position_size * leverage * pips * 10  # $10 per pip for 0.01 lot
        
        # Update statistics
        if pips > 0:
            self.winning_trades += 1
            self.current_streak = max(1, self.current_streak + 1)
            self.max_consecutive_wins = max(self.max_consecutive_wins, self.current_streak)
        else:
            self.losing_trades += 1
            self.current_streak = min(-1, self.current_streak - 1)
            self.max_consecutive_losses = max(self.max_consecutive_losses, abs(self.current_streak))
            
        self.total_pips += pips
        
        # Create trade result
        trade_result = {
            'trade_id': self.total_trades,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'pips': pips,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'confidence': confidence,
            'position_size': position_size,
            'leverage': leverage,
            'timestamp': datetime.now()
        }
        
        self.trade_history.append(trade_result)
        
        return trade_result
        
    def simulate_trade_execution(self, entry: float, tp: float, sl: float, 
                                direction: str, confidence: float) -> Tuple[float, str]:
        """
        Simulate realistic trade execution with market dynamics
        """
        
        # Base win probability (Renaissance: 50.75%)
        base_win_rate = 0.5075
        
        # Adjust based on confidence
        win_probability = base_win_rate + (confidence - 0.5) * 0.2
        win_probability = np.clip(win_probability, 0.3, 0.7)
        
        # Add streak influence (momentum/mean reversion)
        if self.current_streak > 3:
            # Winning streak - slightly lower probability (mean reversion)
            win_probability *= 0.95
        elif self.current_streak < -3:
            # Losing streak - slightly higher probability (mean reversion)
            win_probability *= 1.05
            
        # Determine outcome
        if np.random.random() < win_probability:
            # Win - hit take profit
            exit_price = tp
            exit_reason = 'TAKE_PROFIT'
            
            # Sometimes get extra pips (runner)
            if np.random.random() < 0.1:  # 10% chance
                extra_pips = np.random.uniform(0.5, 2.0) * self.pip_value
                if direction.upper() == 'BUY':
                    exit_price += extra_pips
                else:
                    exit_price -= extra_pips
                exit_reason = 'RUNNER'
        else:
            # Loss - hit stop loss
            exit_price = sl
            exit_reason = 'STOP_LOSS'
            
            # Sometimes lose extra (slippage)
            if np.random.random() < 0.05:  # 5% chance
                slippage = np.random.uniform(0.1, 0.5) * self.pip_value
                if direction.upper() == 'BUY':
                    exit_price -= slippage
                else:
                    exit_price += slippage
                exit_reason = 'STOP_LOSS_SLIPPAGE'
                
        return exit_price, exit_reason
        
    def get_simulated_price(self, base: float = 1.0850) -> float:
        """
        Get simulated price for testing
        In production, this would fetch real quote
        """
        # Add realistic micro-movements
        movement = np.random.normal(0, 0.0001)
        return base + movement
        
    def analyze_entry_quality(self, patterns: Dict, ml_signal: Dict) -> float:
        """
        Analyze entry quality for position sizing
        """
        quality_score = 0.5  # Base score
        
        # Pattern alignment
        if patterns:
            pattern_count = len(patterns)
            quality_score += min(pattern_count * 0.05, 0.2)
            
        # ML confidence
        if ml_signal and 'confidence' in ml_signal:
            quality_score += (ml_signal['confidence'] - 0.5) * 0.3
            
        return np.clip(quality_score, 0.0, 1.0)
        
    def should_enter_trade(self, spread: float, volatility: float) -> bool:
        """
        Quick decision on whether to enter trade
        """
        
        # Check spread (critical for scalping)
        if spread > 2.0:  # More than 2 pips
            return False
            
        # Check volatility
        if volatility < 0.0001:  # Too quiet
            return False
        elif volatility > 0.005:  # Too volatile
            return False
            
        # Check recent performance
        if self.current_streak < -5:  # 5 losses in a row
            return False  # Take a break
            
        return True
        
    def get_optimal_entry_time(self) -> bool:
        """
        Check if current time is optimal for scalping
        """
        current_hour = datetime.now().hour
        
        # Best hours for scalping (London/NY sessions)
        optimal_hours = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        
        return current_hour in optimal_hours
        
    def calculate_position_size(self, capital: float, quality: float) -> float:
        """
        Calculate position size for micro-scalping
        """
        
        # Base size (micro lot)
        base_size = 0.01
        
        # Adjust based on capital
        if capital > 100:
            base_size = 0.02
        if capital > 500:
            base_size = 0.05
        if capital > 1000:
            base_size = 0.10
            
        # Adjust based on quality
        size = base_size * (0.5 + quality * 0.5)
        
        # Adjust based on streak
        if self.current_streak > 3:
            size *= 1.2  # Increase on winning streak
        elif self.current_streak < -3:
            size *= 0.8  # Decrease on losing streak
            
        return round(size, 2)
        
    def get_performance_stats(self) -> Dict:
        """
        Get detailed performance statistics
        """
        
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        avg_win = self.total_pips / self.winning_trades if self.winning_trades > 0 else 0
        avg_loss = abs(self.total_pips) / self.losing_trades if self.losing_trades > 0 else 0
        
        # Profit factor
        total_wins = self.winning_trades * self.target_pips
        total_losses = self.losing_trades * self.stop_pips
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Expectancy
        expectancy = (win_rate * self.target_pips) - ((1 - win_rate) * self.stop_pips)
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_pips': self.total_pips,
            'avg_win_pips': avg_win,
            'avg_loss_pips': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'current_streak': self.current_streak
        }
        
    def reset_daily_stats(self):
        """
        Reset statistics for new trading day
        """
        self.trade_history = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pips = 0
        self.current_streak = 0


if __name__ == "__main__":
    # Test micro scalper
    print("Testing Micro Scalper...")
    
    scalper = MicroScalper()
    
    # Simulate 100 trades
    total_pnl = 0
    
    for i in range(100):
        # Random signal
        direction = np.random.choice(['BUY', 'SELL'])
        confidence = np.random.uniform(0.5, 0.8)
        position_size = 0.01  # Micro lot
        
        # Execute trade
        result = scalper.execute_micro_trade(direction, confidence, position_size)
        total_pnl += result['pnl']
        
        if (i + 1) % 20 == 0:
            stats = scalper.get_performance_stats()
            print(f"\nAfter {i+1} trades:")
            print(f"  Win Rate: {stats['win_rate']*100:.1f}%")
            print(f"  Total P&L: ${total_pnl:.2f}")
            print(f"  Expectancy: {stats['expectancy']:.2f} pips")
            
    # Final stats
    final_stats = scalper.get_performance_stats()
    print("\n" + "="*50)
    print("FINAL MICRO SCALPING RESULTS")
    print("="*50)
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")