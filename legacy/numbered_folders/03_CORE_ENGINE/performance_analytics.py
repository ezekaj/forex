#!/usr/bin/env python
"""
PERFORMANCE ANALYTICS - Comprehensive Trading Performance Tracker
Measures and optimizes trading performance with professional metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceAnalytics:
    """
    Professional-grade performance tracking and analysis
    Calculates Sharpe ratio, maximum drawdown, and other key metrics
    """
    
    def __init__(self, initial_capital: float = 1000.0):
        """
        Initialize Performance Analytics
        
        Args:
            initial_capital: Starting capital for calculations
        """
        self.initial_capital = initial_capital
        self.trades = []
        self.equity_curve = [initial_capital]
        self.timestamps = [datetime.now()]
        self.performance_metrics = {}
        self.benchmark_returns = None
        
    def record_trade(self, trade: Dict):
        """
        Record a completed trade
        
        Args:
            trade: Dictionary with trade details
                - entry_time: datetime
                - exit_time: datetime
                - pair: str
                - direction: str ('BUY' or 'SELL')
                - entry_price: float
                - exit_price: float
                - position_size: float
                - profit: float
                - commission: float (optional)
        """
        # Add calculated fields
        trade['duration'] = (trade['exit_time'] - trade['entry_time']).total_seconds() / 60
        trade['return_pct'] = (trade['profit'] / trade['position_size']) * 100 if trade['position_size'] > 0 else 0
        trade['win'] = trade['profit'] > 0
        
        # Store trade
        self.trades.append(trade)
        
        # Update equity curve
        new_equity = self.equity_curve[-1] + trade['profit']
        self.equity_curve.append(new_equity)
        self.timestamps.append(trade['exit_time'])
        
    def calculate_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Returns:
            Dictionary with all performance metrics
        """
        if not self.trades:
            return {"error": "No trades to analyze"}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.trades)
        equity_series = pd.Series(self.equity_curve)
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = df[df['win'] == True]
        losing_trades = df[df['win'] == False]
        
        # Win/Loss metrics
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        # Profit metrics
        gross_profit = winning_trades['profit'].sum() if not winning_trades.empty else 0
        gross_loss = abs(losing_trades['profit'].sum()) if not losing_trades.empty else 0
        net_profit = gross_profit - gross_loss
        
        # Average metrics
        avg_win = winning_trades['profit'].mean() if not winning_trades.empty else 0
        avg_loss = abs(losing_trades['profit'].mean()) if not losing_trades.empty else 0
        avg_trade = df['profit'].mean()
        
        # Risk metrics
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # Return metrics
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        returns_series = equity_series.pct_change().dropna()
        
        # Sharpe Ratio
        sharpe_ratio = self._calculate_sharpe_ratio(returns_series)
        
        # Sortino Ratio
        sortino_ratio = self._calculate_sortino_ratio(returns_series)
        
        # Maximum Drawdown
        max_dd, max_dd_duration = self._calculate_max_drawdown(equity_series)
        
        # Calmar Ratio (Annual Return / Max Drawdown)
        if len(self.timestamps) > 1:
            days_traded = (self.timestamps[-1] - self.timestamps[0]).days
            annual_return = total_return * (365 / days_traded) if days_traded > 0 else 0
            calmar_ratio = annual_return / abs(max_dd) if max_dd != 0 else 0
        else:
            annual_return = 0
            calmar_ratio = 0
        
        # Recovery Factor (Net Profit / Max Drawdown)
        recovery_factor = net_profit / abs(max_dd * self.initial_capital) if max_dd != 0 else 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Kelly Criterion
        kelly_percentage = self._calculate_kelly_criterion(win_rate, risk_reward_ratio)
        
        # Trade duration metrics
        avg_win_duration = winning_trades['duration'].mean() if not winning_trades.empty else 0
        avg_loss_duration = losing_trades['duration'].mean() if not losing_trades.empty else 0
        
        # Consecutive metrics
        max_consecutive_wins = self._calculate_max_consecutive(df['win'], True)
        max_consecutive_losses = self._calculate_max_consecutive(df['win'], False)
        
        # Risk-adjusted metrics
        information_ratio = self._calculate_information_ratio(returns_series)
        omega_ratio = self._calculate_omega_ratio(returns_series)
        
        # Store all metrics
        self.performance_metrics = {
            # Basic metrics
            'total_trades': total_trades,
            'win_rate': win_rate,
            'win_count': win_count,
            'loss_count': loss_count,
            
            # Profit metrics
            'net_profit': net_profit,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            
            # Average metrics
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'expectancy': expectancy,
            
            # Return metrics
            'total_return': total_return,
            'annual_return': annual_return,
            
            # Risk metrics
            'max_drawdown': max_dd,
            'max_dd_duration_days': max_dd_duration,
            'risk_reward_ratio': risk_reward_ratio,
            
            # Risk-adjusted returns
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'recovery_factor': recovery_factor,
            'information_ratio': information_ratio,
            'omega_ratio': omega_ratio,
            
            # Position sizing
            'kelly_percentage': kelly_percentage,
            
            # Trade duration
            'avg_win_duration_min': avg_win_duration,
            'avg_loss_duration_min': avg_loss_duration,
            
            # Consecutive metrics
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            
            # Current status
            'current_equity': self.equity_curve[-1],
            'equity_peak': max(self.equity_curve),
            'current_drawdown': self._get_current_drawdown()
        }
        
        return self.performance_metrics
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe Ratio
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0
            
        # Adjust risk-free rate to match return frequency
        periods_per_year = 252  # Assuming daily returns
        rf_per_period = risk_free_rate / periods_per_year
        
        excess_returns = returns - rf_per_period
        
        if excess_returns.std() == 0:
            return 0.0
            
        sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
        
        return sharpe
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino Ratio (Sharpe but only penalizes downside volatility)
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
            
        periods_per_year = 252
        rf_per_period = risk_free_rate / periods_per_year
        
        excess_returns = returns - rf_per_period
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')  # No downside risk
            
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_std == 0:
            return 0.0
            
        sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_std
        
        return sortino
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> Tuple[float, int]:
        """
        Calculate maximum drawdown and duration
        
        Args:
            equity_series: Series of equity values
            
        Returns:
            Tuple of (max_drawdown_percentage, duration_in_days)
        """
        if len(equity_series) < 2:
            return 0.0, 0
            
        # Calculate running maximum
        running_max = equity_series.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_series - running_max) / running_max
        
        # Find maximum drawdown
        max_dd = drawdown.min()
        
        # Calculate duration
        if max_dd == 0:
            duration = 0
        else:
            # Find when max drawdown started and ended
            max_dd_idx = drawdown.idxmin()
            
            # Find start (last peak before max dd)
            start_idx = equity_series[:max_dd_idx][equity_series[:max_dd_idx] == running_max[:max_dd_idx]].index[-1]
            
            # Find end (recovery to previous peak)
            recovery_mask = equity_series[max_dd_idx:] >= running_max[max_dd_idx]
            if recovery_mask.any():
                end_idx = recovery_mask.idxmax()
                duration = end_idx - start_idx
            else:
                # Still in drawdown
                duration = len(equity_series) - start_idx
        
        return max_dd, duration
    
    def _get_current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        if not self.equity_curve:
            return 0.0
            
        peak = max(self.equity_curve)
        current = self.equity_curve[-1]
        
        if peak == 0:
            return 0.0
            
        return (current - peak) / peak
    
    def _calculate_kelly_criterion(self, win_rate: float, win_loss_ratio: float) -> float:
        """
        Calculate Kelly Criterion for optimal position sizing
        
        Args:
            win_rate: Probability of winning
            win_loss_ratio: Average win / Average loss
            
        Returns:
            Kelly percentage (fraction of capital to risk)
        """
        if win_loss_ratio <= 0:
            return 0.0
            
        p = win_rate
        q = 1 - win_rate
        b = win_loss_ratio
        
        kelly = (p * b - q) / b
        
        # Apply safety factor (quarter Kelly)
        safe_kelly = kelly * 0.25
        
        # Ensure reasonable bounds
        return max(0.01, min(safe_kelly, 0.25))
    
    def _calculate_max_consecutive(self, series: pd.Series, value: bool) -> int:
        """Calculate maximum consecutive occurrences of a value"""
        if len(series) == 0:
            return 0
            
        max_consecutive = 0
        current_consecutive = 0
        
        for item in series:
            if item == value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
    
    def _calculate_information_ratio(self, returns: pd.Series) -> float:
        """Calculate Information Ratio (if benchmark available)"""
        if self.benchmark_returns is None or len(returns) == 0:
            return 0.0
            
        active_returns = returns - self.benchmark_returns
        
        if active_returns.std() == 0:
            return 0.0
            
        return active_returns.mean() / active_returns.std() * np.sqrt(252)
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega Ratio"""
        if len(returns) == 0:
            return 0.0
            
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if losses.sum() == 0:
            return float('inf')
            
        return gains.sum() / losses.sum()
    
    def generate_report(self) -> str:
        """
        Generate comprehensive performance report
        
        Returns:
            Formatted performance report
        """
        if not self.performance_metrics:
            self.calculate_metrics()
            
        m = self.performance_metrics
        
        report = f"""
╔══════════════════════════════════════════════════════╗
║         PERFORMANCE ANALYTICS REPORT                  ║
╚══════════════════════════════════════════════════════╝

📊 TRADING SUMMARY
──────────────────
Total Trades:       {m['total_trades']}
Win Rate:           {m['win_rate']:.1%}
Profit Factor:      {m['profit_factor']:.2f}
Expectancy:         ${m['expectancy']:.2f}

💰 PROFIT ANALYSIS
──────────────────
Net Profit:         ${m['net_profit']:.2f}
Total Return:       {m['total_return']:.1%}
Annual Return:      {m['annual_return']:.1%}
Current Equity:     ${m['current_equity']:.2f}

📈 RISK METRICS
───────────────
Max Drawdown:       {m['max_drawdown']:.1%}
DD Duration:        {m['max_dd_duration_days']} days
Current DD:         {m['current_drawdown']:.1%}
Risk:Reward:        1:{m['risk_reward_ratio']:.1f}

🎯 RISK-ADJUSTED RETURNS
────────────────────────
Sharpe Ratio:       {m['sharpe_ratio']:.2f}
Sortino Ratio:      {m['sortino_ratio']:.2f}
Calmar Ratio:       {m['calmar_ratio']:.2f}
Recovery Factor:    {m['recovery_factor']:.2f}
Omega Ratio:        {m['omega_ratio']:.2f}

📊 TRADE STATISTICS
───────────────────
Avg Win:            ${m['avg_win']:.2f}
Avg Loss:           ${m['avg_loss']:.2f}
Avg Trade:          ${m['avg_trade']:.2f}
Max Consec Wins:    {m['max_consecutive_wins']}
Max Consec Losses:  {m['max_consecutive_losses']}

⏱️ TIMING ANALYSIS
──────────────────
Avg Win Duration:   {m['avg_win_duration_min']:.0f} min
Avg Loss Duration:  {m['avg_loss_duration_min']:.0f} min

📐 POSITION SIZING
──────────────────
Kelly Criterion:    {m['kelly_percentage']:.1%} of capital

🎯 PERFORMANCE RATING
─────────────────────"""
        
        # Add performance rating
        if m['sharpe_ratio'] > 2:
            rating = "⭐⭐⭐⭐⭐ EXCELLENT"
        elif m['sharpe_ratio'] > 1.5:
            rating = "⭐⭐⭐⭐ VERY GOOD"
        elif m['sharpe_ratio'] > 1:
            rating = "⭐⭐⭐ GOOD"
        elif m['sharpe_ratio'] > 0.5:
            rating = "⭐⭐ ACCEPTABLE"
        else:
            rating = "⭐ NEEDS IMPROVEMENT"
            
        report += f"\n{rating}"
        
        # Add recommendations
        report += "\n\n💡 RECOMMENDATIONS\n───────────────────"
        
        if m['win_rate'] < 0.40:
            report += "\n• Improve entry criteria - win rate too low"
        if m['risk_reward_ratio'] < 1.5:
            report += "\n• Increase profit targets or reduce stop losses"
        if abs(m['max_drawdown']) > 0.20:
            report += "\n• Reduce position sizes - drawdown too high"
        if m['sharpe_ratio'] < 1:
            report += "\n• Focus on consistency - reduce volatility"
        if m['avg_loss_duration_min'] > m['avg_win_duration_min'] * 2:
            report += "\n• Cut losses quicker - holding losers too long"
            
        return report
    
    def plot_equity_curve(self):
        """Plot equity curve with drawdown"""
        if len(self.equity_curve) < 2:
            print("Insufficient data for plotting")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
        
        # Equity curve
        equity_series = pd.Series(self.equity_curve, index=self.timestamps)
        ax1.plot(equity_series, label='Equity', linewidth=2, color='blue')
        ax1.fill_between(equity_series.index, self.initial_capital, equity_series.values,
                         where=(equity_series.values > self.initial_capital),
                         color='green', alpha=0.3, label='Profit')
        ax1.fill_between(equity_series.index, self.initial_capital, equity_series.values,
                         where=(equity_series.values <= self.initial_capital),
                         color='red', alpha=0.3, label='Loss')
        ax1.axhline(y=self.initial_capital, color='black', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Equity ($)')
        ax1.set_title('Equity Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        ax2.fill_between(drawdown.index, 0, drawdown.values,
                         color='red', alpha=0.5)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.set_title('Drawdown')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Integration helper

def analyze_trading_performance(trades: List[Dict], initial_capital: float = 1000) -> Dict:
    """
    Quick function to analyze trading performance
    
    Args:
        trades: List of trade dictionaries
        initial_capital: Starting capital
        
    Returns:
        Performance metrics dictionary
    """
    analytics = PerformanceAnalytics(initial_capital)
    
    for trade in trades:
        analytics.record_trade(trade)
    
    return analytics.calculate_metrics()

if __name__ == "__main__":
    # Test the performance analytics
    analytics = PerformanceAnalytics(initial_capital=1000)
    
    # Simulate some trades
    from datetime import datetime, timedelta
    import random
    
    for i in range(50):
        entry_time = datetime.now() - timedelta(days=50-i)
        exit_time = entry_time + timedelta(hours=random.randint(1, 24))
        
        # Simulate win/loss (60% win rate)
        is_win = random.random() < 0.60
        
        if is_win:
            profit = random.uniform(10, 50)
        else:
            profit = -random.uniform(5, 30)
        
        trade = {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'pair': 'EURUSD',
            'direction': random.choice(['BUY', 'SELL']),
            'entry_price': 1.0850,
            'exit_price': 1.0850 + (profit / 10000),
            'position_size': 100,
            'profit': profit,
            'commission': 0.5
        }
        
        analytics.record_trade(trade)
    
    # Calculate and print metrics
    metrics = analytics.calculate_metrics()
    print(analytics.generate_report())