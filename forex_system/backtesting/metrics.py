"""
Performance metrics for backtesting.

Calculates comprehensive trading metrics including risk-adjusted returns.
"""
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass


class BacktestMetrics:
    """
    Calculate comprehensive trading performance metrics.

    Metrics include:
    - Return metrics: Total return, CAGR
    - Risk metrics: Sharpe ratio, max drawdown, volatility
    - Trade metrics: Win rate, profit factor, avg win/loss
    - Consistency: Longest win/loss streaks
    """

    def __init__(
        self,
        trades: List,
        equity_curve: List[float],
        initial_capital: float,
        risk_free_rate: float = 0.02  # 2% annual risk-free rate
    ):
        """
        Initialize metrics calculator.

        Args:
            trades: List of Trade objects
            equity_curve: List of equity values over time
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.trades = trades
        self.equity_curve = equity_curve
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

    def calculate_all(self) -> Dict[str, Any]:
        """Calculate all performance metrics."""
        if not self.trades:
            return self._empty_metrics()

        return {
            **self._return_metrics(),
            **self._risk_metrics(),
            **self._trade_metrics(),
            **self._consistency_metrics()
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return metrics for case with no trades."""
        return {
            'total_return': 0.0,
            'total_return_pct': 0.0,
            'cagr': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'volatility': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_trade': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'longest_win_streak': 0,
            'longest_loss_streak': 0,
            'avg_trade_duration': 0.0
        }

    def _return_metrics(self) -> Dict[str, float]:
        """Calculate return-based metrics."""
        final_capital = self.equity_curve[-1]
        total_return = final_capital - self.initial_capital
        total_return_pct = (final_capital / self.initial_capital - 1) * 100

        # Approximate CAGR (assuming hourly data)
        num_hours = len(self.equity_curve)
        num_years = num_hours / (365 * 24)  # Approximate
        if num_years > 0:
            cagr = (pow(final_capital / self.initial_capital, 1 / num_years) - 1) * 100
        else:
            cagr = 0.0

        return {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'cagr': cagr
        }

    def _risk_metrics(self) -> Dict[str, float]:
        """Calculate risk-based metrics."""
        # Calculate returns
        equity_array = np.array(self.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]

        # Sharpe ratio (annualized)
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                # Annualize: sqrt(8760) for hourly data
                sharpe_ratio = (mean_return - self.risk_free_rate / 8760) / std_return * np.sqrt(8760)
            else:
                sharpe_ratio = 0.0
            volatility = std_return * np.sqrt(8760) * 100  # Annualized volatility
        else:
            sharpe_ratio = 0.0
            volatility = 0.0

        # Max drawdown
        peak = equity_array[0]
        max_dd = 0.0
        max_dd_pct = 0.0

        for value in equity_array:
            if value > peak:
                peak = value
            drawdown = peak - value
            drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0

            if drawdown > max_dd:
                max_dd = drawdown
                max_dd_pct = drawdown_pct

        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'volatility': volatility
        }

    def _trade_metrics(self) -> Dict[str, Any]:
        """Calculate trade-based metrics."""
        num_trades = len(self.trades)

        if num_trades == 0:
            return {
                'num_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'avg_trade': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }

        # Separate wins and losses
        wins = [t.profit_loss for t in self.trades if t.profit_loss > 0]
        losses = [t.profit_loss for t in self.trades if t.profit_loss < 0]

        num_wins = len(wins)
        num_losses = len(losses)
        win_rate = (num_wins / num_trades) * 100 if num_trades > 0 else 0

        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Average metrics
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        avg_trade = np.mean([t.profit_loss for t in self.trades])

        # Largest win/loss
        largest_win = max(wins) if wins else 0.0
        largest_loss = min(losses) if losses else 0.0

        return {
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }

    def _consistency_metrics(self) -> Dict[str, Any]:
        """Calculate consistency metrics."""
        if not self.trades:
            return {
                'longest_win_streak': 0,
                'longest_loss_streak': 0,
                'avg_trade_duration': 0.0
            }

        # Calculate streaks
        current_win_streak = 0
        current_loss_streak = 0
        longest_win_streak = 0
        longest_loss_streak = 0

        for trade in self.trades:
            if trade.profit_loss > 0:
                current_win_streak += 1
                current_loss_streak = 0
                longest_win_streak = max(longest_win_streak, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                longest_loss_streak = max(longest_loss_streak, current_loss_streak)

        # Average trade duration (in hours if timestamps are available)
        durations = []
        for trade in self.trades:
            if hasattr(trade.entry_time, 'timestamp') and hasattr(trade.exit_time, 'timestamp'):
                duration = (trade.exit_time.timestamp() - trade.entry_time.timestamp()) / 3600
                durations.append(duration)
            elif isinstance(trade.entry_time, (int, float)) and isinstance(trade.exit_time, (int, float)):
                # Bar indices
                durations.append(trade.exit_time - trade.entry_time)

        avg_duration = np.mean(durations) if durations else 0.0

        return {
            'longest_win_streak': longest_win_streak,
            'longest_loss_streak': longest_loss_streak,
            'avg_trade_duration': avg_duration
        }

    def print_summary(self, metrics: Dict[str, Any]) -> None:
        """Print formatted metrics summary."""
        print("\n" + "=" * 80)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("=" * 80)

        print("\nRETURN METRICS:")
        print(f"  Total Return:        ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2f}%)")
        print(f"  CAGR:                {metrics['cagr']:.2f}%")

        print("\nRISK METRICS:")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:        ${metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_pct']:.2f}%)")
        print(f"  Volatility:          {metrics['volatility']:.2f}%")

        print("\nTRADE METRICS:")
        print(f"  Total Trades:        {metrics['num_trades']}")
        print(f"  Win Rate:            {metrics['win_rate']:.2f}%")
        print(f"  Profit Factor:       {metrics['profit_factor']:.2f}")
        print(f"  Avg Win:             ${metrics['avg_win']:.2f}")
        print(f"  Avg Loss:            ${metrics['avg_loss']:.2f}")
        print(f"  Avg Trade:           ${metrics['avg_trade']:.2f}")
        print(f"  Largest Win:         ${metrics['largest_win']:.2f}")
        print(f"  Largest Loss:        ${metrics['largest_loss']:.2f}")

        print("\nCONSISTENCY METRICS:")
        print(f"  Longest Win Streak:  {metrics['longest_win_streak']}")
        print(f"  Longest Loss Streak: {metrics['longest_loss_streak']}")
        print(f"  Avg Trade Duration:  {metrics['avg_trade_duration']:.1f} hours")

        print("=" * 80)

        # Qualitative assessment
        print("\nQUALITATIVE ASSESSMENT:")
        if metrics['sharpe_ratio'] > 2.0:
            print("  • Excellent risk-adjusted returns")
        elif metrics['sharpe_ratio'] > 1.0:
            print("  • Good risk-adjusted returns")
        elif metrics['sharpe_ratio'] > 0:
            print("  • Positive but modest risk-adjusted returns")
        else:
            print("  • Poor risk-adjusted returns")

        if metrics['win_rate'] > 55:
            print("  • Strong win rate")
        elif metrics['win_rate'] > 50:
            print("  • Above break-even win rate")
        else:
            print("  • Below break-even win rate")

        if metrics['profit_factor'] > 2.0:
            print("  • Excellent profit factor")
        elif metrics['profit_factor'] > 1.5:
            print("  • Good profit factor")
        elif metrics['profit_factor'] > 1.0:
            print("  • Profitable but modest profit factor")
        else:
            print("  • Unprofitable strategy")

        print("=" * 80 + "\n")
