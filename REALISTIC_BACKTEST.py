#!/usr/bin/env python3
"""
REALISTIC BACKTEST
==================
This runs a REAL trading simulation with:
- Actual P&L calculation
- Spread/commission costs
- Proper position sizing
- Drawdown tracking
- Win rate and expectancy

NO FAKE NUMBERS - Everything is calculated from actual simulated trades.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from UPGRADED_FEATURES import (
    generate_all_upgraded_features,
    generate_labels_atr_based,
    calculate_confidence_margin,
    compute_sample_weights,
    calculate_hurst_exponent,
    calculate_atr
)

from sklearn.ensemble import RandomForestClassifier


@dataclass
class Trade:
    """Individual trade record."""
    entry_time: datetime
    exit_time: datetime
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    position_size: float  # In units
    pnl: float  # In dollars
    pnl_pips: float
    outcome: str  # "WIN", "LOSS", "BREAKEVEN"
    confidence: float
    hold_bars: int


@dataclass
class BacktestResult:
    """Complete backtest results."""
    # Basic stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # P&L
    total_pnl: float
    gross_profit: float
    gross_loss: float
    profit_factor: float

    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float

    # Per trade
    avg_win: float
    avg_loss: float
    avg_trade: float
    expectancy: float  # Expected $ per trade
    expectancy_r: float  # Expected R per trade

    # Time
    avg_hold_bars: float

    # Capital
    starting_capital: float
    ending_capital: float
    total_return: float
    monthly_return: float

    # Trade list
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class RealisticBacktester:
    """
    Realistic backtesting engine with proper costs and position sizing.
    """

    # Realistic forex costs
    SPREAD_PIPS = 1.5  # Average spread for major pairs
    COMMISSION_PER_LOT = 7.0  # $7 per standard lot round trip
    SLIPPAGE_PIPS = 0.5  # Average slippage

    # Position sizing
    RISK_PER_TRADE = 0.015  # 1.5% risk per trade

    # Trade management
    MIN_CONFIDENCE = 0.10  # Minimum confidence to take trade

    def __init__(
        self,
        starting_capital: float = 30000,
        risk_per_trade: float = 0.015,
        use_scaling: bool = True  # Scale out at multiple TPs
    ):
        self.starting_capital = starting_capital
        self.risk_per_trade = risk_per_trade
        self.use_scaling = use_scaling

        self.model = None
        self.feature_names = None

    def run_backtest(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.6,
        atr_sl_multiplier: float = 1.5,
        atr_tp_multiplier: float = 3.0
    ) -> BacktestResult:
        """
        Run complete backtest.

        Args:
            df: OHLCV DataFrame
            train_ratio: Ratio of data for training
            atr_sl_multiplier: Stop loss = ATR * multiplier
            atr_tp_multiplier: Take profit = ATR * multiplier

        Returns:
            BacktestResult with all metrics
        """
        print("\n" + "="*70)
        print("  REALISTIC BACKTEST")
        print("="*70)

        # 1. Generate features and labels
        print("\n[1] Preparing data...")
        features_df = generate_all_upgraded_features(df)
        labels = generate_labels_atr_based(df, lookahead_bars=20, atr_multiplier=2.0)

        # Add ATR for position sizing
        features_df['ATR'] = calculate_atr(df, 14)

        # 2. Split train/test
        split_idx = int(len(df) * train_ratio)

        train_features = features_df.iloc[:split_idx]
        train_labels = labels.iloc[:split_idx]

        test_features = features_df.iloc[split_idx:]
        test_df = df.iloc[split_idx:]

        print(f"    Training period: {len(train_features)} bars")
        print(f"    Testing period: {len(test_features)} bars")

        # 3. Train model
        print("\n[2] Training model...")
        self._train_model(train_features, train_labels)

        # 4. Run trading simulation
        print("\n[3] Running trading simulation...")
        result = self._simulate_trading(
            test_df,
            test_features,
            atr_sl_multiplier,
            atr_tp_multiplier
        )

        return result

    def _train_model(self, features: pd.DataFrame, labels: pd.Series):
        """Train the ML model."""
        valid_mask = ~labels.isna()
        X = features[valid_mask]
        y = labels[valid_mask]

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols]
        self.feature_names = numeric_cols

        sample_weights = compute_sample_weights(y)

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X, y, sample_weight=sample_weights)
        print(f"    Model trained on {len(X)} samples")

    def _simulate_trading(
        self,
        price_df: pd.DataFrame,
        features_df: pd.DataFrame,
        atr_sl_mult: float,
        atr_tp_mult: float
    ) -> BacktestResult:
        """
        Simulate actual trading with realistic conditions.
        """
        capital = self.starting_capital
        equity_curve = [capital]
        trades = []

        peak_capital = capital
        max_drawdown = 0
        max_drawdown_pct = 0

        in_trade = False
        current_trade = None

        X = features_df[self.feature_names]

        signals_generated = 0
        signals_filtered = 0

        for i in range(len(price_df) - 1):
            current_bar = price_df.iloc[i]
            next_bar = price_df.iloc[i + 1]

            current_time = price_df.index[i]
            current_price = current_bar['close']
            current_atr = features_df['ATR'].iloc[i]

            if pd.isna(current_atr) or current_atr <= 0:
                current_atr = 0.001

            # Check if we should exit current trade
            if in_trade and current_trade is not None:
                exit_price, exit_reason = self._check_exit(
                    current_trade, next_bar, current_atr
                )

                if exit_price is not None:
                    # Calculate P&L
                    trade = self._close_trade(
                        current_trade,
                        exit_price,
                        price_df.index[i + 1],
                        exit_reason,
                        i + 1 - current_trade['entry_idx']
                    )
                    trades.append(trade)

                    capital += trade.pnl
                    equity_curve.append(capital)

                    # Track drawdown
                    if capital > peak_capital:
                        peak_capital = capital
                    dd = peak_capital - capital
                    dd_pct = dd / peak_capital if peak_capital > 0 else 0
                    if dd > max_drawdown:
                        max_drawdown = dd
                        max_drawdown_pct = dd_pct

                    in_trade = False
                    current_trade = None

            # Check for new signal if not in trade
            if not in_trade:
                row = X.iloc[[i]]
                pred = self.model.predict(row)[0]
                proba = self.model.predict_proba(row)[0]
                confidence = calculate_confidence_margin(proba)

                # Only trade BUY or SELL with sufficient confidence
                if pred != 0 and confidence >= self.MIN_CONFIDENCE:
                    signals_generated += 1

                    direction = "LONG" if pred == 1 else "SHORT"

                    # Calculate position size based on risk
                    sl_distance = current_atr * atr_sl_mult
                    tp_distance = current_atr * atr_tp_mult

                    # Risk amount
                    risk_amount = capital * self.risk_per_trade

                    # Position size (in price units, simplified)
                    # For forex: position_size = risk / (sl_pips * pip_value)
                    pip_value = 0.0001  # For pairs like EURUSD
                    sl_pips = sl_distance / pip_value

                    if sl_pips > 0:
                        # Lots calculation (simplified)
                        position_value = risk_amount / (sl_pips * 10)  # $10 per pip per lot

                        # Entry price with slippage
                        entry_price = current_price + (self.SLIPPAGE_PIPS * pip_value * (1 if direction == "LONG" else -1))

                        if direction == "LONG":
                            stop_loss = entry_price - sl_distance
                            take_profit = entry_price + tp_distance
                        else:
                            stop_loss = entry_price + sl_distance
                            take_profit = entry_price - tp_distance

                        current_trade = {
                            'entry_time': current_time,
                            'entry_price': entry_price,
                            'direction': direction,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'position_value': position_value,
                            'risk_amount': risk_amount,
                            'confidence': confidence,
                            'entry_idx': i,
                            'sl_distance': sl_distance,
                            'tp_distance': tp_distance
                        }

                        in_trade = True
                else:
                    if pred != 0:
                        signals_filtered += 1

        # Calculate results
        if not trades:
            print("    No trades executed!")
            return self._empty_result()

        winning = [t for t in trades if t.outcome == "WIN"]
        losing = [t for t in trades if t.outcome == "LOSS"]

        total_pnl = sum(t.pnl for t in trades)
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))

        win_rate = len(winning) / len(trades) if trades else 0
        avg_win = np.mean([t.pnl for t in winning]) if winning else 0
        avg_loss = abs(np.mean([t.pnl for t in losing])) if losing else 0
        avg_trade = np.mean([t.pnl for t in trades])

        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Expectancy in R (risk units)
        avg_risk = np.mean([self.starting_capital * self.risk_per_trade for _ in trades])
        expectancy_r = expectancy / avg_risk if avg_risk > 0 else 0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Sharpe ratio (simplified)
        returns = pd.Series([t.pnl / self.starting_capital for t in trades])
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        # Time calculations
        test_days = len(price_df) * 4 / 24  # 4H bars to days
        test_months = test_days / 30

        total_return = (capital - self.starting_capital) / self.starting_capital
        monthly_return = (1 + total_return) ** (1 / test_months) - 1 if test_months > 0 else 0

        result = BacktestResult(
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            total_pnl=total_pnl,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            expectancy=expectancy,
            expectancy_r=expectancy_r,
            avg_hold_bars=np.mean([t.hold_bars for t in trades]),
            starting_capital=self.starting_capital,
            ending_capital=capital,
            total_return=total_return,
            monthly_return=monthly_return,
            trades=trades,
            equity_curve=equity_curve
        )

        # Print results
        self._print_results(result, signals_generated, signals_filtered, test_days)

        return result

    def _check_exit(self, trade: Dict, bar: pd.Series, atr: float) -> Tuple[Optional[float], str]:
        """Check if trade should be exited."""
        high = bar['high']
        low = bar['low']
        close = bar['close']

        if trade['direction'] == "LONG":
            # Check stop loss
            if low <= trade['stop_loss']:
                return trade['stop_loss'] - (self.SLIPPAGE_PIPS * 0.0001), "SL"
            # Check take profit
            if high >= trade['take_profit']:
                return trade['take_profit'] - (self.SPREAD_PIPS * 0.0001), "TP"
        else:  # SHORT
            # Check stop loss
            if high >= trade['stop_loss']:
                return trade['stop_loss'] + (self.SLIPPAGE_PIPS * 0.0001), "SL"
            # Check take profit
            if low <= trade['take_profit']:
                return trade['take_profit'] + (self.SPREAD_PIPS * 0.0001), "TP"

        return None, ""

    def _close_trade(
        self,
        trade: Dict,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        hold_bars: int
    ) -> Trade:
        """Close trade and calculate P&L."""

        entry_price = trade['entry_price']
        direction = trade['direction']

        # Calculate pip movement
        if direction == "LONG":
            pnl_pips = (exit_price - entry_price) / 0.0001
        else:
            pnl_pips = (entry_price - exit_price) / 0.0001

        # Subtract spread
        pnl_pips -= self.SPREAD_PIPS

        # Calculate dollar P&L
        # Simplified: $10 per pip per standard lot
        position_lots = trade['position_value']
        pnl = pnl_pips * 10 * position_lots

        # Subtract commission
        pnl -= self.COMMISSION_PER_LOT * position_lots

        outcome = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BREAKEVEN")

        return Trade(
            entry_time=trade['entry_time'],
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=trade['stop_loss'],
            take_profit=trade['take_profit'],
            position_size=trade['position_value'],
            pnl=pnl,
            pnl_pips=pnl_pips,
            outcome=outcome,
            confidence=trade['confidence'],
            hold_bars=hold_bars
        )

    def _print_results(self, result: BacktestResult, signals: int, filtered: int, days: float):
        """Print formatted results."""
        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         BACKTEST RESULTS                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  TEST PERIOD: {days:.0f} days                                                       ║
║                                                                              ║
║  TRADE STATISTICS:                                                           ║
║  ├── Total Trades: {result.total_trades:5d}                                               ║
║  ├── Winning: {result.winning_trades:5d} ({result.win_rate:.1%})                                         ║
║  ├── Losing: {result.losing_trades:5d} ({1-result.win_rate:.1%})                                         ║
║  ├── Signals Generated: {signals:5d}                                             ║
║  └── Signals Filtered (low conf): {filtered:5d}                                  ║
║                                                                              ║
║  PROFIT & LOSS:                                                              ║
║  ├── Total P&L: ${result.total_pnl:>+10,.2f}                                         ║
║  ├── Gross Profit: ${result.gross_profit:>10,.2f}                                       ║
║  ├── Gross Loss: ${result.gross_loss:>10,.2f}                                        ║
║  └── Profit Factor: {result.profit_factor:>6.2f}                                            ║
║                                                                              ║
║  PER TRADE:                                                                  ║
║  ├── Average Win: ${result.avg_win:>+10,.2f}                                          ║
║  ├── Average Loss: ${result.avg_loss:>10,.2f}                                        ║
║  ├── Average Trade: ${result.avg_trade:>+10,.2f}                                        ║
║  ├── Expectancy: ${result.expectancy:>+10,.2f} per trade                               ║
║  └── Expectancy (R): {result.expectancy_r:>+6.2f}R per trade                              ║
║                                                                              ║
║  RISK METRICS:                                                               ║
║  ├── Max Drawdown: ${result.max_drawdown:>10,.2f} ({result.max_drawdown_pct:.1%})                        ║
║  ├── Sharpe Ratio: {result.sharpe_ratio:>6.2f}                                             ║
║  └── Avg Hold Time: {result.avg_hold_bars:.1f} bars                                       ║
║                                                                              ║
║  CAPITAL:                                                                    ║
║  ├── Starting: ${result.starting_capital:>12,.2f}                                       ║
║  ├── Ending: ${result.ending_capital:>12,.2f}                                         ║
║  ├── Total Return: {result.total_return:>+8.1%}                                           ║
║  └── Monthly Return: {result.monthly_return:>+7.1%}                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

        # Interpretation
        print("\n  INTERPRETATION:")
        print("  " + "-"*60)

        if result.win_rate >= 0.55:
            print(f"  ✅ Win rate {result.win_rate:.1%} is GOOD (above 55%)")
        elif result.win_rate >= 0.45:
            print(f"  ⚠️  Win rate {result.win_rate:.1%} is OK (needs good R:R)")
        else:
            print(f"  ❌ Win rate {result.win_rate:.1%} is LOW (needs improvement)")

        if result.profit_factor >= 1.5:
            print(f"  ✅ Profit factor {result.profit_factor:.2f} is GOOD (above 1.5)")
        elif result.profit_factor >= 1.0:
            print(f"  ⚠️  Profit factor {result.profit_factor:.2f} is MARGINAL")
        else:
            print(f"  ❌ Profit factor {result.profit_factor:.2f} is LOSING")

        if result.expectancy_r >= 0.2:
            print(f"  ✅ Expectancy {result.expectancy_r:.2f}R is GOOD")
        elif result.expectancy_r >= 0:
            print(f"  ⚠️  Expectancy {result.expectancy_r:.2f}R is MARGINAL")
        else:
            print(f"  ❌ Expectancy {result.expectancy_r:.2f}R is NEGATIVE")

        if result.max_drawdown_pct <= 0.15:
            print(f"  ✅ Max drawdown {result.max_drawdown_pct:.1%} is ACCEPTABLE")
        elif result.max_drawdown_pct <= 0.25:
            print(f"  ⚠️  Max drawdown {result.max_drawdown_pct:.1%} is HIGH")
        else:
            print(f"  ❌ Max drawdown {result.max_drawdown_pct:.1%} is DANGEROUS")

        print()

    def _empty_result(self) -> BacktestResult:
        """Return empty result when no trades."""
        return BacktestResult(
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
            total_pnl=0, gross_profit=0, gross_loss=0, profit_factor=0,
            max_drawdown=0, max_drawdown_pct=0, sharpe_ratio=0,
            avg_win=0, avg_loss=0, avg_trade=0, expectancy=0, expectancy_r=0,
            avg_hold_bars=0, starting_capital=self.starting_capital,
            ending_capital=self.starting_capital, total_return=0, monthly_return=0
        )


def run_realistic_backtest():
    """Run the realistic backtest with simulated market data."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    REALISTIC TRADING BACKTEST                                ║
║                                                                              ║
║  This simulation includes:                                                   ║
║  • Spread costs (1.5 pips)                                                  ║
║  • Commission ($7 per lot)                                                   ║
║  • Slippage (0.5 pips)                                                      ║
║  • Proper position sizing (1.5% risk)                                       ║
║  • ATR-based stops and targets                                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Generate realistic market data
    # Using random walk with some trending behavior
    np.random.seed(42)
    n = 3000  # 3000 4H bars = 500 days

    print("Generating realistic market data (500 days of 4H bars)...\n")

    dates = pd.date_range(start='2023-01-01', periods=n, freq='4h')

    # Create more realistic price movement with trends
    returns = np.random.normal(0, 0.0008, n)  # Random returns

    # Add some trending behavior
    trend = np.zeros(n)
    current_trend = 0
    for i in range(n):
        if np.random.random() < 0.02:  # 2% chance to change trend
            current_trend = np.random.choice([-0.0002, 0, 0.0002])
        trend[i] = current_trend

    returns = returns + trend

    # Generate prices
    close = 1.1000 * np.cumprod(1 + returns)

    # Generate OHLC
    volatility = np.abs(np.random.normal(0.0003, 0.0001, n))
    high = close + volatility
    low = close - volatility
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    # Add some realistic OHLC relationships
    for i in range(n):
        if close[i] > open_[i]:  # Bullish bar
            low[i] = min(low[i], open_[i] - np.random.uniform(0, 0.0002))
            high[i] = max(high[i], close[i] + np.random.uniform(0, 0.0002))
        else:  # Bearish bar
            high[i] = max(high[i], open_[i] + np.random.uniform(0, 0.0002))
            low[i] = min(low[i], close[i] - np.random.uniform(0, 0.0002))

    volume = np.random.randint(5000, 25000, n)

    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    # Run backtest
    backtester = RealisticBacktester(
        starting_capital=30000,
        risk_per_trade=0.015  # 1.5% risk
    )

    result = backtester.run_backtest(
        df,
        train_ratio=0.6,  # 60% train, 40% test
        atr_sl_multiplier=1.5,
        atr_tp_multiplier=3.0  # 1:2 R:R
    )

    # Project to different timeframes
    if result.total_trades > 0:
        print("\n" + "="*70)
        print("  PROJECTIONS WITH $30,000 CAPITAL")
        print("="*70)

        # Calculate trades per month
        test_months = (len(df) * 0.4 * 4) / 24 / 30  # 40% test, 4H bars to days to months
        trades_per_month = result.total_trades / test_months if test_months > 0 else 0

        print(f"""
  Based on backtest results:

  ┌────────────────────────────────────────────────────────────────┐
  │  METRIC              │  BACKTEST     │  PROJECTED/MONTH       │
  ├────────────────────────────────────────────────────────────────┤
  │  Trades              │  {result.total_trades:5d}         │  {trades_per_month:.1f}/month             │
  │  Win Rate            │  {result.win_rate:.1%}         │  -                      │
  │  Avg Trade           │  ${result.avg_trade:>+8.2f}   │  -                      │
  │  Monthly Return      │  -            │  {result.monthly_return:>+6.1%}                │
  │  Expected Monthly $  │  -            │  ${result.monthly_return * 30000:>+8.0f}             │
  └────────────────────────────────────────────────────────────────┘

  YEARLY PROJECTION (with compounding):
  ├── Conservative (50% of backtest): ${30000 * ((1 + result.monthly_return * 0.5) ** 12 - 1):>+10,.0f} ({((1 + result.monthly_return * 0.5) ** 12 - 1) * 100:>+.0f}%)
  ├── Expected (75% of backtest):     ${30000 * ((1 + result.monthly_return * 0.75) ** 12 - 1):>+10,.0f} ({((1 + result.monthly_return * 0.75) ** 12 - 1) * 100:>+.0f}%)
  └── Optimistic (100% of backtest):  ${30000 * ((1 + result.monthly_return) ** 12 - 1):>+10,.0f} ({((1 + result.monthly_return) ** 12 - 1) * 100:>+.0f}%)

  ⚠️  IMPORTANT CAVEATS:
  • Backtest ≠ Live trading (expect 30-50% worse in reality)
  • Market conditions change
  • Emotions affect execution
  • These are SIMULATED results, not guaranteed
""")

    return result


if __name__ == "__main__":
    result = run_realistic_backtest()
