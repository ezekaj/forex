#!/usr/bin/env python3
"""
UNIFIED FOREX TRADING SYSTEM
============================
Merges news_lenci_forex intelligence with forex trading system.

This is the main entry point for the integrated, profitable trading system.

USAGE:
    python run_integrated_system.py --mode backtest --pair EURUSD --days 30
    python run_integrated_system.py --mode live --pair EURUSD
    python run_integrated_system.py --mode analyze --pair EURUSD

Features:
1. News-enhanced signal generation
2. Multi-timeframe confirmation
3. Regime-adaptive position sizing
4. Professional risk management
5. Comprehensive backtesting
"""
import argparse
import asyncio
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add forex_system to path
sys.path.insert(0, str(Path(__file__).parent))

from forex_system.strategies.integrated_news_strategy import (
    IntegratedNewsStrategy, create_integrated_strategy
)
from forex_system.strategies.random_forest import RandomForestStrategy
from forex_system.services.feature_engineering import FeatureEngineer
from forex_system.risk_manager import create_conservative_risk_manager, RiskManager


class Colors:
    """Terminal colors for output."""
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    DIM = '\033[2m'
    END = '\033[0m'


def print_banner():
    """Print system banner."""
    print(f"""
{Colors.CYAN}{'='*70}{Colors.END}
{Colors.BOLD}  UNIFIED FOREX TRADING SYSTEM{Colors.END}
{Colors.CYAN}  News Intelligence + ML + Risk Management{Colors.END}
{Colors.CYAN}{'='*70}{Colors.END}

{Colors.GREEN}Features:{Colors.END}
  • Multi-source signal aggregation (news, technical, sentiment)
  • Multi-timeframe trend confirmation (H4 direction, H1 entry)
  • News event filter (avoid high-impact events)
  • VIX-based regime adaptation
  • Correlation-aware position sizing
  • Professional risk management with circuit breakers

{Colors.YELLOW}IMPORTANT:{Colors.END} This system is designed for educational purposes.
Always use proper risk management and never trade with money you can't afford to lose.
""")


async def run_analysis(pair: str):
    """Run market analysis for a pair."""
    print(f"\n{Colors.BOLD}MARKET ANALYSIS: {pair}{Colors.END}")
    print(f"{Colors.CYAN}{'─'*50}{Colors.END}\n")

    # Create strategy
    strategy = create_integrated_strategy(pair=pair, conservative=True)

    # Get all signals
    from forex_system.services.forex_news_service import ForexNewsService
    from forex_system.services.enhanced_forex_signals import EnhancedForexSignals

    news_service = ForexNewsService()
    enhanced_signals = EnhancedForexSignals()

    # Fundamental analysis
    print(f"{Colors.BOLD}1. FUNDAMENTAL ANALYSIS{Colors.END}")
    fund_signal = news_service.get_fundamental_signal(pair)
    print(f"   Signal: {fund_signal['signal']}")
    print(f"   Strength: {fund_signal['strength']:.2f}")
    print(f"   Avoid Trading: {fund_signal['avoid_trading']}")
    for reason in fund_signal['reasons'][:3]:
        print(f"   → {reason}")

    # Multi-timeframe analysis
    print(f"\n{Colors.BOLD}2. MULTI-TIMEFRAME ANALYSIS{Colors.END}")
    mtf_signal = await enhanced_signals.get_multi_timeframe_signal(pair)
    print(f"   Signal: {mtf_signal['signal']}")
    print(f"   Strength: {mtf_signal['strength']:.2f}")
    print(f"   H4 Trend: {mtf_signal.get('h4_trend', 'N/A')}")
    print(f"   H1 Trend: {mtf_signal.get('h1_trend', 'N/A')}")
    print(f"   RSI (1H): {mtf_signal.get('rsi_1h', 0):.1f}")

    # Risk sentiment
    print(f"\n{Colors.BOLD}3. RISK SENTIMENT{Colors.END}")
    risk_signal = await enhanced_signals.get_risk_sentiment_signal(pair)
    print(f"   Signal: {risk_signal['signal']}")
    print(f"   Regime: {risk_signal['regime']}")
    print(f"   VIX: {risk_signal.get('vix', 'N/A')}")
    print(f"   Confidence Multiplier: {risk_signal['confidence_multiplier']:.2f}")

    # Correlation analysis
    print(f"\n{Colors.BOLD}4. CORRELATION ANALYSIS{Colors.END}")
    corr_signal = await enhanced_signals.get_correlation_signal(pair)
    print(f"   Signal: {corr_signal['signal']}")
    print(f"   Strength: {corr_signal['strength']:.2f}")
    print(f"   Reason: {corr_signal.get('reason', 'N/A')}")

    # Combined signal
    print(f"\n{Colors.BOLD}5. COMBINED SIGNAL{Colors.END}")
    combined = await enhanced_signals.get_combined_signal(pair)
    signal_color = Colors.GREEN if combined['signal'] == 'BUY' else Colors.RED if combined['signal'] == 'SELL' else Colors.YELLOW
    print(f"   {signal_color}Signal: {combined['signal']}{Colors.END}")
    print(f"   Strength: {combined['strength']:.2f}")
    print(f"   Score: {combined['score']:+.3f}")

    print(f"\n{Colors.CYAN}{'─'*50}{Colors.END}")
    print(f"{Colors.BOLD}RECOMMENDATION:{Colors.END}")

    if fund_signal['avoid_trading']:
        print(f"   {Colors.RED}⚠️  AVOID TRADING - High-impact event nearby{Colors.END}")
    elif combined['signal'] == 'NEUTRAL':
        print(f"   {Colors.YELLOW}⏸  NO TRADE - Signals not aligned{Colors.END}")
    elif combined['strength'] < 0.3:
        print(f"   {Colors.YELLOW}⚠️  WEAK SIGNAL - Wait for better setup{Colors.END}")
    else:
        if combined['signal'] == 'BUY':
            print(f"   {Colors.GREEN}✅ POTENTIAL LONG - Strength: {combined['strength']:.0%}{Colors.END}")
        else:
            print(f"   {Colors.RED}✅ POTENTIAL SHORT - Strength: {combined['strength']:.0%}{Colors.END}")

        print(f"\n   Key Reasons:")
        for reason in combined.get('explanations', [])[:3]:
            print(f"   → {reason}")

    # Cleanup
    await news_service.close()
    await enhanced_signals.close()

    print()


async def run_backtest(pair: str, days: int, initial_capital: float):
    """Run backtest of the integrated strategy."""
    print(f"\n{Colors.BOLD}BACKTEST: {pair} ({days} days){Colors.END}")
    print(f"{Colors.CYAN}{'─'*50}{Colors.END}")
    print(f"   Initial Capital: ${initial_capital:,.2f}")
    print(f"   Strategy: Integrated News + Technical")
    print()

    # This is a simplified backtest - in production you'd use the full engine
    from forex_system.services.enhanced_forex_signals import EnhancedForexSignals

    signals_service = EnhancedForexSignals()

    # Fetch historical data
    print(f"   Fetching historical data...")
    ohlc = await signals_service.fetch_ohlc(pair, '1h', bars=24 * days)

    if not ohlc:
        print(f"   {Colors.RED}Error: Could not fetch historical data{Colors.END}")
        return

    print(f"   Bars fetched: {len(ohlc)}")

    # Create strategy
    strategy = create_integrated_strategy(pair=pair, conservative=True)

    # Create risk manager
    risk_manager = create_conservative_risk_manager(initial_capital)

    # Simulate trading
    trades = []
    equity_curve = [initial_capital]

    print(f"\n   Running simulation...")

    # Simplified simulation (in production, use full backtesting engine)
    position = None
    wins = 0
    losses = 0

    for i in range(50, len(ohlc)):
        bar = ohlc[i]

        # Get signal for this bar
        combined = await signals_service.get_combined_signal(pair)

        current_equity = equity_curve[-1]

        if position is None:
            # Check for entry
            if combined['signal'] == 'BUY' and combined['strength'] > 0.3:
                can_trade, reason = risk_manager.can_open_trade(pair, 'LONG', 1.5)
                if can_trade:
                    position = {
                        'direction': 'LONG',
                        'entry': bar['close'],
                        'entry_bar': i,
                        'atr': combined.get('atr', 0.001)
                    }
            elif combined['signal'] == 'SELL' and combined['strength'] > 0.3:
                can_trade, reason = risk_manager.can_open_trade(pair, 'SHORT', 1.5)
                if can_trade:
                    position = {
                        'direction': 'SHORT',
                        'entry': bar['close'],
                        'entry_bar': i,
                        'atr': combined.get('atr', 0.001)
                    }
        else:
            # Check for exit
            bars_held = i - position['entry_bar']
            pnl_pips = (bar['close'] - position['entry']) * 10000
            if position['direction'] == 'SHORT':
                pnl_pips = -pnl_pips

            # Simple exit logic: SL/TP or opposite signal
            atr_pips = position['atr'] * 10000
            stop_loss = atr_pips * 2
            take_profit = atr_pips * 3

            should_exit = False
            exit_reason = ""

            if pnl_pips <= -stop_loss:
                should_exit = True
                exit_reason = "SL"
            elif pnl_pips >= take_profit:
                should_exit = True
                exit_reason = "TP"
            elif bars_held > 48:  # Max 48 bars hold
                should_exit = True
                exit_reason = "TIME"
            elif (position['direction'] == 'LONG' and combined['signal'] == 'SELL') or \
                 (position['direction'] == 'SHORT' and combined['signal'] == 'BUY'):
                should_exit = True
                exit_reason = "SIGNAL"

            if should_exit:
                # Calculate P&L
                risk_pct = 0.01  # 1% risk
                risk_amount = current_equity * risk_pct
                position_size = risk_amount / stop_loss if stop_loss > 0 else 0

                pnl = pnl_pips * position_size * 0.1  # Simplified P&L

                trades.append({
                    'direction': position['direction'],
                    'entry': position['entry'],
                    'exit': bar['close'],
                    'pnl_pips': pnl_pips,
                    'pnl': pnl,
                    'exit_reason': exit_reason
                })

                current_equity += pnl
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

                position = None

        equity_curve.append(current_equity)

    await signals_service.close()

    # Calculate metrics
    final_equity = equity_curve[-1]
    total_return = (final_equity - initial_capital) / initial_capital

    # Max drawdown
    peak = initial_capital
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0

    # Display results
    print(f"\n{Colors.BOLD}BACKTEST RESULTS{Colors.END}")
    print(f"{Colors.CYAN}{'─'*50}{Colors.END}")
    color = Colors.GREEN if total_return > 0 else Colors.RED
    print(f"   Total Return: {color}{total_return:+.2%}{Colors.END}")
    print(f"   Final Equity: ${final_equity:,.2f}")
    print(f"   Max Drawdown: {max_dd:.2%}")
    print(f"   Total Trades: {total_trades}")
    print(f"   Win Rate: {win_rate:.1%}")
    print(f"   Winners: {wins}")
    print(f"   Losers: {losses}")

    # Show recent trades
    if trades:
        print(f"\n   Recent Trades:")
        for trade in trades[-5:]:
            trade_color = Colors.GREEN if trade['pnl'] > 0 else Colors.RED
            print(f"   {trade['direction']:5} | {trade['pnl_pips']:+6.1f} pips | "
                  f"{trade_color}${trade['pnl']:+.2f}{Colors.END} | {trade['exit_reason']}")

    print()


async def run_live(pair: str, capital: float):
    """Run live trading mode (paper trading)."""
    print(f"\n{Colors.BOLD}LIVE TRADING MODE: {pair}{Colors.END}")
    print(f"{Colors.RED}⚠️  THIS IS PAPER TRADING - NO REAL MONEY{Colors.END}")
    print(f"{Colors.CYAN}{'─'*50}{Colors.END}")
    print(f"   Capital: ${capital:,.2f}")
    print(f"   Strategy: Integrated News + Technical")
    print(f"   Risk Profile: Conservative")
    print()

    # Create components
    strategy = create_integrated_strategy(pair=pair, conservative=True)
    risk_manager = create_conservative_risk_manager(capital)

    from forex_system.services.enhanced_forex_signals import EnhancedForexSignals
    signals_service = EnhancedForexSignals()

    print(f"{Colors.YELLOW}Starting live monitoring...{Colors.END}")
    print(f"Press Ctrl+C to stop\n")

    try:
        while True:
            timestamp = datetime.utcnow().strftime('%H:%M:%S')

            # Get current signals
            combined = await signals_service.get_combined_signal(pair)

            # Get risk status
            risk_report = risk_manager.get_risk_report()

            # Display update
            signal_color = Colors.GREEN if combined['signal'] == 'BUY' else Colors.RED if combined['signal'] == 'SELL' else Colors.YELLOW

            print(f"[{timestamp}] {pair} | "
                  f"{signal_color}{combined['signal']:8}{Colors.END} | "
                  f"Strength: {combined['strength']:.2f} | "
                  f"Regime: {combined.get('regime', 'N/A')[:15]} | "
                  f"Equity: ${risk_report['equity']:,.0f}")

            # Show reasons for strong signals
            if combined['strength'] > 0.4:
                for reason in combined.get('explanations', [])[:2]:
                    print(f"         → {reason}")

            # Wait before next update
            await asyncio.sleep(60)  # Update every minute

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Stopped by user{Colors.END}")
    finally:
        await signals_service.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Unified Forex Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_integrated_system.py --mode analyze --pair EURUSD
  python run_integrated_system.py --mode backtest --pair GBPUSD --days 30
  python run_integrated_system.py --mode live --pair EURUSD --capital 10000
        """
    )

    parser.add_argument(
        '--mode', '-m',
        choices=['analyze', 'backtest', 'live'],
        default='analyze',
        help='Operating mode'
    )

    parser.add_argument(
        '--pair', '-p',
        default='EURUSD',
        help='Currency pair to trade'
    )

    parser.add_argument(
        '--days', '-d',
        type=int,
        default=30,
        help='Days to backtest (backtest mode)'
    )

    parser.add_argument(
        '--capital', '-c',
        type=float,
        default=10000.0,
        help='Initial capital'
    )

    args = parser.parse_args()

    print_banner()

    if args.mode == 'analyze':
        asyncio.run(run_analysis(args.pair))
    elif args.mode == 'backtest':
        asyncio.run(run_backtest(args.pair, args.days, args.capital))
    elif args.mode == 'live':
        asyncio.run(run_live(args.pair, args.capital))


if __name__ == '__main__':
    main()
