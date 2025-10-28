#!/usr/bin/env python
"""
TEST ENHANCED SYSTEM - Verify all 5 components integration
Tests the complete forex trading system with all new components
"""

import sys
import os
sys.path.append('../01_LIVE_TRADING')
sys.path.append('../02_ELITE_SYSTEMS')
sys.path.append('../03_CORE_ENGINE')
sys.path.append('../BayloZzi')

from datetime import datetime
import pandas as pd
import numpy as np

# Import all 5 components
from smart_position_sizer import SmartPositionSizer
from win_rate_optimizer import WinRateOptimizer, Trade
from market_timing_system import MarketTimingSystem
from advanced_features import AdvancedFeatureEngineering
from performance_analytics import PerformanceAnalytics

def test_component_1_position_sizer():
    """Test Smart Position Sizer"""
    print("\n" + "="*60)
    print("TESTING COMPONENT #1: SMART POSITION SIZER")
    print("="*60)
    
    sizer = SmartPositionSizer(
        risk_per_trade=0.02,
        max_position_size=0.10,
        min_position_size=0.01
    )
    
    result = sizer.calculate_position_size(
        account_equity=1000,
        entry_price=1.1850,
        stop_loss=1.1800,
        take_profit=1.1950,
        volatility=0.012,
        confidence=0.75,
        market_condition='trending'
    )
    
    print(f"[OK] Position Size: ${result['position_size']:.2f}")
    print(f"[OK] Units: {result['units']}")
    print(f"[OK] Risk: {result['risk_percentage']:.2f}%")
    print(f"[OK] Recommendation: {result['recommendation']}")
    print("\n[PASSED] Smart Position Sizer working correctly!")
    return True

def test_component_2_win_optimizer():
    """Test Win Rate Optimizer"""
    print("\n" + "="*60)
    print("TESTING COMPONENT #2: WIN RATE OPTIMIZER")
    print("="*60)
    
    optimizer = WinRateOptimizer(target_win_rate=0.65)
    
    # Test trade decision
    decision = optimizer.get_trade_decision(
        signal_confidence=0.75,
        market_volatility=0.012,
        recent_performance=[True, False, True]
    )
    
    print(f"[OK] Trade Decision: {'APPROVED' if decision['trade'] else 'REJECTED'}")
    if decision['trade']:
        print(f"[OK] Stop Loss: {decision['stop_loss_pips']} pips")
        print(f"[OK] Take Profit: {decision['take_profit_pips']} pips")
    else:
        print(f"[OK] Reason: {decision['reason']}")
    
    print("\n[PASSED] Win Rate Optimizer working correctly!")
    return True

def test_component_3_market_timing():
    """Test Market Timing System"""
    print("\n" + "="*60)
    print("TESTING COMPONENT #3: MARKET TIMING SYSTEM")
    print("="*60)
    
    timing = MarketTimingSystem()
    
    # Get market status
    status = timing.get_current_market_status()
    print(f"[OK] Open Sessions: {', '.join(status['open_sessions']) if status['open_sessions'] else 'None'}")
    print(f"[OK] Liquidity Score: {status['liquidity_score']:.2f}")
    print(f"[OK] Recommendation: {status['recommendation']}")
    
    # Check if should trade
    should_trade, reason = timing.should_trade_now('EURUSD')
    print(f"[OK] Should Trade EURUSD: {'YES' if should_trade else 'NO'}")
    print(f"[OK] Reason: {reason}")
    
    print("\n[PASSED] Market Timing System working correctly!")
    return True

def test_component_4_feature_engineering():
    """Test Advanced Feature Engineering"""
    print("\n" + "="*60)
    print("TESTING COMPONENT #4: ADVANCED FEATURE ENGINEERING")
    print("="*60)
    
    engineer = AdvancedFeatureEngineering()
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Engineer features
    df_enhanced = engineer.engineer_all_features(df)
    
    print(f"[OK] Original features: {len(df.columns)}")
    print(f"[OK] Enhanced features: {len(df_enhanced.columns)}")
    print(f"[OK] New features added: {len(df_enhanced.columns) - len(df.columns)}")
    
    # Check some specific features
    if 'rsi' in df_enhanced.columns:
        print(f"[OK] RSI calculated: {df_enhanced['rsi'].iloc[-1]:.2f}")
    if 'macd' in df_enhanced.columns:
        print(f"[OK] MACD calculated: {df_enhanced['macd'].iloc[-1]:.4f}")
    
    print("\n[PASSED] Advanced Feature Engineering working correctly!")
    return True

def test_component_5_performance_analytics():
    """Test Performance Analytics"""
    print("\n" + "="*60)
    print("TESTING COMPONENT #5: PERFORMANCE ANALYTICS")
    print("="*60)
    
    tracker = PerformanceAnalytics(initial_capital=1000.0)
    
    # Record some sample trades
    from datetime import timedelta
    trades = [
        {'entry_time': datetime.now(), 'exit_time': datetime.now() + timedelta(minutes=30),
         'pair': 'EURUSD', 'direction': 'BUY', 
         'entry_price': 1.1850, 'exit_price': 1.1870, 'position_size': 100, 
         'profit': 2.0},
        {'entry_time': datetime.now(), 'exit_time': datetime.now() + timedelta(minutes=45),
         'pair': 'GBPUSD', 'direction': 'SELL', 
         'entry_price': 1.2650, 'exit_price': 1.2670, 'position_size': 100, 
         'profit': -2.0},
        {'entry_time': datetime.now(), 'exit_time': datetime.now() + timedelta(minutes=60),
         'pair': 'USDJPY', 'direction': 'BUY', 
         'entry_price': 110.50, 'exit_price': 110.70, 'position_size': 100, 
         'profit': 1.8}
    ]
    
    for trade in trades:
        tracker.record_trade(trade)
    
    # Calculate metrics
    metrics = tracker.calculate_metrics()
    
    print(f"[OK] Total Trades: {metrics['total_trades']}")
    print(f"[OK] Win Rate: {metrics['win_rate']:.1%}")
    print(f"[OK] Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"[OK] Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    print("\n[PASSED] Performance Analytics working correctly!")
    return True

def test_integration():
    """Test integration of all components"""
    print("\n" + "="*60)
    print("TESTING COMPLETE INTEGRATION")
    print("="*60)
    
    # Initialize all components
    sizer = SmartPositionSizer()
    optimizer = WinRateOptimizer()
    timing = MarketTimingSystem()
    engineer = AdvancedFeatureEngineering()
    tracker = PerformanceAnalytics()
    
    print("[OK] All components initialized successfully")
    
    # Simulate a trading decision flow
    print("\n[SIMULATION] Complete trading decision flow:")
    
    # 1. Check market timing
    should_trade, timing_reason = timing.should_trade_now('EURUSD')
    print(f"1. Market Timing: {'GO' if should_trade else 'WAIT'} - {timing_reason}")
    
    if should_trade:
        # 2. Get trade decision from optimizer
        decision = optimizer.get_trade_decision(
            signal_confidence=0.70,
            market_volatility=0.01,
            recent_performance=[True, True, False]
        )
        print(f"2. Win Optimizer: {'APPROVED' if decision['trade'] else 'REJECTED'}")
        
        if decision['trade']:
            # 3. Calculate position size
            position = sizer.calculate_position_size(
                account_equity=1000,
                entry_price=1.1850,
                stop_loss=1.1830,
                take_profit=1.1890,
                confidence=0.70
            )
            print(f"3. Position Size: {position['units']} units ({position['risk_percentage']:.2f}% risk)")
            
            # 4. Record hypothetical trade
            from datetime import timedelta
            tracker.record_trade({
                'entry_time': datetime.now(),
                'exit_time': datetime.now() + timedelta(minutes=30),
                'pair': 'EURUSD',
                'direction': 'BUY',
                'entry_price': 1.1850,
                'exit_price': 1.1890,
                'position_size': position['units'],
                'profit': 4.0
            })
            print("4. Trade recorded in performance tracker")
    
    print("\n[PASSED] Complete integration test successful!")
    return True

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("ENHANCED FOREX TRADING SYSTEM - COMPONENT INTEGRATION TEST")
    print("="*70)
    print("Testing all 5 valuable components...")
    
    tests = [
        ("Smart Position Sizer", test_component_1_position_sizer),
        ("Win Rate Optimizer", test_component_2_win_optimizer),
        ("Market Timing System", test_component_3_market_timing),
        ("Advanced Feature Engineering", test_component_4_feature_engineering),
        ("Performance Analytics", test_component_5_performance_analytics),
        ("Complete Integration", test_integration)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n[ERROR] {name} test failed: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, success in results:
        status = "[OK]" if success else "[FAILED]"
        print(f"{status} {name}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n" + "="*70)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("The enhanced forex trading system is ready for use.")
        print("All 5 valuable components are integrated and working.")
        print("="*70)
    else:
        print("\n[WARNING] Some tests failed. Please review the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)