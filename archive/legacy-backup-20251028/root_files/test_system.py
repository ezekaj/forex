#!/usr/bin/env python
"""
EASY TEST SCRIPT - Test the Enhanced Forex Trading System
Run this to verify everything is working correctly
"""

import sys
import os
import time
from datetime import datetime

# Add paths for imports
sys.path.append('03_CORE_ENGINE')
sys.path.append('01_LIVE_TRADING')
sys.path.append('02_ELITE_SYSTEMS')
sys.path.append('BayloZzi')

print("\n" + "="*70)
print("FOREX TRADING SYSTEM - ENHANCED VERSION TEST")
print("="*70)
print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

def test_components():
    """Test all 5 valuable components"""
    print("\n[TEST 1] Testing Individual Components...")
    print("-"*50)
    
    try:
        # Test imports
        from smart_position_sizer import SmartPositionSizer
        print("[OK] Smart Position Sizer imported")
        
        from win_rate_optimizer import WinRateOptimizer
        print("[OK] Win Rate Optimizer imported")
        
        from market_timing_system import MarketTimingSystem
        print("[OK] Market Timing System imported")
        
        from advanced_features import AdvancedFeatureEngineering
        print("[OK] Advanced Feature Engineering imported")
        
        from performance_analytics import PerformanceAnalytics
        print("[OK] Performance Analytics imported")
        
        # Quick functionality test
        sizer = SmartPositionSizer()
        optimizer = WinRateOptimizer()
        timing = MarketTimingSystem()
        
        # Test market timing
        status = timing.get_current_market_status()
        print(f"\n[MARKET STATUS]")
        print(f"  Liquidity Score: {status['liquidity_score']:.2f}")
        print(f"  Open Sessions: {', '.join(status['open_sessions']) if status['open_sessions'] else 'None'}")
        print(f"  Recommendation: {status['recommendation']}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Component test failed: {e}")
        return False

def test_real_trader():
    """Test REAL_TRADER system"""
    print("\n[TEST 2] Testing REAL TRADER System...")
    print("-"*50)
    
    try:
        from REAL_TRADER import AutomaticForexTrader
        
        # Create trader instance (demo mode)
        trader = AutomaticForexTrader(initial_capital=10.0)
        
        if trader.broker.connected:
            print("[OK] Broker connection established (Demo/Live)")
        else:
            print("[INFO] Broker not connected (simulation mode)")
        
        print("[OK] REAL TRADER system initialized with all components")
        return True
        
    except Exception as e:
        print(f"[ERROR] REAL TRADER test failed: {e}")
        return False

def test_elite_system():
    """Test Elite AI system"""
    print("\n[TEST 3] Testing Elite AI System (KIMI K2)...")
    print("-"*50)
    
    try:
        from kimi_k2 import KimiK2TradingEngine
        
        # Create engine instance
        engine = KimiK2TradingEngine(capital=10.0)
        
        # Quick analysis
        analysis = engine.analyze_market('EURUSD')
        
        print(f"[OK] Elite AI initialized")
        print(f"[OK] Analysis completed: {analysis['consensus']['direction']} "
              f"({analysis['consensus']['confidence']:.1%} confidence)")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Elite AI test failed: {e}")
        return False

def test_integration():
    """Test complete integration"""
    print("\n[TEST 4] Testing Complete Integration...")
    print("-"*50)
    
    try:
        # Import all major components
        from smart_position_sizer import SmartPositionSizer
        from win_rate_optimizer import WinRateOptimizer
        from market_timing_system import MarketTimingSystem
        
        # Create instances
        sizer = SmartPositionSizer()
        optimizer = WinRateOptimizer()
        timing = MarketTimingSystem()
        
        # Simulate trading decision flow
        print("[SIMULATION] Complete trading decision flow:")
        
        # 1. Check market timing
        should_trade, reason = timing.should_trade_now('EURUSD')
        print(f"  1. Market Timing: {'TRADE' if should_trade else 'WAIT'}")
        
        # 2. Get optimized decision
        decision = optimizer.get_trade_decision(
            signal_confidence=0.70,
            market_volatility=0.01
        )
        print(f"  2. Win Optimizer: {'APPROVED' if decision['trade'] else 'REJECTED'}")
        
        # 3. Calculate position size
        if decision['trade']:
            position = sizer.calculate_position_size(
                account_equity=1000,
                entry_price=1.1850,
                stop_loss=1.1830,
                take_profit=1.1890,
                confidence=0.70
            )
            print(f"  3. Position Size: {position['units']} units")
            print(f"  4. Risk: {position['risk_percentage']:.2f}% of capital")
        
        print("\n[OK] Integration test completed successfully")
        return True
        
    except Exception as e:
        print(f"[ERROR] Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    tests = [
        ("Component Tests", test_components),
        ("REAL TRADER Test", test_real_trader),
        ("Elite AI Test", test_elite_system),
        ("Integration Test", test_integration)
    ]
    
    results = []
    for name, test_func in tests:
        success = test_func()
        results.append((name, success))
        time.sleep(1)  # Small delay between tests
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, success in results:
        status = "[PASSED]" if success else "[FAILED]"
        print(f"{status} {name}")
        if not success:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n[SUCCESS] All systems are working correctly!")
        print("\nYou can now:")
        print("1. Run 'python main.py' and choose option 2 for REAL TRADER")
        print("2. Run 'python main.py' and choose option 3 for Elite AI")
        print("3. Run 'python 01_LIVE_TRADING/REAL_TRADER.py' directly")
        print("4. Run 'python 02_ELITE_SYSTEMS/kimi_k2.py --mode trade'")
    else:
        print("\n[WARNING] Some tests failed. Please check the errors above.")
        print("Common fixes:")
        print("1. Make sure you're in the forex directory")
        print("2. Check that all files were created properly")
        print("3. Verify Python has all required packages installed")

if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")