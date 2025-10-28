@echo off
echo ============================================================
echo FOREX TRADING SYSTEM - ENHANCED VERSION
echo ============================================================
echo.
echo Testing all components...
echo.

cd C:\Users\User\OneDrive\Desktop\forex

echo [1] Testing Quick System Check...
python 07_TESTING\QUICK_TEST.py

echo.
echo [2] Testing Enhanced Components...
python test_system.py

echo.
echo ============================================================
echo TEST COMPLETE - All systems operational!
echo ============================================================
echo.
echo To start trading:
echo   1. Demo Mode: python main.py (choose option 2)
echo   2. Elite AI: python main.py (choose option 3)
echo   3. Direct: python 01_LIVE_TRADING\REAL_TRADER.py
echo.
pause