@echo off
echo ============================================================
echo OANDA PRACTICE ACCOUNT SETUP
echo ============================================================
echo.
echo This will configure your system for PRACTICE trading
echo (Using fake money - perfect for testing!)
echo.
echo You need your OANDA Practice Account credentials:
echo 1. API Token (from OANDA website)
echo 2. Account ID (from OANDA website)
echo.
set /p API_KEY="Paste your OANDA API Token: "
set /p ACCOUNT_ID="Enter your Account ID (like 101-001-1234567-001): "

echo.
echo Configuring for PRACTICE mode...

(
echo # OANDA Practice Account Configuration
echo OANDA_API_KEY=%API_KEY%
echo OANDA_ACCOUNT_ID=%ACCOUNT_ID%
echo TRADING_ENABLED=false
echo.
echo # Alpha Vantage API for market data
echo ALPHAVANTAGE_API_KEY=KNF41ZTAUM44W2LN
echo.
echo # Risk Settings for Practice
echo MAX_RISK_PER_TRADE=0.02
echo MAX_DAILY_LOSS=0.10
echo TARGET_DAILY_PROFIT=0.05
echo STARTING_CAPITAL=100000
) > BayloZzi\.env

echo.
echo ============================================================
echo SUCCESS! Practice account configured!
echo ============================================================
echo.
echo You have $100,000 practice money to trade with!
echo.
echo To start practice trading:
echo   python start_practice.py
echo.
echo Or run the main menu:
echo   python main.py
echo.
pause