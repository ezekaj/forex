@echo off
echo ============================================================
echo LIVE TRADING SETUP - OANDA CONFIGURATION
echo ============================================================
echo.
echo IMPORTANT: You need OANDA account credentials first!
echo.
echo Enter your OANDA credentials:
echo.
set /p API_KEY="Enter OANDA API Key: "
set /p ACCOUNT_ID="Enter OANDA Account ID: "
echo.
echo Select trading mode:
echo 1. DEMO (Practice account - fake money)
echo 2. LIVE (Real account - REAL MONEY!)
echo.
set /p MODE="Enter choice (1 or 2): "

if "%MODE%"=="2" (
    set TRADING_MODE=true
    echo.
    echo WARNING: LIVE TRADING WITH REAL MONEY!
    echo Are you absolutely sure? Type YES to confirm:
    set /p CONFIRM="Confirmation: "
    if not "%CONFIRM%"=="YES" (
        echo Cancelled - staying in demo mode
        set TRADING_MODE=false
    )
) else (
    set TRADING_MODE=false
)

echo.
echo Writing configuration to BayloZzi\.env...

(
echo # OANDA Configuration
echo OANDA_API_KEY=%API_KEY%
echo OANDA_ACCOUNT_ID=%ACCOUNT_ID%
echo TRADING_ENABLED=%TRADING_MODE%
echo.
echo # Alpha Vantage API
echo ALPHAVANTAGE_API_KEY=KNF41ZTAUM44W2LN
echo.
echo # Risk Settings
echo MAX_RISK_PER_TRADE=0.02
echo MAX_DAILY_LOSS=0.10
echo TARGET_DAILY_PROFIT=0.05
) > BayloZzi\.env

echo.
echo Configuration saved!
echo.
echo To start LIVE trading 24/7, run:
echo   python start_live_24_7.py
echo.
pause