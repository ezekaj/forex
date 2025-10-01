@echo off
cls
echo ============================================================
echo           FOREX TRADING SYSTEM - MAIN LAUNCHER
echo ============================================================
echo.
echo Choose what you want to do:
echo.
echo   [1] TEST SYSTEM - See real market data and signals
echo   [2] REAL TRADER - Start automatic trading (needs OANDA)
echo   [3] ELITE AI - Run advanced AI trading (kimi_k2)
echo   [4] TURBO MODE - Aggressive 50%% daily target
echo   [5] SAFE MODE - Conservative 10%% daily target
echo   [6] VIEW GUIDE - Open system documentation
echo   [7] SETUP HELP - Show setup instructions
echo.
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" (
    echo.
    echo Starting system test...
    python 07_TESTING\QUICK_TEST.py
) else if "%choice%"=="2" (
    echo.
    echo Starting real trader...
    python 01_LIVE_TRADING\REAL_TRADER.py --hours 1
) else if "%choice%"=="3" (
    echo.
    echo Starting Elite AI system...
    python 02_ELITE_SYSTEMS\kimi_k2.py --mode trade --hours 1
) else if "%choice%"=="4" (
    echo.
    echo Starting Turbo mode...
    python 01_LIVE_TRADING\forex.py turbo
) else if "%choice%"=="5" (
    echo.
    echo Starting Safe mode...
    python 01_LIVE_TRADING\forex.py safe
) else if "%choice%"=="6" (
    echo.
    echo Opening system guide...
    notepad SYSTEM_INDEX.md
) else if "%choice%"=="7" (
    echo.
    echo ============================================================
    echo                    SETUP INSTRUCTIONS
    echo ============================================================
    echo.
    echo 1. GET OANDA ACCOUNT:
    echo    - Go to: https://www.oanda.com
    echo    - Create Practice Account (free)
    echo    - Get API key from settings
    echo.
    echo 2. UPDATE CONFIGURATION:
    echo    - Open: BayloZzi\.env
    echo    - Add your OANDA_API_KEY
    echo    - Add your OANDA_ACCOUNT_ID
    echo.
    echo 3. START TRADING:
    echo    - Run this launcher again
    echo    - Choose option 2 for real trading
    echo.
    pause
) else (
    echo Invalid choice!
)

echo.
pause