@echo off
echo ============================================================
echo         FOREX TRADING SYSTEM - MASTER LAUNCHER
echo ============================================================
echo.
echo Select Trading Mode:
echo.
echo 1. TURBO MODE (50%% daily target - HIGH RISK)
echo 2. SAFE MODE (10%% daily target - LOW RISK)  
echo 3. SCALP MODE (30%% daily target - MEDIUM RISK)
echo 4. ANALYZE MARKET (No trading)
echo 5. CHECK STATUS
echo 6. INTERACTIVE MENU
echo.
set /p choice="Enter choice (1-6): "

if "%choice%"=="1" (
    echo.
    echo Starting TURBO MODE...
    python forex.py turbo -c 10
) else if "%choice%"=="2" (
    echo.
    echo Starting SAFE MODE...
    python forex.py safe -c 10
) else if "%choice%"=="3" (
    echo.
    echo Starting SCALP MODE...
    python forex.py scalp -c 10
) else if "%choice%"=="4" (
    echo.
    echo Analyzing market...
    python forex.py analyze
) else if "%choice%"=="5" (
    echo.
    echo Checking status...
    python forex.py status
) else if "%choice%"=="6" (
    echo.
    echo Starting interactive menu...
    echo Note: Use number keys to navigate
    python forex.py
) else (
    echo Invalid choice!
)

echo.
pause