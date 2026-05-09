@echo off
REM ============================================================
REM Gait Analysis System - Windows Quick Start Script
REM ============================================================

echo ============================================================
echo   Dual-Camera Treadmill Gait Analysis System
echo   Quick Setup and Run Script
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9-3.11.
    pause
    exit /b 1
)

REM Check if venv exists
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created.
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install dependencies
echo.
echo [INFO] Installing dependencies...
pip install -r requirements.txt --quiet

REM Check environment
echo.
echo [INFO] Checking environment...
python check_env.py

echo.
echo ============================================================
echo   Setup complete!
echo.
echo   Usage:
echo     python main.py -c config.yaml -v1 camera1.mkv -v2 camera2.mkv
echo.
echo   For help:
echo     python main.py --help
echo ============================================================
pause
