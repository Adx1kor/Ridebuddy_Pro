@echo off
REM RideBuddy Pro Vehicle Launcher (Windows Batch)
REM Double-click this file to launch RideBuddy in vehicle mode

echo.
echo ========================================
echo  RideBuddy Pro - Vehicle Launcher
echo  Professional Driver Monitoring
echo ========================================
echo.

REM Check if Python is available
py --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if main application exists
if not exist "ridebuddy_optimized_gui.py" (
    echo ERROR: ridebuddy_optimized_gui.py not found!
    echo Please ensure you're in the correct directory
    pause
    exit /b 1
)

echo Starting RideBuddy Pro for Vehicle Deployment...
echo.

REM Launch the Python vehicle launcher
py vehicle_launcher.py

echo.
echo Vehicle session completed.
pause