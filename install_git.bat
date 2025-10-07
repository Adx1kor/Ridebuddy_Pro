@echo off
echo.
echo ============================================================
echo          Git Installation for RideBuddy Pro Upload
echo ============================================================
echo.

REM Check if Git is already installed
git --version >nul 2>&1
if %errorlevel% equ 0 (
    echo SUCCESS: Git is already installed!
    git --version
    echo.
    echo You can now run: upload_to_adx1kor_repo.bat
    pause
    exit /b 0
)

echo Git is not currently installed. Let's install it!
echo.

REM Try to install via winget (Windows 11/10)
echo [Option 1] Trying winget installation...
winget install Git.Git >nul 2>&1
if %errorlevel% equ 0 (
    echo SUCCESS: Git installed via winget
    echo Please restart PowerShell and run upload_to_adx1kor_repo.bat
    pause
    exit /b 0
)

REM Try to install via chocolatey
echo [Option 2] Trying Chocolatey installation...
choco install git -y >nul 2>&1
if %errorlevel% equ 0 (
    echo SUCCESS: Git installed via Chocolatey
    echo Please restart PowerShell and run upload_to_adx1kor_repo.bat
    pause
    exit /b 0
)

REM Manual installation
echo [Option 3] Manual installation required
echo.
echo Automated installation not available.
echo Please install Git manually:
echo.
echo 1. Open browser and go to: https://git-scm.com/download/windows
echo 2. Download Git for Windows
echo 3. Run the installer with default settings
echo 4. Restart PowerShell
echo 5. Run upload_to_adx1kor_repo.bat
echo.
echo Opening download page in 3 seconds...
timeout /t 3 /nobreak >nul

REM Try to open browser
start https://git-scm.com/download/windows

echo.
echo Download page opened in browser.
echo After installation, run: upload_to_adx1kor_repo.bat
echo.
pause