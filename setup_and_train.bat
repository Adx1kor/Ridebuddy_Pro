@echo off
echo ==========================================
echo RideBuddy Setup and Training Script
echo ==========================================

echo.
echo Step 1: Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Please install Python from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    echo After installing Python, run this script again.
    pause
    exit /b 1
)

echo Python found!
python --version

echo.
echo Step 2: Organizing dataset...
python organize_dataset.py
if %errorlevel% neq 0 (
    echo Dataset organization failed!
    pause
    exit /b 1
)

echo.
echo Step 3: Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Dependency installation failed!
    pause
    exit /b 1
)

echo.
echo Step 4: Testing model creation...
python -c "from src.models.ridebuddy_model import create_lightweight_model; model = create_lightweight_model(); print(f'Model created successfully with {model.count_parameters():,} parameters')"
if %errorlevel% neq 0 (
    echo Model creation test failed!
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Setup Complete! Ready for training.
echo ==========================================
echo.
echo To start training, run:
echo python src/train.py --config configs/lightweight_model.yaml --data_dir data/organized --output_dir models
echo.
echo To run inference on webcam:
echo python src/inference.py --model models/best_model.pth --webcam
echo.
pause
