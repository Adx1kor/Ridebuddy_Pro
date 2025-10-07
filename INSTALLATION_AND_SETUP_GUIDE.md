# RideBuddy Pro v2.1.0 - Installation & Setup Guide

## 🚀 Quick Start

### Prerequisites Check
Before installation, ensure your system meets the minimum requirements:
- **Operating System**: Windows 10/11 or Linux Ubuntu 18.04+
- **Python**: Version 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB available space
- **Camera**: USB webcam or integrated camera

---

## 📋 Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Dependency Installation](#dependency-installation)
4. [Configuration Setup](#configuration-setup)
5. [First Run](#first-run)
6. [Vehicle Deployment](#vehicle-deployment)
7. [Troubleshooting](#troubleshooting)
8. [Verification](#verification)

---

## 💻 System Requirements

### Minimum Requirements
```
Hardware:
├── CPU: Intel i3 / AMD Ryzen 3 (2.0GHz+)
├── RAM: 4GB
├── GPU: Not required (CPU inference)
├── Storage: 2GB available space
├── Camera: USB 2.0+ webcam
└── Network: Internet for initial setup

Software:
├── OS: Windows 10/11 (64-bit) or Linux Ubuntu 18.04+
├── Python: 3.8+ (3.9-3.11 recommended)
├── Drivers: Camera drivers installed
└── Permissions: Admin rights for installation
```

### Recommended Requirements
```
Hardware:
├── CPU: Intel i5 / AMD Ryzen 5 (3.0GHz+)
├── RAM: 8GB
├── GPU: Integrated graphics sufficient
├── Storage: 5GB available space
├── Camera: HD webcam (720p+)
└── Network: Stable internet connection

Software:
├── OS: Windows 11 or Linux Ubuntu 20.04+
├── Python: 3.9-3.11
├── Updates: Latest system updates
└── Antivirus: Configured to allow RideBuddy
```

### Vehicle-Specific Requirements
```
Automotive Environment:
├── Power: 12V DC adapter or battery pack
├── Mounting: Dashboard or windshield mount
├── Vibration: Shock-resistant hardware
├── Temperature: -10°C to 60°C operating range
└── Connectivity: Optional 4G/5G for fleet mode
```

---

## 📥 Installation Methods

### Method 1: Automated Installation (Recommended)

#### Windows Installation
```powershell
# 1. Download the installation package
git clone https://github.com/your-repo/ridebuddy-pro.git
cd ridebuddy-pro

# 2. Run the automated installer
.\install_dependencies.py

# 3. Execute the setup script
.\setup_and_train.bat
```

#### Linux Installation
```bash
# 1. Clone the repository
git clone https://github.com/your-repo/ridebuddy-pro.git
cd ridebuddy-pro

# 2. Make scripts executable
chmod +x install_dependencies.py
chmod +x setup_and_train.sh

# 3. Run installation
python3 install_dependencies.py
./setup_and_train.sh
```

### Method 2: Manual Installation

#### Step 1: Python Environment Setup
```bash
# Check Python version
python --version  # Should be 3.8+

# Create virtual environment (recommended)
python -m venv ridebuddy_env

# Activate virtual environment
# Windows:
ridebuddy_env\Scripts\activate
# Linux/Mac:
source ridebuddy_env/bin/activate
```

#### Step 2: Install Core Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version for edge deployment)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install OpenCV
pip install opencv-python

# Install additional requirements
pip install -r requirements.txt
```

#### Step 3: Install Optional Components
```bash
# For enhanced performance monitoring
pip install psutil

# For model optimization (optional)
pip install onnx onnxruntime

# For advanced data processing
pip install albumentations
```

---

## 📦 Dependency Installation

### Core Dependencies (Required)

#### requirements.txt
```txt
# Core ML Framework
torch>=2.0.0
torchvision>=0.15.0

# Computer Vision
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=9.5.0

# Data Processing
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# YOLO Detection
ultralytics>=8.0.0

# Progress Tracking
tqdm>=4.65.0

# Configuration
pyyaml>=6.0
psutil>=5.9.0

# GUI (Built-in with Python)
# tkinter - included with Python installation
```

#### Minimal Requirements (requirements-minimal.txt)
```txt
# Lightweight installation for resource-constrained environments
torch>=2.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=9.5.0
```

### Platform-Specific Installation

#### Windows-Specific Setup
```powershell
# Install Microsoft Visual C++ Redistributable (if needed)
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

# Install Windows-specific packages
pip install pywin32  # For Windows services integration

# Set up Windows camera permissions
# Go to Settings > Privacy > Camera > Allow desktop apps to access camera
```

#### Linux-Specific Setup
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install python3-pip python3-venv python3-tk
sudo apt install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# Install camera support
sudo apt install v4l-utils  # Video4Linux utilities
sudo apt install cheese     # Test camera functionality

# Add user to video group (for camera access)
sudo usermod -a -G video $USER
# Log out and log back in for changes to take effect
```

#### MacOS-Specific Setup
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python via Homebrew
brew install python@3.9

# Install system dependencies
brew install pkg-config
brew install cmake
```

---

## ⚙️ Configuration Setup

### Initial Configuration Files

#### 1. Main Configuration (ridebuddy_config.ini)
```ini
[camera]
# Camera settings
camera_index = 0
frame_width = 640
frame_height = 480
max_fps = 30
auto_exposure = true

[ai]
# AI model settings
model_path = ./trained_models/enhanced_driver_net.pth
alert_sensitivity = 0.7
confidence_threshold = 0.75
batch_processing = false

[alerts]
# Alert configuration
audio_alerts = true
visual_alerts = true
alert_cooldown = 5.0
severity_levels = ["Low", "Medium", "High"]

[gui]
# GUI settings
theme = dark
fullscreen = false
auto_save_settings = true
alert_history_limit = 100

[performance]
# Performance tuning
max_memory_mb = 512
frame_drop_threshold = 0.8
cpu_limit_percent = 80
reconnect_attempts = 5

[logging]
# Logging configuration
log_level = INFO
log_rotation = true
max_log_size_mb = 10
backup_count = 5

[data]
# Data management
auto_save_results = true
data_retention_days = 30
database_path = ./ridebuddy_data.db
export_format = json
```

#### 2. Vehicle Configuration (vehicle_config.json)
```json
{
  "vehicle_info": {
    "vehicle_id": "VEHICLE_001",
    "fleet_id": "FLEET_A",
    "make": "Generic",
    "model": "Vehicle",
    "year": 2024
  },
  "deployment_settings": {
    "vehicle_mode": false,
    "fleet_mode": false,
    "kiosk_mode": false,
    "auto_start": false,
    "power_save_mode": false
  },
  "alert_thresholds": {
    "drowsiness": 0.75,
    "distraction": 0.8,
    "phone_usage": 0.7,
    "seatbelt": 0.8
  },
  "camera_settings": {
    "primary_camera": 0,
    "backup_camera": 1,
    "resolution": "640x480",
    "auto_switch": true
  },
  "connectivity": {
    "fleet_server": "https://fleet.ridebuddy.com",
    "api_key": "",
    "sync_interval": 300,
    "offline_mode": true
  }
}
```

### Environment Variables Setup

#### Windows Environment Variables
```powershell
# Set via PowerShell (temporary)
$env:RIDEBUDDY_VEHICLE_MODE = "false"
$env:RIDEBUDDY_LOG_LEVEL = "INFO"
$env:RIDEBUDDY_CONFIG_PATH = "./ridebuddy_config.ini"

# Set permanently via System Properties
# Computer > Properties > Advanced System Settings > Environment Variables
```

#### Linux/Mac Environment Variables
```bash
# Add to ~/.bashrc or ~/.zshrc
export RIDEBUDDY_VEHICLE_MODE=false
export RIDEBUDDY_LOG_LEVEL=INFO
export RIDEBUDDY_CONFIG_PATH=./ridebuddy_config.ini

# Reload configuration
source ~/.bashrc
```

---

## 🏃‍♂️ First Run

### Pre-Flight Check

#### System Validation Script
```bash
# Run system validation
python system_validation.py

# Expected output:
✅ Python version: 3.9.7 (Compatible)
✅ Dependencies: All required packages installed
✅ Camera: Device found at index 0
✅ Model: Enhanced driver model loaded successfully
✅ Configuration: Valid configuration files found
✅ Permissions: Camera and file system access granted
```

#### Camera Diagnostic Test
```bash
# Test camera functionality
python camera_diagnostics.py

# Expected output:
🎥 Camera Diagnostic Report
├── Camera Index 0: ✅ Available (640x480 @ 30fps)
├── Camera Index 1: ❌ Not available
├── Resolution Test: ✅ Supports 640x480, 1280x720
├── Frame Rate Test: ✅ Stable at 30fps
└── Auto-exposure: ✅ Working correctly
```

### Application Launch

#### Standard Launch
```bash
# Navigate to project directory
cd /path/to/ridebuddy-pro

# Launch application
python ridebuddy_optimized_gui.py
```

#### Vehicle Mode Launch
```bash
# Set vehicle mode
set RIDEBUDDY_VEHICLE_MODE=true

# Launch with vehicle configuration
python ridebuddy_optimized_gui.py --config vehicle_config.json
```

#### Debug Mode Launch
```bash
# Launch with verbose logging
python ridebuddy_optimized_gui.py --debug --log-level DEBUG
```

### Initial Setup Wizard

When launching for the first time, the application will guide you through:

1. **Camera Selection**: Choose primary camera device
2. **AI Model Loading**: Verify model file integrity
3. **Sensitivity Calibration**: Adjust alert thresholds
4. **Audio Test**: Verify alert sound functionality
5. **Performance Baseline**: Establish performance metrics

---

## 🚛 Vehicle Deployment

### Automotive Installation

#### Hardware Setup
```
Installation Steps:
1. Mount camera securely (dashboard or A-pillar)
2. Connect to vehicle power (12V adapter)
3. Install processing unit (laptop/mini-PC)
4. Configure display (optional dashboard screen)
5. Test in various lighting conditions
```

#### Software Configuration
```bash
# Enable vehicle mode
echo "RIDEBUDDY_VEHICLE_MODE=true" >> .env

# Configure for automotive use
python vehicle_deployment_guide.py --setup

# Set auto-start (Windows)
# Add to Windows Startup folder:
# C:\Users\{username}\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup
```

#### Fleet Management Setup
```json
{
  "fleet_configuration": {
    "central_server": "https://fleet.company.com",
    "vehicle_registration": true,
    "real_time_sync": true,
    "offline_capability": true,
    "data_encryption": true
  }
}
```

### Batch Installation Scripts

#### Windows Batch Setup (setup_vehicle.bat)
```batch
@echo off
echo Setting up RideBuddy for vehicle deployment...

REM Set environment variables
setx RIDEBUDDY_VEHICLE_MODE "true"
setx RIDEBUDDY_AUTO_START "true"

REM Install as Windows service (optional)
python install_service.py

REM Create desktop shortcut
python create_shortcuts.py

echo Vehicle setup complete!
pause
```

#### Linux Service Setup
```bash
#!/bin/bash
# Create systemd service for auto-start

sudo tee /etc/systemd/system/ridebuddy.service > /dev/null <<EOF
[Unit]
Description=RideBuddy Pro Driver Monitoring
After=multi-user.target

[Service]
Type=simple
User=ridebuddy
Environment=RIDEBUDDY_VEHICLE_MODE=true
ExecStart=/usr/bin/python3 /opt/ridebuddy/ridebuddy_optimized_gui.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable ridebuddy.service
sudo systemctl start ridebuddy.service
```

---

## 🔧 Troubleshooting

### Common Installation Issues

#### 1. Python Version Conflicts
```bash
# Problem: Multiple Python versions
# Solution: Use specific Python version
python3.9 -m venv ridebuddy_env
# or
py -3.9 -m venv ridebuddy_env  # Windows with Python Launcher
```

#### 2. PyTorch Installation Issues
```bash
# Problem: CUDA version conflicts or large downloads
# Solution: Use CPU-only version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 3. OpenCV Import Errors
```bash
# Problem: OpenCV not importing
# Solutions:
pip uninstall opencv-python opencv-python-headless
pip install opencv-python==4.8.1.78

# For headless servers:
pip install opencv-python-headless
```

#### 4. Camera Access Issues
```bash
# Windows: Check camera permissions
# Settings > Privacy & Security > Camera

# Linux: Add user to video group
sudo usermod -a -G video $USER

# Test camera independently
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera:', cap.isOpened())"
```

#### 5. Permission Errors
```bash
# Windows: Run as Administrator
# Right-click PowerShell/Command Prompt > "Run as Administrator"

# Linux: Check file permissions
chmod +x *.py
sudo chown -R $USER:$USER /path/to/ridebuddy
```

### Performance Optimization

#### Memory Issues
```ini
# Reduce memory usage in ridebuddy_config.ini
[performance]
max_memory_mb = 256        # Reduce from 512
frame_drop_threshold = 0.9  # More aggressive frame dropping
cpu_limit_percent = 60     # Lower CPU limit
```

#### CPU Performance
```bash
# Check system resources
python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%, RAM: {psutil.virtual_memory().percent}%')"

# Optimize for low-power devices
export RIDEBUDDY_POWER_SAVE=true
```

#### Disk Space Issues
```bash
# Clean up temporary files
python -c "
import os, glob
for f in glob.glob('*.log'): os.remove(f)
for f in glob.glob('__pycache__/*'): os.remove(f)
"
```

---

## ✅ Verification

### Installation Verification Checklist

#### ✅ Environment Check
- [ ] Python 3.8+ installed and accessible
- [ ] Virtual environment created and activated
- [ ] All required dependencies installed
- [ ] Camera device detected and accessible
- [ ] Configuration files present and valid

#### ✅ Functionality Check  
- [ ] Application launches without errors
- [ ] Camera feed displays correctly
- [ ] AI model loads successfully
- [ ] GUI elements render properly
- [ ] Alert system functions correctly

#### ✅ Performance Check
- [ ] Frame rate maintains 25-30 FPS
- [ ] Memory usage stays below 512MB
- [ ] CPU usage reasonable (<50% average)
- [ ] No memory leaks during extended use
- [ ] Alert response time <100ms

### Automated Verification Script

```python
# verification_script.py
import sys
import cv2
import torch
import tkinter as tk
from pathlib import Path

def verify_installation():
    """Comprehensive installation verification"""
    
    checks = []
    
    # Python version
    python_ok = sys.version_info >= (3, 8)
    checks.append(("Python 3.8+", python_ok))
    
    # Dependencies
    try:
        import numpy, PIL, matplotlib
        deps_ok = True
    except ImportError:
        deps_ok = False
    checks.append(("Dependencies", deps_ok))
    
    # Camera
    cap = cv2.VideoCapture(0)
    camera_ok = cap.isOpened()
    cap.release()
    checks.append(("Camera", camera_ok))
    
    # Model file
    model_exists = Path("./trained_models/enhanced_driver_net.pth").exists()
    checks.append(("AI Model", model_exists))
    
    # Configuration
    config_exists = Path("./ridebuddy_config.ini").exists()
    checks.append(("Configuration", config_exists))
    
    # GUI
    try:
        root = tk.Tk()
        root.destroy()
        gui_ok = True
    except:
        gui_ok = False
    checks.append(("GUI Framework", gui_ok))
    
    # Print results
    print("\n🔍 RideBuddy Installation Verification")
    print("=" * 50)
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:<20} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 All checks passed! RideBuddy is ready to use.")
    else:
        print("⚠️  Some checks failed. Please review installation steps.")
    
    return all_passed

if __name__ == "__main__":
    verify_installation()
```

### Performance Benchmark Script

```python
# benchmark_script.py
import time
import psutil
import cv2
from datetime import datetime

def benchmark_performance():
    """Benchmark system performance for RideBuddy"""
    
    print("\n⚡ RideBuddy Performance Benchmark")
    print("=" * 50)
    
    # System info
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    # Camera test
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        frame_times = []
        for i in range(100):  # Test 100 frames
            start_time = time.time()
            ret, frame = cap.read()
            if ret:
                # Simulate processing
                frame = cv2.resize(frame, (640, 480))
            frame_times.append(time.time() - start_time)
        
        cap.release()
        
        avg_frame_time = sum(frame_times) / len(frame_times)
        estimated_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        print(f"\n📹 Camera Performance:")
        print(f"Average frame time: {avg_frame_time*1000:.1f} ms")
        print(f"Estimated FPS: {estimated_fps:.1f}")
        
        if estimated_fps >= 25:
            print("✅ Camera performance: EXCELLENT")
        elif estimated_fps >= 15:
            print("⚠️  Camera performance: ADEQUATE")
        else:
            print("❌ Camera performance: NEEDS IMPROVEMENT")
    else:
        print("❌ Camera not accessible")
    
    # CPU and Memory test
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    
    print(f"\n💻 System Performance:")
    print(f"CPU Usage: {cpu_percent:.1f}%")
    print(f"Memory Usage: {memory_percent:.1f}%")
    
    if cpu_percent < 50 and memory_percent < 70:
        print("✅ System performance: EXCELLENT")
    elif cpu_percent < 80 and memory_percent < 85:
        print("⚠️  System performance: ADEQUATE") 
    else:
        print("❌ System performance: NEEDS IMPROVEMENT")

if __name__ == "__main__":
    benchmark_performance()
```

---

## 🎯 Quick Reference

### Essential Commands
```bash
# Standard launch
python ridebuddy_optimized_gui.py

# Vehicle mode
set RIDEBUDDY_VEHICLE_MODE=true && python ridebuddy_optimized_gui.py

# Debug mode  
python ridebuddy_optimized_gui.py --debug

# Verification
python verification_script.py

# Performance test
python benchmark_script.py
```

### Key File Locations
```
Project Structure:
├── ridebuddy_optimized_gui.py     # Main application
├── ridebuddy_config.ini          # Configuration
├── vehicle_config.json           # Vehicle settings  
├── requirements.txt              # Dependencies
├── trained_models/               # AI models
├── logs/                        # Log files
└── data/                        # Data storage
```

### Support Resources
- **System Documentation**: RIDEBUDDY_SYSTEM_DOCUMENTATION.md
- **Configuration Reference**: ridebuddy_config.ini comments
- **Troubleshooting Guide**: This document, Troubleshooting section
- **Performance Optimization**: benchmark_script.py output

---

## 📞 Support Contact

For installation support:
1. **Check this guide** - Most issues covered here
2. **Run verification script** - Automated diagnosis
3. **Check log files** - Review error messages in logs/
4. **System requirements** - Ensure minimum specs met
5. **Contact support** - With verification results if needed

---

*Installation Guide Version 2.1.0 - October 2025*
*Compatible with RideBuddy Pro v2.1.0*