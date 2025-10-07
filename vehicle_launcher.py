#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RideBuddy Pro - Vehicle Launcher
Optimized launcher for in-vehicle deployment with enhanced error handling
"""

import sys
import os
import json
import subprocess
import time
from pathlib import Path

def display_banner():
    """Display vehicle launcher banner"""
    banner = """
🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗
🚗                                          🚗
🚗    RideBuddy Pro - Vehicle Edition       🚗  
🚗    Professional Driver Monitoring       🚗
🚗    Automotive Deployment Ready          🚗
🚗                                          🚗
🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗
"""
    print(banner)

def load_vehicle_config():
    """Load vehicle-specific configuration with detailed feedback"""
    config_path = Path("vehicle_config.json")
    
    if not config_path.exists():
        print("⚠️  No vehicle config found, using default settings")
        print(f"   Expected config file: {config_path.absolute()}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Vehicle Configuration Loaded:")
        print(f"   • Name: {config.get('name', 'Unknown')}")
        print(f"   • Description: {config.get('description', 'No description')}")
        print(f"   • Frame Rate: {config.get('max_fps', 'default')} FPS")
        print(f"   • Resolution: {config.get('frame_width', 'default')}x{config.get('frame_height', 'default')}")
        print(f"   • Memory Limit: {config.get('max_memory_mb', 'default')} MB")
        
        # Check for special modes
        if config.get('fleet_mode'):
            print(f"   • Fleet Mode: ENABLED")
        if config.get('power_save_mode'):
            print(f"   • Power Save: ENABLED")
            
        return config
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid vehicle config JSON: {e}")
        print("   Please check your vehicle_config.json file")
        return None
    except Exception as e:
        print(f"⚠️  Error loading vehicle config: {e}")
        print("   Continuing with default settings...")
        return None

def setup_vehicle_environment(config):
    """Setup environment variables for vehicle deployment"""
    if not config:
        print("ℹ️  Using default environment settings")
        return
    
    print("\n🔧 Configuring Vehicle Environment:")
    
    # Set environment variables for optimization
    if config.get('power_save_mode'):
        os.environ['RIDEBUDDY_POWER_SAVE'] = '1'
        print("   • Power save mode: ENABLED")
    
    if config.get('fleet_mode'):
        os.environ['RIDEBUDDY_FLEET_MODE'] = '1' 
        print("   • Fleet management mode: ENABLED")
        
    # Set vehicle mode flag
    os.environ['RIDEBUDDY_VEHICLE_MODE'] = '1'
    print("   • Vehicle deployment mode: ENABLED")
    
    # Apply performance optimizations
    if config.get('max_memory_mb'):
        print(f"   • Memory limit: {config['max_memory_mb']} MB")
    
    if config.get('alert_sensitivity'):
        print(f"   • Alert sensitivity: {config['alert_sensitivity']}")

def check_system_requirements():
    """Check if system meets vehicle deployment requirements"""
    print("\n🔍 Vehicle System Requirements Check:")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print(f"   ✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"   ❌ Python {python_version.major}.{python_version.minor}.{python_version.micro} (3.8+ required)")
        return False
    
    # Check main application
    main_script = Path("ridebuddy_optimized_gui.py")
    if main_script.exists():
        print(f"   ✅ Main application found")
    else:
        print(f"   ❌ Main application missing: {main_script}")
        return False
    
    # Check dependencies (basic check)
    try:
        import cv2
        print(f"   ✅ OpenCV available")
    except ImportError:
        print(f"   ❌ OpenCV missing (required for camera)")
        return False
    
    try:
        from PIL import Image
        print(f"   ✅ PIL/Pillow available")
    except ImportError:
        print(f"   ❌ PIL/Pillow missing (required for image processing)")
        return False
    
    return True

def launch_ridebuddy():
    """Launch the main RideBuddy application"""
    print("\n🚀 Launching RideBuddy Pro for Vehicle Deployment...")
    
    script_path = Path("ridebuddy_optimized_gui.py")
    
    if not script_path.exists():
        print(f"❌ Main application not found: {script_path.absolute()}")
        return False
    
    try:
        print(f"   Starting application: {script_path}")
        print(f"   Working directory: {Path.cwd()}")
        
        # Launch with real-time output (no capture for debugging)
        process = subprocess.Popen([
            sys.executable, str(script_path)
        ], 
        cwd=Path.cwd()
        )
        
        print("✅ Application launched successfully!")
        print("📱 RideBuddy Pro should now be running...")
        print("📊 Debug output will appear in the application console...")
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            print("✅ Application completed successfully")
            return True
        else:
            print(f"❌ Application exited with error code: {return_code}")
            return False
            
    except FileNotFoundError:
        print(f"❌ Python executable not found: {sys.executable}")
        return False
    except Exception as e:
        print(f"❌ Failed to launch application: {e}")
        return False

def main():
    """Main vehicle launcher function"""
    # Display banner
    display_banner()
    
    # Check system requirements first
    if not check_system_requirements():
        print("\n❌ VEHICLE DEPLOYMENT FAILED:")
        print("   System requirements not met")
        print("\n🔧 Troubleshooting:")
        print("   • Install Python 3.8 or higher")
        print("   • Install required packages: pip install opencv-python pillow")
        print("   • Ensure ridebuddy_optimized_gui.py is in the same directory")
        input("\nPress Enter to exit...")
        return 1
    
    # Load vehicle configuration
    config = load_vehicle_config()
    
    # Setup vehicle environment
    setup_vehicle_environment(config)
    
    # Launch main application
    if launch_ridebuddy():
        print("\n🎉 Vehicle deployment successful!")
        return 0
    else:
        print("\n❌ VEHICLE DEPLOYMENT FAILED:")
        print("   Application could not start")
        print("\n🔧 Troubleshooting:")
        print("   • Check camera is connected and not in use")
        print("   • Try running: python ridebuddy_optimized_gui.py")
        print("   • Check the logs folder for detailed errors")
        print("   • Verify vehicle_config.json is valid JSON")
        input("\nPress Enter to exit...")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n🛑 Vehicle launcher interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error in vehicle launcher: {e}")
        input("Press Enter to exit...")
        sys.exit(1)
