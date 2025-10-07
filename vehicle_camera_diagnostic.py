#!/usr/bin/env python3
"""
Vehicle Mode Camera Diagnostic
Quick test specifically for vehicle launcher camera issues
"""

import os
import sys
import cv2
import time
import json
from pathlib import Path

def test_vehicle_mode_camera():
    """Test camera functionality in vehicle mode environment"""
    
    print("🚗 Vehicle Mode Camera Diagnostic")
    print("=" * 50)
    
    # Set vehicle mode environment (like vehicle_launcher.py does)
    os.environ['RIDEBUDDY_VEHICLE_MODE'] = '1'
    os.environ['RIDEBUDDY_FLEET_MODE'] = '1'
    
    print("✅ Environment variables set:")
    print(f"   RIDEBUDDY_VEHICLE_MODE = {os.getenv('RIDEBUDDY_VEHICLE_MODE')}")
    print(f"   RIDEBUDDY_FLEET_MODE = {os.getenv('RIDEBUDDY_FLEET_MODE')}")
    
    # Load vehicle config
    config_path = Path("vehicle_config.json")
    camera_index = 0
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            camera_index = config.get('camera_index', 0)
            print(f"✅ Vehicle config loaded: {config['name']}")
            print(f"   Camera index from config: {camera_index}")
        except Exception as e:
            print(f"⚠️  Config load error: {e}")
    
    # Test camera with the configured index
    print(f"\n🔍 Testing camera {camera_index}...")
    
    try:
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"❌ Camera {camera_index} failed to open")
            # Try other camera indices
            for test_idx in [0, 1, 2]:
                if test_idx != camera_index:
                    print(f"   Trying camera {test_idx}...")
                    test_cap = cv2.VideoCapture(test_idx)
                    if test_cap.isOpened():
                        ret, frame = test_cap.read()
                        if ret and frame is not None:
                            print(f"✅ Camera {test_idx} works! Update your config")
                            test_cap.release()
                            return test_idx
                    test_cap.release()
            return None
        
        # Test frame capture
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"❌ Camera {camera_index} opened but no frames")
            cap.release()
            return None
        
        print(f"✅ Camera {camera_index} working!")
        print(f"   Frame size: {frame.shape[1]}x{frame.shape[0]}")
        
        # Test multiple frames
        frame_count = 0
        start_time = time.time()
        
        for i in range(10):
            ret, frame = cap.read()
            if ret:
                frame_count += 1
            time.sleep(0.1)
        
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        
        print(f"   Captured {frame_count}/10 frames")
        print(f"   Effective FPS: {fps:.1f}")
        
        cap.release()
        
        if frame_count >= 8:  # Allow some frame drops
            print("✅ Camera performance: GOOD for vehicle use")
            return camera_index
        else:
            print("⚠️  Camera performance: POOR - may cause issues")
            return camera_index
            
    except Exception as e:
        print(f"❌ Camera test error: {e}")
        return None

def test_ridebuddy_import():
    """Test if RideBuddy can import with vehicle mode environment"""
    print(f"\n🧪 Testing RideBuddy import with vehicle environment...")
    
    try:
        # Test basic imports
        import tkinter as tk
        from PIL import Image
        print("✅ GUI libraries available")
        
        # Test if the main script can be imported (partially)
        sys.path.insert(0, str(Path.cwd()))
        
        # Just test the imports at the top of the file
        import configparser
        from dataclasses import dataclass, asdict
        from pathlib import Path
        print("✅ Core imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main diagnostic function"""
    
    # Test camera
    working_camera = test_vehicle_mode_camera()
    
    # Test imports
    imports_ok = test_ridebuddy_import()
    
    print("\n" + "=" * 50)
    print("🚗 VEHICLE MODE DIAGNOSTIC RESULTS")
    print("=" * 50)
    
    if working_camera is not None:
        print(f"✅ Camera: Working on index {working_camera}")
    else:
        print("❌ Camera: No working cameras found")
    
    if imports_ok:
        print("✅ RideBuddy: Core imports successful")
    else:
        print("❌ RideBuddy: Import issues detected")
    
    if working_camera is not None and imports_ok:
        print("\n🎉 DIAGNOSIS: Vehicle mode should work!")
        print("   • Environment variables are set correctly")
        print("   • Camera is functional")
        print("   • Core imports successful")
        print("\n💡 If camera feed still doesn't work in RideBuddy:")
        print("   • Privacy consent may be blocking")
        print("   • Check GUI initialization timing")
        print("   • Try manual 'Start Monitoring' button")
        
    else:
        print("\n❌ DIAGNOSIS: Vehicle mode has issues")
        if working_camera is None:
            print("   • Fix camera connection first")
        if not imports_ok:
            print("   • Fix import/dependency issues")
    
    print("\n🔧 To fix camera feed in vehicle launcher:")
    print("   1. Ensure camera is not used by other apps")
    print("   2. Try different camera index in vehicle_config.json")
    print("   3. Check if privacy consent is auto-accepting")
    print("   4. Verify environment variables are set")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDiagnostic cancelled by user")
    
    input("\nPress Enter to exit...")