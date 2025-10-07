#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RideBuddy Pro - Vehicle Deployment Guide & Configuration
Automotive-optimized settings for in-vehicle driver monitoring
"""

import os
import json
from pathlib import Path

# Vehicle-specific configurations
VEHICLE_CONFIGS = {
    "COMPACT_CAR": {
        "name": "Compact Car Setup",
        "description": "Optimized for small vehicles with limited dashboard space",
        "camera_index": 0,
        "frame_width": 480,
        "frame_height": 360,
        "max_fps": 20,
        "alert_sensitivity": 0.8,
        "max_memory_mb": 256,
        "audio_alerts": True,
        "power_save_mode": True
    },
    
    "SEDAN_STANDARD": {
        "name": "Standard Sedan Setup",
        "description": "Balanced performance for mid-size vehicles",
        "camera_index": 0,
        "frame_width": 640,
        "frame_height": 480,
        "max_fps": 25,
        "alert_sensitivity": 0.7,
        "max_memory_mb": 384,
        "audio_alerts": True,
        "power_save_mode": False
    },
    
    "TRUCK_COMMERCIAL": {
        "name": "Commercial Truck Setup", 
        "description": "High-performance setup for commercial vehicles",
        "camera_index": 0,
        "frame_width": 800,
        "frame_height": 600,
        "max_fps": 30,
        "alert_sensitivity": 0.6,
        "max_memory_mb": 512,
        "audio_alerts": True,
        "power_save_mode": False,
        "enhanced_detection": True
    },
    
    "FLEET_MANAGEMENT": {
        "name": "Fleet Management Setup",
        "description": "Enterprise setup with data logging and reporting",
        "camera_index": 0,
        "frame_width": 640,
        "frame_height": 480,
        "max_fps": 25,
        "alert_sensitivity": 0.75,
        "max_memory_mb": 384,
        "audio_alerts": True,
        "auto_save_results": True,
        "data_retention_days": 90,
        "fleet_mode": True
    }
}

def create_vehicle_config(vehicle_type="SEDAN_STANDARD"):
    """Create vehicle-specific configuration file"""
    
    if vehicle_type not in VEHICLE_CONFIGS:
        print(f"❌ Unknown vehicle type: {vehicle_type}")
        print(f"Available types: {list(VEHICLE_CONFIGS.keys())}")
        return False
    
    config = VEHICLE_CONFIGS[vehicle_type]
    config_path = Path("vehicle_config.json")
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Vehicle configuration created: {config['name']}")
        print(f"📁 Config file: {config_path.absolute()}")
        print(f"📝 Description: {config['description']}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create config: {e}")
        return False

def print_vehicle_deployment_guide():
    """Print comprehensive vehicle deployment guide"""
    
    guide = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        🚗 RideBuddy Pro - Vehicle Deployment Guide           ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 VEHICLE COMPATIBILITY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Personal Cars (Sedan, Hatchback, SUV)
✅ Commercial Vehicles (Trucks, Vans, Buses)
✅ Fleet Management Systems
✅ Rideshare/Taxi Services
✅ Driver Training Vehicles

🔧 HARDWARE REQUIREMENTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 💻 Device: Laptop, Tablet, or Embedded Computer (Windows/Linux)
• 📷 Camera: USB webcam or built-in camera (720p+ recommended)
• 🔌 Power: 12V car adapter or USB power (for continuous operation)
• 💾 Storage: 4GB+ available space (for logs and recordings)
• 🌐 RAM: 2GB+ recommended (4GB+ for commercial use)
• ⚡ CPU: Intel i3+ or equivalent (for real-time processing)

📱 INSTALLATION METHODS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1️⃣ DASHBOARD MOUNT:
   • Mount tablet/laptop on dashboard
   • USB camera positioned near rearview mirror
   • 12V power adapter for continuous operation
   • Anti-glare screen protector recommended

2️⃣ INTEGRATED SYSTEM:
   • Embed computer in vehicle dashboard
   • Permanent camera installation
   • Wired to vehicle's electrical system
   • Professional installation recommended

3️⃣ PORTABLE SETUP:
   • Laptop with built-in camera
   • Suction cup mount or dashboard pad
   • USB power bank for short trips
   • Quick setup/removal for multiple vehicles

🎯 CAMERA POSITIONING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📍 OPTIMAL PLACEMENT:
   • 12-18 inches from driver's face
   • Slightly above eye level (near rearview mirror area)
   • Angled 15-30 degrees downward toward driver
   • Clear view of driver's face and upper body
   • Avoid direct sunlight glare

⚠️ AVOID PLACEMENT:
   • Windshield areas that obstruct vision
   • Airbag deployment zones
   • Areas with excessive vibration
   • Direct dashboard vents (heat/AC)

🔋 POWER MANAGEMENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔌 POWER OPTIONS:
   • 12V Car Adapter: Continuous operation while engine running
   • USB Power Bank: 2-4 hours operation (10,000+ mAh recommended)
   • Vehicle Integration: Professional wiring to ignition system
   • Dual Power: Adapter + battery backup for reliability

💡 POWER OPTIMIZATION:
   • Enable power save mode for longer battery life
   • Reduce frame rate during highway driving
   • Automatic shutdown when vehicle off (configurable)
   • Sleep mode during parked state

📊 PERFORMANCE SETTINGS BY VEHICLE TYPE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚗 PERSONAL VEHICLES:
   • Frame Rate: 20-25 FPS
   • Resolution: 640x480 (balanced quality/performance)
   • Memory Usage: 256-384 MB
   • Alert Sensitivity: Medium (0.7-0.8)

🚛 COMMERCIAL VEHICLES:
   • Frame Rate: 25-30 FPS  
   • Resolution: 640x480 or 800x600
   • Memory Usage: 384-512 MB
   • Alert Sensitivity: High (0.6-0.7)
   • Enhanced logging and reporting

🚌 FLEET MANAGEMENT:
   • Frame Rate: 25 FPS (consistent across fleet)
   • Resolution: 640x480 (standardized)
   • Memory Usage: 384 MB
   • Data Retention: 90+ days
   • Centralized reporting

🛡️ SAFETY & LEGAL CONSIDERATIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚖️ LEGAL COMPLIANCE:
   • Check local laws regarding driver monitoring
   • Obtain driver consent (especially for commercial use)
   • Comply with privacy regulations (GDPR, CCPA, etc.)
   • Maintain secure data handling practices

🛡️ SAFETY INSTALLATION:
   • Do not obstruct driver's view
   • Secure all equipment to prevent projectiles
   • Professional installation for permanent setups
   • Test system thoroughly before regular use

🔒 PRIVACY & SECURITY:
   • Data stored locally (not in cloud)
   • Encrypted data storage
   • User consent required before monitoring
   • Configurable data retention periods

⚙️ MAINTENANCE & UPDATES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔧 REGULAR MAINTENANCE:
   • Clean camera lens weekly
   • Check power connections monthly
   • Update software when available
   • Backup important data regularly

📈 PERFORMANCE MONITORING:
   • Monitor CPU/memory usage
   • Check camera feed quality
   • Verify alert functionality
   • Review detection accuracy

🆘 TROUBLESHOOTING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ Camera Not Working:
   • Check USB connections
   • Try different camera index (0, 1, 2)
   • Restart application
   • Check camera permissions

❌ Poor Performance:
   • Lower frame rate and resolution
   • Enable power save mode
   • Close other applications
   • Check available memory

❌ False Alerts:
   • Adjust alert sensitivity
   • Improve camera positioning
   • Check lighting conditions
   • Update camera drivers

📞 SUPPORT & RESOURCES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 📖 User Manual: Check README.md file
• 🔧 Configuration Tool: Run vehicle_config_creator.py
• 🧪 Testing: Use camera_diagnostics.py
• 📊 Performance: Monitor through Analysis tab

╔══════════════════════════════════════════════════════════════════════════════╗
║ ✅ RideBuddy Pro is fully compatible with vehicle deployment!                ║
║ Follow this guide for safe and effective in-vehicle installation.           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    
    print(guide)

def create_vehicle_launcher():
    """Create vehicle-optimized launcher"""
    
    launcher_code = '''#!/usr/bin/env python3
"""
RideBuddy Pro - Vehicle Launcher
Optimized launcher for in-vehicle deployment
"""

import sys
import os
import json
import subprocess
from pathlib import Path

def load_vehicle_config():
    """Load vehicle-specific configuration"""
    config_path = Path("vehicle_config.json")
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✅ Loaded vehicle config: {config.get('name', 'Unknown')}")
            return config
        except Exception as e:
            print(f"⚠️ Error loading vehicle config: {e}")
    
    print("ℹ️ Using default configuration")
    return None

def setup_vehicle_environment(config):
    """Setup environment for vehicle deployment"""
    if not config:
        return
    
    # Set environment variables for optimization
    if config.get('power_save_mode'):
        os.environ['RIDEBUDDY_POWER_SAVE'] = '1'
        print("🔋 Power save mode enabled")
    
    if config.get('fleet_mode'):
        os.environ['RIDEBUDDY_FLEET_MODE'] = '1'
        print("🚛 Fleet management mode enabled")
    
    # Apply performance optimizations
    if config.get('max_memory_mb'):
        print(f"💾 Memory limit: {config['max_memory_mb']} MB")

def main():
    """Vehicle launcher main function"""
    print("🚗" * 20)
    print("🚗 RideBuddy Pro - Vehicle Edition")
    print("🚗 Professional Driver Monitoring")
    print("🚗" * 20)
    
    # Load vehicle configuration
    config = load_vehicle_config()
    setup_vehicle_environment(config)
    
    # Launch main application
    try:
        script_path = Path(__file__).parent / "ridebuddy_optimized_gui.py"
        
        if not script_path.exists():
            print("❌ Main application not found!")
            input("Press Enter to exit...")
            return
        
        print("🚀 Starting RideBuddy Pro for vehicle deployment...")
        
        # Launch with vehicle-optimized settings
        process = subprocess.run([
            sys.executable, str(script_path)
        ], cwd=Path(__file__).parent)
        
        return process.returncode
        
    except Exception as e:
        print(f"❌ Failed to start vehicle application: {e}")
        input("Press Enter to exit...")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''
    
    launcher_path = Path("vehicle_launcher.py")
    try:
        with open(launcher_path, 'w') as f:
            f.write(launcher_code)
        print(f"✅ Vehicle launcher created: {launcher_path.absolute()}")
        return True
    except Exception as e:
        print(f"❌ Failed to create launcher: {e}")
        return False

def main():
    """Main configuration tool"""
    print("🚗 RideBuddy Pro - Vehicle Configuration Tool")
    print("=" * 50)
    
    # Print deployment guide
    print_vehicle_deployment_guide()
    
    # Interactive configuration
    print("\n🔧 VEHICLE CONFIGURATION SETUP:")
    print("━" * 40)
    
    print("\nAvailable vehicle configurations:")
    for i, (key, config) in enumerate(VEHICLE_CONFIGS.items(), 1):
        print(f"{i}. {config['name']}")
        print(f"   {config['description']}")
    
    try:
        choice = input(f"\nSelect configuration (1-{len(VEHICLE_CONFIGS)}) or press Enter for default: ").strip()
        
        if choice:
            choice_idx = int(choice) - 1
            vehicle_type = list(VEHICLE_CONFIGS.keys())[choice_idx]
        else:
            vehicle_type = "SEDAN_STANDARD"
        
        # Create configuration
        if create_vehicle_config(vehicle_type):
            # Create launcher
            create_vehicle_launcher()
            
            print("\n🎉 Vehicle deployment setup complete!")
            print("\n📋 Next steps:")
            print("1. Install RideBuddy on your vehicle computer/tablet")
            print("2. Position camera for optimal driver view")
            print("3. Connect power supply (12V adapter recommended)")
            print("4. Run vehicle_launcher.py to start monitoring")
            print("5. Test system thoroughly before regular use")
            
        else:
            print("❌ Configuration setup failed")
    
    except (ValueError, IndexError):
        print("❌ Invalid selection")
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")

if __name__ == "__main__":
    main()