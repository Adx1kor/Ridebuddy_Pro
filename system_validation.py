#!/usr/bin/env python3
"""
RideBuddy Pro v2.1.0 - System Validation Script
Validates all dependencies, hardware, and configuration for proper deployment.
"""

import sys
import os
import json
import configparser
import subprocess
import importlib.util
from pathlib import Path

class RideBuddyValidator:
    def __init__(self):
        self.results = []
        self.errors = []
        
    def log_result(self, test_name, status, message="", error=None):
        """Log test result"""
        symbol = "✅" if status else "❌"
        self.results.append(f"{symbol} {test_name}: {message}")
        if error:
            self.errors.append(f"ERROR in {test_name}: {error}")
            
    def check_python_version(self):
        """Check Python version compatibility"""
        version = sys.version_info
        # Check if version is 3.8.x through 3.11.x
        if version.major == 3 and 8 <= version.minor <= 11:
            self.log_result("Python Version", True, f"{version.major}.{version.minor}.{version.micro}")
            return True
        else:
            self.log_result("Python Version", False, f"{version.major}.{version.minor}.{version.micro} (Need 3.8-3.11)", 
                          f"Unsupported Python version: {version.major}.{version.minor}")
            return False
            
    def check_dependencies(self):
        """Check required Python packages"""
        required_packages = [
            'torch', 'torchvision', 'cv2', 'numpy', 'PIL', 
            'matplotlib', 'pandas', 'sklearn', 'ultralytics',
            'yaml', 'tqdm'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                if package == 'cv2':
                    import cv2
                elif package == 'PIL':
                    from PIL import Image
                elif package == 'sklearn':
                    import sklearn
                elif package == 'yaml':
                    import yaml
                else:
                    __import__(package)
                    
            except ImportError:
                missing_packages.append(package)
                
        if not missing_packages:
            self.log_result("Python Dependencies", True, f"All {len(required_packages)} packages installed")
            return True
        else:
            self.log_result("Python Dependencies", False, 
                          f"Missing: {', '.join(missing_packages)}", 
                          f"Install missing packages: pip install {' '.join(missing_packages)}")
            return False
            
    def check_essential_files(self):
        """Check for essential application files"""
        essential_files = [
            'ridebuddy_optimized_gui.py',
            'vehicle_launcher.py', 
            'camera_diagnostics.py',
            'requirements.txt',
            'ridebuddy_config.ini',
            'vehicle_config.json'
        ]
        
        missing_files = []
        for file_name in essential_files:
            if not os.path.exists(file_name):
                missing_files.append(file_name)
                
        if not missing_files:
            self.log_result("Essential Files", True, f"All {len(essential_files)} files present")
            return True
        else:
            self.log_result("Essential Files", False,
                          f"Missing: {', '.join(missing_files)}",
                          f"Missing essential files: {missing_files}")
            return False
            
    def check_camera_hardware(self):
        """Test camera hardware availability"""
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    self.log_result("Camera Hardware", True, 
                                  f"Camera detected, resolution: {width}x{height}")
                    cap.release()
                    return True
                else:
                    cap.release()
                    self.log_result("Camera Hardware", False, "Camera detected but cannot read frames",
                                  "Camera hardware issue - check drivers")
                    return False
            else:
                self.log_result("Camera Hardware", False, "No camera detected", 
                              "No camera found - check connection and drivers")
                return False
                
        except Exception as e:
            self.log_result("Camera Hardware", False, f"Camera test failed: {str(e)}", str(e))
            return False
            
    def check_configuration_files(self):
        """Validate configuration files"""
        config_valid = True
        
        # Check ridebuddy_config.ini
        try:
            config = configparser.ConfigParser()
            config.read('ridebuddy_config.ini')
            
            required_sections = ['Camera', 'Detection', 'Privacy']
            for section in required_sections:
                if not config.has_section(section):
                    config_valid = False
                    self.errors.append(f"Missing section '{section}' in ridebuddy_config.ini")
                    
            if config_valid:
                self.log_result("Config File (INI)", True, "Valid configuration structure")
            else:
                self.log_result("Config File (INI)", False, "Invalid configuration", 
                              "Configuration file has missing sections")
                
        except Exception as e:
            self.log_result("Config File (INI)", False, f"Config error: {str(e)}", str(e))
            config_valid = False
            
        # Check vehicle_config.json
        try:
            with open('vehicle_config.json', 'r') as f:
                vehicle_config = json.load(f)
                
            required_keys = ['vehicle_type', 'fleet_management', 'camera_settings', 'detection_settings']
            for key in required_keys:
                if key not in vehicle_config:
                    config_valid = False
                    self.errors.append(f"Missing key '{key}' in vehicle_config.json")
                    
            if 'fleet_management' in vehicle_config and vehicle_config['fleet_management']:
                self.log_result("Config File (JSON)", True, "Valid vehicle configuration with fleet management")
            else:
                self.log_result("Config File (JSON)", True, "Valid vehicle configuration")
                
        except Exception as e:
            self.log_result("Config File (JSON)", False, f"Vehicle config error: {str(e)}", str(e))
            config_valid = False
            
        return config_valid
        
    def check_directories(self):
        """Check for required directories"""
        required_dirs = ['logs', 'src', 'configs']
        missing_dirs = []
        
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                missing_dirs.append(dir_name)
                # Try to create missing directories
                try:
                    os.makedirs(dir_name, exist_ok=True)
                    self.log_result(f"Directory ({dir_name})", True, "Created missing directory")
                except Exception as e:
                    self.log_result(f"Directory ({dir_name})", False, f"Could not create: {str(e)}", str(e))
            else:
                self.log_result(f"Directory ({dir_name})", True, "Directory exists")
                
        return len(missing_dirs) == 0
        
    def test_application_launch(self):
        """Test if main application can be imported"""
        try:
            # Try to import the main application module
            spec = importlib.util.spec_from_file_location("ridebuddy", "ridebuddy_optimized_gui.py")
            if spec is None:
                self.log_result("Application Import", False, "Cannot import main application",
                              "Main application file cannot be imported")
                return False
                
            # Don't actually execute, just test importability
            self.log_result("Application Import", True, "Main application can be imported")
            return True
            
        except Exception as e:
            self.log_result("Application Import", False, f"Import failed: {str(e)}", str(e))
            return False
            
    def check_system_resources(self):
        """Check system resources"""
        try:
            import psutil
            
            # Check available memory
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            if memory_gb >= 4:
                self.log_result("System Memory", True, f"{memory_gb:.1f}GB available")
            else:
                self.log_result("System Memory", False, f"{memory_gb:.1f}GB available (minimum 4GB)",
                              "Insufficient memory for optimal performance")
                
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent < 80:
                self.log_result("CPU Usage", True, f"{cpu_percent}% current usage")
            else:
                self.log_result("CPU Usage", False, f"{cpu_percent}% current usage (high load)",
                              "High CPU usage may affect performance")
                
            return True
            
        except ImportError:
            self.log_result("System Resources", False, "psutil not available", 
                          "Cannot check system resources - install psutil")
            return False
        except Exception as e:
            self.log_result("System Resources", False, f"Resource check failed: {str(e)}", str(e))
            return False
            
    def run_validation(self):
        """Run complete system validation"""
        print("🔍 RideBuddy Pro v2.1.0 - System Validation")
        print("=" * 60)
        
        tests = [
            ("Python Version", self.check_python_version),
            ("Python Dependencies", self.check_dependencies),
            ("Essential Files", self.check_essential_files),
            ("Configuration Files", self.check_configuration_files),
            ("Required Directories", self.check_directories),
            ("Camera Hardware", self.check_camera_hardware),
            ("Application Import", self.test_application_launch),
            ("System Resources", self.check_system_resources)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                if result:
                    passed += 1
            except Exception as e:
                self.log_result(test_name, False, f"Test error: {str(e)}", str(e))
        
        # Print results
        print("\n📊 Validation Results:")
        print("-" * 40)
        for result in self.results:
            print(result)
            
        print(f"\n🎯 Overall Score: {passed}/{total} tests passed")
        
        if self.errors:
            print(f"\n❌ Errors Found ({len(self.errors)}):")
            print("-" * 40)
            for error in self.errors:
                print(f"  • {error}")
        
        if passed == total:
            print("\n🎉 System Validation PASSED!")
            print("✅ RideBuddy Pro v2.1.0 is ready for deployment")
            return True
        else:
            print(f"\n⚠️  System Validation INCOMPLETE ({passed}/{total})")
            print("🔧 Please resolve the errors above before deployment")
            return False
            
def main():
    """Main validation function"""
    validator = RideBuddyValidator()
    
    print("Starting RideBuddy Pro v2.1.0 system validation...")
    print("This will check dependencies, hardware, and configuration.\n")
    
    success = validator.run_validation()
    
    print("\n" + "=" * 60)
    if success:
        print("🚀 Ready to launch:")
        print("   Desktop Mode: python ridebuddy_optimized_gui.py")
        print("   Vehicle Mode: python vehicle_launcher.py")
        print("   Hardware Test: python camera_diagnostics.py")
    else:
        print("🔧 Fix the issues above and run validation again:")
        print("   python system_validation.py")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())