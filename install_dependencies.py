#!/usr/bin/env python3
"""
RideBuddy Pro - Dependencies Installer
Automatically installs missing dependencies for optimal performance
"""

import subprocess
import sys
import importlib
import os

def check_and_install_package(package_name, import_name=None, pip_name=None):
    """Check if package is available and install if missing"""
    if import_name is None:
        import_name = package_name
    if pip_name is None:
        pip_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} - Already installed")
        return True
    except ImportError:
        print(f"❌ {package_name} - Missing, attempting to install...")
        
        try:
            # Install using pip
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", pip_name, "--upgrade"
            ])
            
            # Verify installation
            importlib.import_module(import_name)
            print(f"✅ {package_name} - Successfully installed!")
            return True
            
        except (subprocess.CalledProcessError, ImportError) as e:
            print(f"❌ {package_name} - Installation failed: {e}")
            return False

def main():
    """Main installer function"""
    print("🔧 RideBuddy Pro - Dependencies Installer")
    print("="*50)
    
    # Core dependencies (required)
    core_dependencies = [
        ("OpenCV", "cv2", "opencv-python"),
        ("PIL/Pillow", "PIL", "Pillow"),
        ("NumPy", "numpy", "numpy"),
    ]
    
    # Optional dependencies (for enhanced features)
    optional_dependencies = [
        ("psutil", "psutil", "psutil"),
        ("matplotlib", "matplotlib", "matplotlib"),
        ("scikit-learn", "sklearn", "scikit-learn"),
    ]
    
    print("\n📦 Installing Core Dependencies:")
    core_success = []
    for package_name, import_name, pip_name in core_dependencies:
        success = check_and_install_package(package_name, import_name, pip_name)
        core_success.append(success)
    
    print(f"\n🎯 Optional Dependencies (for enhanced performance):")
    optional_success = []
    for package_name, import_name, pip_name in optional_dependencies:
        success = check_and_install_package(package_name, import_name, pip_name)
        optional_success.append(success)
    
    # Summary
    print("\n" + "="*50)
    print("📊 INSTALLATION SUMMARY")
    print("="*50)
    
    core_installed = sum(core_success)
    total_core = len(core_dependencies)
    optional_installed = sum(optional_success)
    total_optional = len(optional_dependencies)
    
    print(f"Core Dependencies: {core_installed}/{total_core} installed")
    print(f"Optional Dependencies: {optional_installed}/{total_optional} installed")
    
    if all(core_success):
        print("\n✅ All core dependencies are installed!")
        print("🚗 RideBuddy Pro is ready to run with full functionality.")
        
        if optional_installed == total_optional:
            print("🚀 All optional dependencies are also installed for maximum performance!")
        elif optional_installed > 0:
            print(f"⚡ {optional_installed} optional dependencies installed for enhanced features.")
        else:
            print("⚠️ No optional dependencies installed. Some advanced features may be limited.")
            
    else:
        missing_core = [dep[0] for dep, success in zip(core_dependencies, core_success) if not success]
        print(f"\n❌ Missing core dependencies: {', '.join(missing_core)}")
        print("RideBuddy Pro may not function properly without these packages.")
        print("\nPlease install manually using:")
        for dep, success in zip(core_dependencies, core_success):
            if not success:
                print(f"  pip install {dep[2]}")
    
    print("\n" + "="*50)
    input("Press Enter to continue...")

if __name__ == "__main__":
    main()