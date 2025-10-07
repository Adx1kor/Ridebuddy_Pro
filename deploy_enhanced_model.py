#!/usr/bin/env python3
"""
RideBuddy Pro v2.1.0 - Enhanced Model Deployment Script
Deploys the trained model for production use.
"""

import os
import shutil
import torch
from pathlib import Path

def deploy_model():
    """Deploy enhanced model for production"""
    
    print("🚀 Deploying Enhanced RideBuddy Model...")
    
    # Create production directories
    prod_dir = Path("production")
    prod_dir.mkdir(exist_ok=True)
    
    models_dir = prod_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Copy model files
    if os.path.exists("models/best_driver_model.pth"):
        shutil.copy2("models/best_driver_model.pth", models_dir / "driver_model.pth")
        print("✅ Model copied to production directory")
    
    # Copy integration files
    files_to_copy = [
        "model_integration.py",
        "enhanced_model_integration.py",
        "ridebuddy_optimized_gui.py",
        "vehicle_launcher.py",
        "camera_diagnostics.py"
    ]
    
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy2(file, prod_dir / file)
            print(f"✅ Copied {file}")
    
    # Create production config
    prod_config = f"""[Production]
model_path = models/driver_model.pth
deployment_date = {datetime.now().isoformat()}
version = 2.1.0-enhanced

[Camera]
device_id = 0
width = 640
height = 480
fps = 25

[Detection]
enhanced_model = True
confidence_threshold = 0.75
temporal_smoothing = True
"""
    
    with open(prod_dir / "production_config.ini", 'w', encoding='utf-8') as f:
        f.write(prod_config)
    
    print("✅ Production configuration created")
    print(f"📦 Production deployment ready in: {prod_dir.absolute()}")
    
    return True

if __name__ == "__main__":
    deploy_model()
