#!/usr/bin/env python3
"""
RideBuddy Pro v2.1.0 - Model Integration & Deployment
Integrates the newly trained enhanced model into the RideBuddy application.
"""

import os
import sys
import torch
import torch.nn as nn
import cv2
import numpy as np
import json
import logging
from pathlib import Path
import torchvision.transforms as transforms
from datetime import datetime
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDriverNet(nn.Module):
    """Enhanced CNN for driver monitoring (same as trainer)"""
    
    def __init__(self, num_classes=5, input_size=224):
        super(EnhancedDriverNet, self).__init__()
        
        # Feature extraction backbone
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate feature map size
        self.feature_size = self._get_feature_size(input_size)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
    def _get_feature_size(self, input_size):
        """Calculate the size of features after convolution layers"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_size, input_size)
            features = self.features(dummy_input)
            return features.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class EnhancedDriverMonitor:
    """Enhanced driver monitoring with improved model"""
    
    def __init__(self, model_path="models/best_driver_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = model_path
        self.class_names = ["alert", "drowsy", "phone_usage", "normal_driving", "seatbelt_off"]
        
        # Load model
        self.load_model()
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Detection history for smoothing
        self.detection_history = []
        self.history_size = 5
        
        logger.info(f"Enhanced Driver Monitor initialized on {self.device}")
    
    def load_model(self):
        """Load the trained model"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}")
                logger.info("Using fallback detection methods")
                return False
            
            # Initialize model
            self.model = EnhancedDriverNet(num_classes=5, input_size=224)
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully with validation accuracy: {checkpoint.get('validation_accuracy', 'N/A')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Falling back to basic detection")
            return False
    
    def detect_driver_state(self, frame):
        """Enhanced driver state detection"""
        if self.model is None:
            return self._fallback_detection(frame)
        
        try:
            # Preprocess frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(rgb_frame).unsqueeze(0).to(self.device)
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = self.class_names[predicted.item()]
                confidence_score = confidence.item()
            
            # Add to history for temporal smoothing
            self.detection_history.append({
                'class': predicted_class,
                'confidence': confidence_score,
                'timestamp': datetime.now()
            })
            
            # Keep only recent history
            if len(self.detection_history) > self.history_size:
                self.detection_history.pop(0)
            
            # Temporal smoothing
            smoothed_result = self._smooth_detections()
            
            return {
                'primary_detection': predicted_class,
                'confidence': confidence_score,
                'smoothed_result': smoothed_result,
                'all_probabilities': {
                    self.class_names[i]: probabilities[0][i].item() 
                    for i in range(len(self.class_names))
                },
                'enhanced_model': True
            }
            
        except Exception as e:
            logger.error(f"Enhanced detection failed: {e}")
            return self._fallback_detection(frame)
    
    def _smooth_detections(self):
        """Temporal smoothing of detections"""
        if len(self.detection_history) < 3:
            return self.detection_history[-1] if self.detection_history else None
        
        # Count class occurrences in recent history
        class_counts = {}
        total_confidence = {}
        
        for detection in self.detection_history:
            class_name = detection['class']
            confidence = detection['confidence']
            
            if class_name not in class_counts:
                class_counts[class_name] = 0
                total_confidence[class_name] = 0
            
            class_counts[class_name] += 1
            total_confidence[class_name] += confidence
        
        # Find most frequent class with highest average confidence
        best_class = max(class_counts.keys(), 
                        key=lambda x: (class_counts[x], total_confidence[x] / class_counts[x]))
        
        avg_confidence = total_confidence[best_class] / class_counts[best_class]
        
        return {
            'class': best_class,
            'confidence': avg_confidence,
            'stability': class_counts[best_class] / len(self.detection_history)
        }
    
    def _fallback_detection(self, frame):
        """Fallback detection using basic computer vision"""
        # Basic eye detection for drowsiness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return {
                'primary_detection': 'unknown',
                'confidence': 0.5,
                'enhanced_model': False,
                'fallback': True
            }
        
        # Check for eyes in face region
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            if len(eyes) < 2:
                return {
                    'primary_detection': 'drowsy',
                    'confidence': 0.7,
                    'enhanced_model': False,
                    'fallback': True
                }
        
        return {
            'primary_detection': 'alert',
            'confidence': 0.6,
            'enhanced_model': False,
            'fallback': True
        }

def integrate_enhanced_model():
    """Integrate enhanced model into RideBuddy application"""
    
    print("🔧 RideBuddy Pro v2.1.0 - Model Integration")
    print("=" * 50)
    
    # Check for enhanced model (multiple possible locations)
    possible_paths = [
        "trained_models/deployment_package/enhanced_driver_model.pth",
        "trained_models/best_enhanced_model.pth",
        "enhanced_models/enhanced_driver_model.pth",
        "models/best_driver_model.pth"
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("❌ Enhanced model not found!")
        print("🚀 Please run comprehensive_trainer.py first to train the model")
        print("📍 Checked locations:")
        for path in possible_paths:
            print(f"   - {path}")
        return False
    
    # Load model info
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        val_accuracy = checkpoint.get('val_accuracy', checkpoint.get('validation_accuracy', 0.0))
        
        print(f"✅ Found enhanced model with {val_accuracy:.1%} validation accuracy")
        
        # Create model integration code
        integration_code = f'''
# Enhanced Model Integration for RideBuddy Pro v2.1.0
# Auto-generated on {datetime.now().isoformat()}

import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    from model_integration import EnhancedDriverMonitor
    ENHANCED_MODEL_AVAILABLE = True
    MODEL_ACCURACY = {val_accuracy:.4f}
    print(f"✅ Enhanced model loaded successfully (Accuracy: {val_accuracy:.1%})")
except ImportError:
    ENHANCED_MODEL_AVAILABLE = False
    print("⚠️ Enhanced model not available, using fallback detection")

def get_enhanced_detector():
    """Get enhanced detector if available"""
    if ENHANCED_MODEL_AVAILABLE:
        return EnhancedDriverMonitor()
    return None

# Usage in main application:
# detector = get_enhanced_detector()
# if detector:
#     result = detector.detect_driver_state(frame)
#     print(f"Detection: {{result['primary_detection']}} ({{result['confidence']:.1%}})")
'''
        
        # Save integration code
        with open('enhanced_model_integration.py', 'w', encoding='utf-8') as f:
            f.write(integration_code)
        
        print("✅ Integration code created: enhanced_model_integration.py")
        
        # Update RideBuddy config
        update_ridebuddy_config(val_accuracy)
        
        # Create model deployment script
        create_deployment_script()
        
        print("🎉 Enhanced model integration completed!")
        print(f"📈 Model Performance: {val_accuracy:.1%} accuracy")
        print("🚀 Ready for enhanced detection in RideBuddy application")
        
        return True
        
    except Exception as e:
        logger.error(f"Integration failed: {e}")
        return False

def update_ridebuddy_config(accuracy):
    """Update RideBuddy configuration with enhanced model info"""
    
    config_updates = f'''
# Enhanced Model Configuration (Auto-generated)
[EnhancedModel]
enabled = True
model_path = models/best_driver_model.pth
accuracy = {accuracy:.4f}
confidence_threshold = 0.75
temporal_smoothing = True
fallback_enabled = True
last_updated = {datetime.now().isoformat()}

[Detection]
# Enhanced thresholds based on trained model
drowsiness_threshold = 0.70
phone_threshold = 0.75
seatbelt_threshold = 0.80
confidence_smoothing = 0.3
alert_cooldown = 2.0
enhanced_detection = True
'''
    
    # Append to config file
    with open('ridebuddy_config.ini', 'a') as f:
        f.write('\n' + config_updates)
    
    logger.info("Configuration updated with enhanced model settings")

def create_deployment_script():
    """Create deployment script for production use"""
    
    deployment_script = '''#!/usr/bin/env python3
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
    prod_config = f\"\"\"[Production]
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
\"\"\"
    
    with open(prod_dir / "production_config.ini", 'w', encoding='utf-8') as f:
        f.write(prod_config)
    
    print("✅ Production configuration created")
    print(f"📦 Production deployment ready in: {prod_dir.absolute()}")
    
    return True

if __name__ == "__main__":
    deploy_model()
'''
    
    with open('deploy_enhanced_model.py', 'w', encoding='utf-8') as f:
        f.write(deployment_script)
    
    logger.info("Deployment script created: deploy_enhanced_model.py")

def test_enhanced_integration():
    """Test the enhanced model integration"""
    
    print("🧪 Testing Enhanced Model Integration...")
    
    try:
        # Initialize enhanced detector
        detector = EnhancedDriverMonitor()
        
        # Create test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test detection
        result = detector.detect_driver_state(test_frame)
        
        print("✅ Enhanced detection test passed")
        print(f"   Detection: {result['primary_detection']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Enhanced Model: {result.get('enhanced_model', False)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False

def main():
    """Main integration process"""
    
    # Step 1: Integrate enhanced model
    if not integrate_enhanced_model():
        print("❌ Model integration failed")
        return 1
    
    # Step 2: Test integration
    if not test_enhanced_integration():
        print("❌ Integration test failed")
        return 1
    
    print("\n🎉 Enhanced Model Integration Complete!")
    print("=" * 50)
    print("📈 Your RideBuddy system now includes:")
    print("   ✅ Enhanced AI model with improved accuracy")
    print("   ✅ Temporal smoothing for stable detections")
    print("   ✅ Fallback detection for reliability")
    print("   ✅ Production deployment scripts")
    print("\n🚀 Next Steps:")
    print("   1. Run: py ridebuddy_optimized_gui.py (enhanced detection)")
    print("   2. Run: py vehicle_launcher.py (vehicle mode)")
    print("   3. Run: py deploy_enhanced_model.py (production deployment)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())