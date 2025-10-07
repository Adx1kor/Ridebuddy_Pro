"""
Example usage script for RideBuddy driver monitoring system
Demonstrates model training, validation, and inference workflows
"""

import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_training_workflow():
    """
    Example training workflow for RideBuddy model
    """
    logger.info("="*60)
    logger.info("RIDEBUDDY TRAINING WORKFLOW EXAMPLE")
    logger.info("="*60)
    
    # Step 1: Prepare dataset
    logger.info("Step 1: Dataset Preparation")
    logger.info("Command: python src/data/prepare_dataset.py --source_dir raw_videos --output_dir data")
    logger.info("This organizes your raw video files into train/val/test splits")
    logger.info("")
    
    # Step 2: Train model
    logger.info("Step 2: Model Training")
    logger.info("Command: python src/train.py --config configs/lightweight_model.yaml --data_dir data --output_dir models")
    logger.info("This trains the lightweight RideBuddy model using the prepared dataset")
    logger.info("")
    
    # Step 3: Validate model
    logger.info("Step 3: Model Validation")
    logger.info("Command: python validate_model.py --model_path models/best_model.pth --data_dir data")
    logger.info("This validates the trained model on the test set")
    logger.info("")
    
    # Step 4: Deploy model
    logger.info("Step 4: Model Deployment")
    logger.info("Command: python src/deploy.py --model_path models/best_model.pth --optimization_level medium")
    logger.info("This optimizes and packages the model for deployment")
    logger.info("")


def example_inference_workflow():
    """
    Example inference workflow for RideBuddy model
    """
    logger.info("="*60)
    logger.info("RIDEBUDDY INFERENCE WORKFLOW EXAMPLE")
    logger.info("="*60)
    
    # Single video inference
    logger.info("Single Video Inference:")
    logger.info("Command: python src/inference.py --model models/best_model.pth --input video.mp4 --output output.mp4 --show_viz")
    logger.info("")
    
    # Batch inference
    logger.info("Batch Inference:")
    logger.info("Command: python src/inference.py --model models/best_model.pth --input video_folder --output results --batch")
    logger.info("")
    
    # Real-time webcam inference
    logger.info("Real-time Webcam Inference:")
    logger.info("Command: python src/inference.py --model models/best_model.pth --webcam --save_output")
    logger.info("")


def check_project_structure():
    """
    Check if all required project files exist
    """
    logger.info("="*60)
    logger.info("PROJECT STRUCTURE CHECK")
    logger.info("="*60)
    
    required_files = [
        'requirements.txt',
        'README.md',
        'src/models/ridebuddy_model.py',
        'src/data/dataset.py',
        'src/train.py',
        'src/inference.py',
        'configs/lightweight_model.yaml',
        'validate_model.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            logger.error(f"❌ Missing: {file_path}")
        else:
            logger.info(f"✅ Found: {file_path}")
    
    if missing_files:
        logger.error(f"\n{len(missing_files)} files are missing. Please ensure all required files are present.")
        return False
    else:
        logger.info(f"\n✅ All {len(required_files)} required files are present!")
        return True


def show_model_specifications():
    """
    Display model specifications and performance targets
    """
    logger.info("="*60)
    logger.info("RIDEBUDDY MODEL SPECIFICATIONS")
    logger.info("="*60)
    
    specs = {
        "Model Architecture": "Lightweight CNN with Multi-task Head",
        "Backbone Options": "MobileNetV3-Small, EfficientNet-B0",
        "Input Size": "224x224x3 RGB images",
        "Output Classes": "3 (normal, drowsy, phone_distraction)",
        "Auxiliary Tasks": "Phone detection, Seatbelt detection",
        "Target Accuracy": ">95%",
        "Target Inference Time": "<50ms on CPU",
        "Target Model Size": "<10MB",
        "Memory Usage": "<512MB RAM",
        "Supported Formats": "PyTorch (.pth), ONNX (.onnx)"
    }
    
    for key, value in specs.items():
        logger.info(f"{key:<25}: {value}")
    
    logger.info("\nKey Features:")
    features = [
        "Real-time video processing",
        "Multi-condition robustness (lighting, drivers, positions)",
        "Edge device optimization",
        "Minimal false positive rate",
        "Temporal consistency (optional LSTM module)"
    ]
    
    for feature in features:
        logger.info(f"  • {feature}")


def show_dataset_requirements():
    """
    Display dataset structure and requirements
    """
    logger.info("="*60)
    logger.info("DATASET REQUIREMENTS")
    logger.info("="*60)
    
    logger.info("Expected Dataset Structure:")
    logger.info("""
data/
├── train/
│   ├── normal/          # Videos of alert, focused driving
│   ├── drowsy/          # Videos showing drowsiness signs
│   └── phone_distraction/  # Videos of phone usage while driving
├── val/
│   ├── normal/
│   ├── drowsy/
│   └── phone_distraction/
└── test/
    ├── normal/
    ├── drowsy/
    └── phone_distraction/
""")
    
    logger.info("Video Requirements:")
    requirements = [
        "Format: MP4, AVI, MOV, or MKV",
        "Resolution: Minimum 480p, recommended 720p+",
        "Frame Rate: 15-30 FPS",
        "Duration: 5-60 seconds per clip",
        "Total Videos: ~500 (as mentioned in problem statement)",
        "Content: In-cabin driver footage showing various scenarios"
    ]
    
    for req in requirements:
        logger.info(f"  • {req}")
    
    logger.info("\nScenario Coverage:")
    scenarios = [
        "Different drivers (various ages, genders)",
        "Different lighting conditions (day/night/mixed)",
        "Various phone positions (ear, lap, dashboard)",
        "Different hand positions (left/right/both)",
        "Occlusions (steering wheel, hands)",
        "Seatbelt worn/not worn cases"
    ]
    
    for scenario in scenarios:
        logger.info(f"  • {scenario}")


def main():
    """
    Main function to run all examples
    """
    print("\n🚗 RideBuddy Driver Monitoring System")
    print("=====================================")
    
    # Check project structure
    if not check_project_structure():
        logger.error("Please fix missing files before proceeding.")
        return
    
    print("\n")
    
    # Show model specifications
    show_model_specifications()
    
    print("\n")
    
    # Show dataset requirements
    show_dataset_requirements()
    
    print("\n")
    
    # Show training workflow
    example_training_workflow()
    
    # Show inference workflow
    example_inference_workflow()
    
    logger.info("="*60)
    logger.info("NEXT STEPS")
    logger.info("="*60)
    logger.info("1. Prepare your video dataset using the structure shown above")
    logger.info("2. Install dependencies: pip install -r requirements.txt")
    logger.info("3. Organize dataset: python src/data/prepare_dataset.py --source_dir <your_videos> --output_dir data")
    logger.info("4. Train model: python src/train.py --config configs/lightweight_model.yaml --data_dir data")
    logger.info("5. Validate model: python validate_model.py --model_path models/best_model.pth --data_dir data")
    logger.info("6. Deploy model: python src/deploy.py --model_path models/best_model.pth")
    logger.info("")
    logger.info("For real-time testing: python src/inference.py --model models/best_model.pth --webcam")
    logger.info("")
    logger.info("🎯 Good luck with your RideBuddy implementation!")


if __name__ == "__main__":
    main()
