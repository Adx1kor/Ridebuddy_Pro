<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# RideBuddy Project Instructions

This is a driver monitoring AI project focused on drowsiness and distraction classification. When working on this codebase:

## Project Context
- This is a lightweight AI model for distinguishing drowsiness from phone-related distractions
- Target deployment: Edge devices with limited compute resources
- Framework: PyTorch with YOLO integration
- Dataset: ~500 videos of drivers with various phone usage scenarios

## Code Standards
- Use PyTorch as the primary deep learning framework
- Prefer lightweight models (EfficientNet, MobileNet, YOLOv8n)
- Optimize for inference speed and model size
- Include comprehensive error handling and logging
- Write modular, testable code with clear documentation

## Model Requirements
- Multi-task learning: classification + object detection
- Support for video input processing
- Real-time inference capability (<50ms on CPU)
- Model size target: <10MB
- Memory usage: <512MB RAM

## Key Features to Implement
- Drowsiness vs distraction classification
- Phone usage detection (various holding styles)
- Seatbelt detection
- Robust handling of lighting variations and occlusions
- Model quantization and pruning for deployment

## File Organization
- Keep model architectures in `src/models/`
- Data processing utilities in `src/data/`
- Training scripts in `src/training/`
- Inference code in `src/inference/`
- Configuration files in `configs/`

## Performance Targets
- Classification accuracy: >95%
- Inference time: <50ms on CPU
- Handle different drivers, lighting conditions, and phone positions
- Minimize false positives for drowsiness detection
