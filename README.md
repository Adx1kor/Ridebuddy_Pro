# 🚗 RideBuddy Pro v2.1.0 - Production-Ready Driver Monitoring System

**Professional AI-Powered Driver Safety Solution with Enterprise Features**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0%2B-green.svg)](https://opencv.org)
[![Real-World Ready](https://img.shields.io/badge/Real--World-Ready-brightgreen.svg)](#)

A comprehensive, production-ready driver monitoring system that provides real-time detection of driver drowsiness, phone usage, and safety compliance with enterprise-grade features including data privacy, performance optimization, and comprehensive analytics.

## Project Overview

RideBuddy is an innovative driver safety solution that monitors driver behavior using in-cabin cameras. This secondary AI model runs in the cloud to verify and refine drowsiness detection by distinguishing genuine drowsy events from distractions caused by mobile phone usage.

## Features

- **Drowsiness vs Distraction Classification**: Accurately classifies driver states
- **Phone Usage Detection**: Detects various phone holding styles and usage patterns
- **Seatbelt Detection**: Monitors seatbelt compliance
- **Multi-condition Support**: Handles different lighting conditions, drivers, and occlusions
- **Lightweight Model**: Optimized for edge deployment with minimal compute requirements
- **Real-time Inference**: Suitable for in-vehicle deployment

## Model Architecture

The system uses a multi-task learning approach with:
- **Backbone**: EfficientNet-B0 or MobileNetV3 for feature extraction
- **YOLO Integration**: For object detection (phone, hands, seatbelt)
- **Classification Head**: For drowsiness/distraction classification
- **Optimization**: Model quantization and pruning for edge deployment

## Dataset Structure

```
data/
├── train/
│   ├── drowsy/
│   ├── phone_distraction/
│   └── normal/
├── val/
│   ├── drowsy/
│   ├── phone_distraction/
│   └── normal/
└── test/
    ├── drowsy/
    ├── phone_distraction/
    └── normal/
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**:
   ```bash
   python src/data/prepare_dataset.py --data_dir /path/to/videos
   ```

3. **Train Model**:
   ```bash
   python src/train.py --config configs/lightweight_model.yaml
   ```

4. **Run Inference**:
   ```bash
   python src/inference.py --model models/best_model.pth --input video.mp4
   ```

## Project Structure

```
RideBuddy/
├── src/
│   ├── models/           # Model architectures
│   ├── data/            # Data processing utilities
│   ├── training/        # Training scripts and utilities
│   ├── inference/       # Inference and deployment
│   └── utils/           # Helper functions
├── configs/             # Configuration files
├── models/              # Trained model files
├── data/               # Dataset directory
├── notebooks/          # Jupyter notebooks for analysis
├── tests/              # Unit tests
└── docs/               # Documentation
```

## Model Performance

Target specifications:
- **Accuracy**: >95% classification accuracy
- **Inference Time**: <50ms on CPU
- **Model Size**: <10MB
- **Memory Usage**: <512MB RAM

## Deployment

The model is optimized for:
- Edge devices with limited compute
- Real-time video processing
- Low power consumption
- Robust performance across different conditions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is proprietary to Bosch and developed for the RideBuddy driver monitoring solution.
