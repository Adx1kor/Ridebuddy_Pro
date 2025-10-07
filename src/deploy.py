"""
Model deployment utilities for RideBuddy
Handles model optimization and deployment preparation
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import json
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import logging

from src.models.ridebuddy_model import create_lightweight_model
from src.utils.model_optimization import ModelOptimizer
from src.utils.metrics import calculate_model_efficiency_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeploymentPackager:
    """
    Package models for deployment with all necessary components
    """
    
    def __init__(self, output_dir: str = 'deployment'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def package_pytorch_model(
        self,
        model_path: str,
        config_path: str = None,
        include_sample_data: bool = True
    ) -> str:
        """
        Package PyTorch model for deployment
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to model configuration
            include_sample_data: Whether to include sample input data
            
        Returns:
            Path to deployment package
        """
        logger.info("Packaging PyTorch model for deployment...")
        
        # Create deployment directory
        deployment_dir = self.output_dir / 'pytorch_deployment'
        deployment_dir.mkdir(exist_ok=True)
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        model_config = checkpoint.get('config', {}).get('model', {})
        
        # Create model and load weights
        model = create_lightweight_model(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Save optimized model
        optimized_model_path = deployment_dir / 'model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'class_names': ['normal', 'drowsy', 'phone_distraction']
        }, optimized_model_path)
        
        # Create model info
        model_info = {
            'model_type': 'pytorch',
            'input_shape': [1, 3, 224, 224],
            'output_shape': {'classification': [1, 3]},
            'class_names': ['normal', 'drowsy', 'phone_distraction'],
            'preprocessing': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'input_size': 224
            },
            'model_config': model_config
        }
        
        with open(deployment_dir / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Create inference script template
        self._create_inference_template(deployment_dir, 'pytorch')
        
        # Include sample data if requested
        if include_sample_data:
            sample_input = torch.randn(1, 3, 224, 224)
            torch.save(sample_input, deployment_dir / 'sample_input.pth')
        
        logger.info(f"PyTorch deployment package created: {deployment_dir}")
        return str(deployment_dir)
    
    def package_onnx_model(
        self,
        model_path: str,
        onnx_path: str,
        include_sample_data: bool = True
    ) -> str:
        """
        Package ONNX model for deployment
        
        Args:
            model_path: Path to original PyTorch model
            onnx_path: Path to ONNX model
            include_sample_data: Whether to include sample input data
            
        Returns:
            Path to deployment package
        """
        logger.info("Packaging ONNX model for deployment...")
        
        # Create deployment directory
        deployment_dir = self.output_dir / 'onnx_deployment'
        deployment_dir.mkdir(exist_ok=True)
        
        # Copy ONNX model
        import shutil
        shutil.copy2(onnx_path, deployment_dir / 'model.onnx')
        
        # Load original model for config
        checkpoint = torch.load(model_path, map_location='cpu')
        model_config = checkpoint.get('config', {}).get('model', {})
        
        # Create model info
        model_info = {
            'model_type': 'onnx',
            'input_shape': [1, 3, 224, 224],
            'output_shape': {'classification': [1, 3]},
            'class_names': ['normal', 'drowsy', 'phone_distraction'],
            'preprocessing': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'input_size': 224
            },
            'model_config': model_config
        }
        
        with open(deployment_dir / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Create inference script template
        self._create_inference_template(deployment_dir, 'onnx')
        
        # Include sample data if requested
        if include_sample_data:
            sample_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
            np.save(deployment_dir / 'sample_input.npy', sample_input)
        
        logger.info(f"ONNX deployment package created: {deployment_dir}")
        return str(deployment_dir)
    
    def _create_inference_template(self, deployment_dir: Path, model_type: str):
        """Create inference script template"""
        
        if model_type == 'pytorch':
            template = '''
import torch
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path

# Add your model definition here or import it
# from your_model_module import RideBuddyModel

class RideBuddyInference:
    def __init__(self, model_path, model_info_path):
        # Load model info
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        # Initialize your model here
        # self.model = RideBuddyModel(**checkpoint['model_config'])
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.model.eval()
        
        self.class_names = self.model_info['class_names']
        self.preprocessing = self.model_info['preprocessing']
    
    def preprocess(self, image):
        """Preprocess input image"""
        # Implement preprocessing based on self.preprocessing
        # Normalize with mean and std
        mean = np.array(self.preprocessing['mean'])
        std = np.array(self.preprocessing['std'])
        image = (image / 255.0 - mean) / std
        return torch.from_numpy(image).float().unsqueeze(0)
    
    def predict(self, image):
        """Make prediction on image"""
        input_tensor = self.preprocess(image)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Process outputs
        if isinstance(outputs, dict):
            classification_logits = outputs['classification']
        else:
            classification_logits = outputs
        
        probabilities = F.softmax(classification_logits, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class_idx].item()
        
        return {
            'predicted_class': self.class_names[predicted_class_idx],
            'confidence': confidence,
            'probabilities': probabilities[0].tolist()
        }

# Usage example:
# inference = RideBuddyInference('model.pth', 'model_info.json')
# result = inference.predict(your_image_array)
'''
        
        elif model_type == 'onnx':
            template = '''
import onnxruntime as ort
import json
import numpy as np

class RideBuddyONNXInference:
    def __init__(self, model_path, model_info_path):
        # Load model info
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        # Create ONNX Runtime session
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        self.class_names = self.model_info['class_names']
        self.preprocessing = self.model_info['preprocessing']
    
    def preprocess(self, image):
        """Preprocess input image"""
        # Implement preprocessing based on self.preprocessing
        mean = np.array(self.preprocessing['mean']).reshape(1, 3, 1, 1)
        std = np.array(self.preprocessing['std']).reshape(1, 3, 1, 1)
        image = (image / 255.0 - mean) / std
        return image.astype(np.float32)
    
    def predict(self, image):
        """Make prediction on image"""
        input_tensor = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Process outputs (assuming first output is classification)
        classification_logits = outputs[0]
        
        # Apply softmax
        exp_logits = np.exp(classification_logits - np.max(classification_logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        predicted_class_idx = np.argmax(probabilities, axis=1)[0]
        confidence = probabilities[0, predicted_class_idx]
        
        return {
            'predicted_class': self.class_names[predicted_class_idx],
            'confidence': float(confidence),
            'probabilities': probabilities[0].tolist()
        }

# Usage example:
# inference = RideBuddyONNXInference('model.onnx', 'model_info.json')
# result = inference.predict(your_image_array)
'''
        
        template_path = deployment_dir / 'inference_template.py'
        with open(template_path, 'w') as f:
            f.write(template.strip())


def optimize_for_deployment(
    model_path: str,
    output_dir: str,
    target_device: str = 'cpu',
    optimization_level: str = 'medium'
) -> Dict[str, str]:
    """
    Optimize model for deployment
    
    Args:
        model_path: Path to trained model
        output_dir: Output directory for optimized models
        target_device: Target deployment device
        optimization_level: Level of optimization (light, medium, aggressive)
        
    Returns:
        Dictionary of optimized model paths
    """
    logger.info(f"Optimizing model for {target_device} deployment...")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint.get('config', {}).get('model', {})
    
    model = create_lightweight_model(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize optimizer
    optimizer = ModelOptimizer()
    
    # Define optimization strategies
    optimizations = {
        'light': ['onnx'],
        'medium': ['quantize', 'onnx'],
        'aggressive': ['prune', 'quantize', 'onnx']
    }
    
    optimization_list = optimizations.get(optimization_level, ['onnx'])
    
    # Create dummy calibration data
    calibration_data = [(torch.randn(4, 3, 224, 224), torch.randint(0, 3, (4,))) for _ in range(10)]
    calibration_loader = torch.utils.data.DataLoader(calibration_data, batch_size=4)
    
    # Run optimization pipeline
    results = optimizer.optimize_model_pipeline(
        model=model,
        calibration_data=calibration_loader,
        input_shape=(1, 3, 224, 224),
        output_dir=output_dir,
        optimizations=optimization_list
    )
    
    return results


def benchmark_deployment_models(
    model_paths: Dict[str, str],
    num_iterations: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different deployment models
    
    Args:
        model_paths: Dictionary mapping model types to paths
        num_iterations: Number of benchmark iterations
        
    Returns:
        Benchmark results
    """
    logger.info("Benchmarking deployment models...")
    
    results = {}
    input_shape = (1, 3, 224, 224)
    
    for model_type, model_path in model_paths.items():
        logger.info(f"Benchmarking {model_type} model...")
        
        if model_type == 'pytorch':
            # Benchmark PyTorch model
            checkpoint = torch.load(model_path, map_location='cpu')
            model_config = checkpoint.get('model_config', {})
            
            model = create_lightweight_model(**model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            optimizer = ModelOptimizer()
            benchmark_results = optimizer.benchmark_model(model, input_shape, num_iterations)
            
        elif model_type == 'onnx':
            # Benchmark ONNX model
            optimizer = ModelOptimizer()
            benchmark_results = optimizer.benchmark_onnx_model(model_path, input_shape, num_iterations)
        
        else:
            logger.warning(f"Unknown model type: {model_type}")
            continue
        
        results[model_type] = benchmark_results
        
        # Calculate efficiency score
        accuracy = 0.95  # Placeholder - should be from validation
        efficiency_score = calculate_model_efficiency_score(
            accuracy=accuracy,
            inference_time_ms=benchmark_results['avg_inference_time_ms'],
            model_size_mb=10.0  # Placeholder - should be calculated
        )
        
        results[model_type]['efficiency_score'] = efficiency_score
        
        logger.info(f"{model_type} benchmark: {benchmark_results['avg_inference_time_ms']:.2f}ms avg")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Deploy RideBuddy model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='deployment',
                       help='Output directory for deployment files')
    parser.add_argument('--target_device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Target deployment device')
    parser.add_argument('--optimization_level', type=str, default='medium',
                       choices=['light', 'medium', 'aggressive'],
                       help='Level of model optimization')
    parser.add_argument('--package_only', action='store_true',
                       help='Only create deployment packages without optimization')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark optimized models')
    
    args = parser.parse_args()
    
    # Initialize packager
    packager = DeploymentPackager(args.output_dir)
    
    if args.package_only:
        # Create deployment packages only
        pytorch_package = packager.package_pytorch_model(args.model_path)
        logger.info(f"PyTorch package created: {pytorch_package}")
        return
    
    # Optimize model for deployment
    optimization_results = optimize_for_deployment(
        model_path=args.model_path,
        output_dir=args.output_dir + '/optimized',
        target_device=args.target_device,
        optimization_level=args.optimization_level
    )
    
    # Create deployment packages
    if optimization_results.get('original_model'):
        pytorch_package = packager.package_pytorch_model(
            optimization_results['original_model']
        )
    
    if optimization_results.get('onnx_model'):
        onnx_package = packager.package_onnx_model(
            args.model_path,
            optimization_results['onnx_model']
        )
    
    # Benchmark models if requested
    if args.benchmark:
        model_paths = {}
        
        if optimization_results.get('original_model'):
            model_paths['pytorch'] = optimization_results['original_model']
        
        if optimization_results.get('quantized_model'):
            model_paths['quantized'] = optimization_results['quantized_model']
        
        if optimization_results.get('onnx_model'):
            model_paths['onnx'] = optimization_results['onnx_model']
        
        benchmark_results = benchmark_deployment_models(model_paths)
        
        # Save benchmark results
        benchmark_file = Path(args.output_dir) / 'benchmark_results.json'
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        logger.info(f"Benchmark results saved: {benchmark_file}")
    
    logger.info("Model deployment preparation completed!")


if __name__ == "__main__":
    main()
