"""
Model optimization utilities for edge deployment
Includes quantization, pruning, and ONNX conversion
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.quantization import get_default_qconfig
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional, Dict, Any
import time

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    Utility class for model optimization techniques
    """
    
    def __init__(self):
        self.optimization_results = {}
    
    def quantize_model(
        self, 
        model: nn.Module, 
        calibration_data: torch.utils.data.DataLoader,
        quantization_type: str = 'dynamic'
    ) -> nn.Module:
        """
        Quantize model for faster inference
        
        Args:
            model: PyTorch model to quantize
            calibration_data: Data loader for calibration (for static quantization)
            quantization_type: Type of quantization ('dynamic', 'static', or 'qat')
            
        Returns:
            Quantized model
        """
        logger.info(f"Starting {quantization_type} quantization...")
        
        model.eval()
        
        if quantization_type == 'dynamic':
            # Dynamic quantization (no calibration needed)
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            
        elif quantization_type == 'static':
            # Static quantization (requires calibration)
            model.qconfig = get_default_qconfig('fbgemm')
            quantization.prepare(model, inplace=True)
            
            # Calibration
            logger.info("Running calibration...")
            with torch.no_grad():
                for i, batch in enumerate(calibration_data):
                    if i >= 100:  # Limit calibration samples
                        break
                    frames = batch['frames']
                    model(frames)
            
            quantized_model = quantization.convert(model, inplace=False)
            
        else:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")
        
        logger.info("Quantization completed")
        return quantized_model
    
    def prune_model(
        self, 
        model: nn.Module, 
        pruning_ratio: float = 0.3,
        structured: bool = False
    ) -> nn.Module:
        """
        Prune model to reduce parameters
        
        Args:
            model: PyTorch model to prune
            pruning_ratio: Fraction of parameters to prune
            structured: Whether to use structured pruning
            
        Returns:
            Pruned model
        """
        logger.info(f"Starting pruning with ratio {pruning_ratio}...")
        
        import torch.nn.utils.prune as prune
        
        # Get modules to prune
        modules_to_prune = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                modules_to_prune.append((module, 'weight'))
        
        if structured:
            # Structured pruning
            for module, param_name in modules_to_prune:
                prune.ln_structured(
                    module, 
                    name=param_name, 
                    amount=pruning_ratio, 
                    n=2, 
                    dim=0
                )
        else:
            # Unstructured pruning
            prune.global_unstructured(
                modules_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio,
            )
        
        # Make pruning permanent
        for module, param_name in modules_to_prune:
            prune.remove(module, param_name)
        
        logger.info("Pruning completed")
        return model
    
    def convert_to_onnx(
        self, 
        model: nn.Module, 
        input_shape: Tuple[int, ...],
        output_path: str,
        opset_version: int = 11
    ) -> str:
        """
        Convert PyTorch model to ONNX format
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape (including batch dimension)
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
            
        Returns:
            Path to saved ONNX model
        """
        logger.info("Converting model to ONNX...")
        
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"ONNX model saved to {output_path}")
        return output_path
    
    def benchmark_model(
        self, 
        model: nn.Module, 
        input_shape: Tuple[int, ...],
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        device: str = 'cpu'
    ) -> Dict[str, float]:
        """
        Benchmark model inference performance
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            num_iterations: Number of inference iterations
            warmup_iterations: Number of warmup iterations
            device: Device to run benchmark on
            
        Returns:
            Performance metrics
        """
        logger.info("Benchmarking model performance...")
        
        model.eval()
        device = torch.device(device)
        model.to(device)
        
        dummy_input = torch.randn(input_shape).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(dummy_input)
        
        # Benchmark
        inference_times = []
        
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.time()
                _ = model(dummy_input)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                inference_times.append(end_time - start_time)
        
        # Calculate statistics
        times_ms = np.array(inference_times) * 1000
        
        metrics = {
            'avg_inference_time_ms': float(np.mean(times_ms)),
            'min_inference_time_ms': float(np.min(times_ms)),
            'max_inference_time_ms': float(np.max(times_ms)),
            'std_inference_time_ms': float(np.std(times_ms)),
            'fps': 1000.0 / float(np.mean(times_ms)),
            'device': device.type
        }
        
        logger.info(f"Benchmark results: {metrics}")
        return metrics
    
    def optimize_model_pipeline(
        self, 
        model: nn.Module,
        calibration_data: torch.utils.data.DataLoader,
        input_shape: Tuple[int, ...],
        output_dir: str,
        optimizations: list = ['quantize', 'prune', 'onnx']
    ) -> Dict[str, Any]:
        """
        Complete model optimization pipeline
        
        Args:
            model: Original PyTorch model
            calibration_data: Calibration data loader
            input_shape: Input tensor shape
            output_dir: Directory to save optimized models
            optimizations: List of optimizations to apply
            
        Returns:
            Optimization results and model paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            'original_model': None,
            'quantized_model': None,
            'pruned_model': None,
            'onnx_model': None,
            'benchmarks': {}
        }
        
        # Benchmark original model
        logger.info("Benchmarking original model...")
        original_benchmark = self.benchmark_model(model, input_shape)
        results['benchmarks']['original'] = original_benchmark
        
        # Save original model
        original_path = output_path / 'original_model.pth'
        torch.save(model.state_dict(), original_path)
        results['original_model'] = str(original_path)
        
        current_model = model
        
        # Apply optimizations in sequence
        if 'prune' in optimizations:
            logger.info("Applying pruning...")
            current_model = self.prune_model(current_model.copy() if hasattr(current_model, 'copy') else current_model)
            
            # Save pruned model
            pruned_path = output_path / 'pruned_model.pth'
            torch.save(current_model.state_dict(), pruned_path)
            results['pruned_model'] = str(pruned_path)
            
            # Benchmark pruned model
            pruned_benchmark = self.benchmark_model(current_model, input_shape)
            results['benchmarks']['pruned'] = pruned_benchmark
        
        if 'quantize' in optimizations:
            logger.info("Applying quantization...")
            quantized_model = self.quantize_model(current_model, calibration_data)
            
            # Save quantized model
            quantized_path = output_path / 'quantized_model.pth'
            torch.save(quantized_model.state_dict(), quantized_path)
            results['quantized_model'] = str(quantized_path)
            
            # Benchmark quantized model
            quantized_benchmark = self.benchmark_model(quantized_model, input_shape)
            results['benchmarks']['quantized'] = quantized_benchmark
            
            current_model = quantized_model
        
        if 'onnx' in optimizations:
            logger.info("Converting to ONNX...")
            onnx_path = output_path / 'model.onnx'
            self.convert_to_onnx(current_model, input_shape, str(onnx_path))
            results['onnx_model'] = str(onnx_path)
            
            # Benchmark ONNX model
            onnx_benchmark = self.benchmark_onnx_model(str(onnx_path), input_shape)
            results['benchmarks']['onnx'] = onnx_benchmark
        
        # Generate optimization report
        self._generate_optimization_report(results, output_path / 'optimization_report.txt')
        
        return results
    
    def benchmark_onnx_model(
        self, 
        onnx_path: str, 
        input_shape: Tuple[int, ...],
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark ONNX model performance
        
        Args:
            onnx_path: Path to ONNX model
            input_shape: Input tensor shape
            num_iterations: Number of inference iterations
            
        Returns:
            Performance metrics
        """
        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path)
        
        # Prepare input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        input_name = session.get_inputs()[0].name
        
        # Warmup
        for _ in range(10):
            _ = session.run(None, {input_name: dummy_input})
        
        # Benchmark
        inference_times = []
        for _ in range(num_iterations):
            start_time = time.time()
            _ = session.run(None, {input_name: dummy_input})
            end_time = time.time()
            inference_times.append(end_time - start_time)
        
        times_ms = np.array(inference_times) * 1000
        
        return {
            'avg_inference_time_ms': float(np.mean(times_ms)),
            'min_inference_time_ms': float(np.min(times_ms)),
            'max_inference_time_ms': float(np.max(times_ms)),
            'std_inference_time_ms': float(np.std(times_ms)),
            'fps': 1000.0 / float(np.mean(times_ms)),
            'device': 'cpu'
        }
    
    def _generate_optimization_report(self, results: Dict[str, Any], report_path: Path):
        """Generate optimization report"""
        with open(report_path, 'w') as f:
            f.write("RideBuddy Model Optimization Report\n")
            f.write("=" * 40 + "\n\n")
            
            f.write("Model Files:\n")
            for key, path in results.items():
                if key.endswith('_model') and path:
                    f.write(f"  {key}: {path}\n")
            
            f.write("\nPerformance Benchmarks:\n")
            for model_type, benchmark in results['benchmarks'].items():
                f.write(f"\n{model_type.upper()} Model:\n")
                for metric, value in benchmark.items():
                    f.write(f"  {metric}: {value}\n")
            
            # Calculate improvements
            if 'original' in results['benchmarks'] and 'quantized' in results['benchmarks']:
                orig_time = results['benchmarks']['original']['avg_inference_time_ms']
                quant_time = results['benchmarks']['quantized']['avg_inference_time_ms']
                speedup = orig_time / quant_time
                f.write(f"\nQuantization Speedup: {speedup:.2f}x\n")
        
        logger.info(f"Optimization report saved to {report_path}")


def optimize_model(model: nn.Module, device: str = 'cpu') -> nn.Module:
    """
    Quick model optimization for inference
    
    Args:
        model: PyTorch model
        device: Target device
        
    Returns:
        Optimized model
    """
    model.eval()
    
    # Move to target device
    model = model.to(device)
    
    # Apply TorchScript optimization
    if device == 'cpu':
        # Optimize for CPU inference
        model = torch.jit.optimize_for_inference(torch.jit.script(model))
    
    return model


if __name__ == "__main__":
    # Test model optimization utilities
    from src.models.ridebuddy_model import create_lightweight_model
    
    # Create a test model
    model = create_lightweight_model(backbone='mobilenet_v3_small')
    
    # Initialize optimizer
    optimizer = ModelOptimizer()
    
    # Benchmark original model
    input_shape = (1, 3, 224, 224)
    benchmark_results = optimizer.benchmark_model(model, input_shape)
    
    print("Benchmark Results:")
    for key, value in benchmark_results.items():
        print(f"  {key}: {value}")
    
    # Test ONNX conversion
    onnx_path = "test_model.onnx"
    optimizer.convert_to_onnx(model, input_shape, onnx_path)
    print(f"Model converted to ONNX: {onnx_path}")
