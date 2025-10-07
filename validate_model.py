# RideBuddy Model Validation Script
# Validates model performance against validation dataset

import torch
import numpy as np
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import logging

from src.models.ridebuddy_model import create_lightweight_model
from src.data.dataset import create_dataloaders
from src.utils.metrics import MetricsCalculator
from src.utils.visualization import plot_confusion_matrix, create_performance_comparison_chart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_model(
    model_path: str,
    data_dir: str,
    device: str = 'cpu',
    batch_size: int = 32
) -> Dict:
    """
    Validate model performance on test dataset
    
    Args:
        model_path: Path to trained model
        data_dir: Path to dataset directory
        device: Device to run validation on
        batch_size: Batch size for validation
        
    Returns:
        Validation results
    """
    logger.info("Loading model and data...")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get('config', {}).get('model', {})
    
    model = create_lightweight_model(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load test data
    dataloaders = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0  # Set to 0 for validation
    )
    
    if 'test' not in dataloaders:
        logger.error("Test dataset not found")
        return {}
    
    test_loader = dataloaders['test']
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator()
    
    # Run validation
    logger.info("Running validation...")
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    inference_times = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            frames = batch['frames'].to(device)
            labels = batch['label'].to(device)
            
            start_time = time.time()
            outputs = model(frames)
            inference_time = time.time() - start_time
            
            inference_times.append(inference_time / frames.size(0))  # Per sample
            
            # Get predictions
            if isinstance(outputs, dict):
                logits = outputs['classification']
            else:
                logits = outputs
            
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            if batch_idx % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    
    classification_metrics = metrics_calc.calculate_metrics(
        all_targets, all_predictions, np.array(all_probabilities)
    )
    
    inference_metrics = metrics_calc.calculate_inference_metrics(
        inference_times, model.get_model_size()
    )
    
    # Combine results
    results = {
        'classification_metrics': classification_metrics,
        'inference_metrics': inference_metrics,
        'model_info': {
            'parameters': model.count_parameters(),
            'model_size_mb': model.get_model_size(),
            'config': model_config
        },
        'test_samples': len(all_targets)
    }
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("VALIDATION RESULTS SUMMARY")
    logger.info("="*50)
    logger.info(f"Test Samples: {len(all_targets)}")
    logger.info(f"Accuracy: {classification_metrics['accuracy']:.4f}")
    logger.info(f"Weighted F1: {classification_metrics['weighted_f1']:.4f}")
    logger.info(f"Avg Inference Time: {inference_metrics['avg_inference_time_ms']:.2f}ms")
    logger.info(f"FPS: {inference_metrics['fps']:.1f}")
    logger.info(f"Model Size: {results['model_info']['model_size_mb']:.2f}MB")
    logger.info(f"Parameters: {results['model_info']['parameters']:,}")
    
    # Per-class metrics
    class_names = ['normal', 'drowsy', 'phone_distraction']
    logger.info("\nPer-Class Metrics:")
    for class_name in class_names:
        precision = classification_metrics.get(f'{class_name}_precision', 0)
        recall = classification_metrics.get(f'{class_name}_recall', 0)
        f1 = classification_metrics.get(f'{class_name}_f1', 0)
        logger.info(f"  {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Validate RideBuddy model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Device for validation')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for validation')
    parser.add_argument('--output_dir', type=str, default='validation_results',
                       help='Directory to save validation results')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save validation plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run validation
    results = validate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        device=args.device,
        batch_size=args.batch_size
    )
    
    if not results:
        logger.error("Validation failed")
        return
    
    # Save results
    results_file = output_path / 'validation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Create and save plots if requested
    if args.save_plots:
        logger.info("Creating validation plots...")
        
        # This would require loading the predictions again
        # Implementation depends on your specific needs
        logger.info("Plot generation not implemented in this version")
    
    logger.info("Validation completed successfully!")


if __name__ == "__main__":
    main()
