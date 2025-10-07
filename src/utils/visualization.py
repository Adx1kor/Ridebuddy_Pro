"""
Visualization utilities for RideBuddy predictions and analysis
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import torch
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def create_prediction_overlay(
    frame: np.ndarray, 
    prediction: Dict[str, Union[str, float, Dict]],
    confidence_threshold: float = 0.5
) -> np.ndarray:
    """
    Create overlay on video frame with prediction results
    
    Args:
        frame: Input video frame (BGR format)
        prediction: Prediction dictionary containing results
        confidence_threshold: Minimum confidence to show prediction
        
    Returns:
        Frame with prediction overlay
    """
    overlay_frame = frame.copy()
    height, width = frame.shape[:2]
    
    # Define colors for different states
    colors = {
        'normal': (0, 255, 0),      # Green
        'drowsy': (0, 0, 255),      # Red
        'phone_distraction': (0, 165, 255)  # Orange
    }
    
    # Get prediction info
    predicted_class = prediction.get('predicted_class', 'unknown')
    confidence = prediction.get('confidence', 0.0)
    
    # Only show if confidence is above threshold
    if confidence >= confidence_threshold:
        color = colors.get(predicted_class, (255, 255, 255))
        
        # Main prediction text
        main_text = f"{predicted_class.upper()}: {confidence:.2f}"
        font_scale = 0.8
        thickness = 2
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            main_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Draw background rectangle
        cv2.rectangle(
            overlay_frame, 
            (10, 10), 
            (10 + text_width + 20, 10 + text_height + baseline + 20),
            color, 
            -1
        )
        
        # Draw main prediction text
        cv2.putText(
            overlay_frame,
            main_text,
            (20, 10 + text_height + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness
        )
        
        # Add auxiliary predictions if available
        y_offset = 10 + text_height + baseline + 40
        
        auxiliary_predictions = prediction.get('auxiliary_predictions', {})
        
        if 'phone_detected' in auxiliary_predictions:
            phone_detected = auxiliary_predictions['phone_detected']
            phone_confidence = auxiliary_predictions.get('phone_confidence', 0.0)
            phone_text = f"Phone: {'YES' if phone_detected else 'NO'} ({phone_confidence:.2f})"
            phone_color = (0, 165, 255) if phone_detected else (0, 255, 0)
            
            cv2.putText(
                overlay_frame,
                phone_text,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                phone_color,
                2
            )
            y_offset += 30
        
        if 'seatbelt_worn' in auxiliary_predictions:
            seatbelt_worn = auxiliary_predictions['seatbelt_worn']
            seatbelt_confidence = auxiliary_predictions.get('seatbelt_confidence', 0.0)
            seatbelt_text = f"Seatbelt: {'YES' if seatbelt_worn else 'NO'} ({seatbelt_confidence:.2f})"
            seatbelt_color = (0, 255, 0) if seatbelt_worn else (0, 0, 255)
            
            cv2.putText(
                overlay_frame,
                seatbelt_text,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                seatbelt_color,
                2
            )
        
        # Add status indicator (circle in top-right corner)
        circle_center = (width - 30, 30)
        cv2.circle(overlay_frame, circle_center, 20, color, -1)
        cv2.circle(overlay_frame, circle_center, 20, (255, 255, 255), 2)
    
    # Add timestamp if available
    inference_time = prediction.get('inference_time', 0.0)
    if inference_time > 0:
        time_text = f"Inference: {inference_time*1000:.1f}ms"
        cv2.putText(
            overlay_frame,
            time_text,
            (width - 200, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return overlay_frame


def create_confidence_bar(
    class_probabilities: Dict[str, float],
    width: int = 300,
    height: int = 200
) -> np.ndarray:
    """
    Create confidence bar chart for class probabilities
    
    Args:
        class_probabilities: Dictionary of class names to probabilities
        width: Width of the chart
        height: Height of the chart
        
    Returns:
        Chart as numpy array
    """
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    classes = list(class_probabilities.keys())
    probs = list(class_probabilities.values())
    
    colors = ['green' if cls == 'normal' else 
              'red' if cls == 'drowsy' else 
              'orange' for cls in classes]
    
    bars = ax.barh(classes, probs, color=colors, alpha=0.7)
    
    # Add value labels
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{prob:.3f}', va='center', fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_xlabel('Confidence')
    ax.set_title('Classification Confidence')
    ax.grid(True, alpha=0.3)
    
    # Convert to numpy array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return buf


def plot_training_history(
    history: Dict[str, List[float]], 
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training history curves
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss curves
    if 'train_loss' in history and 'val_loss' in history:
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot accuracy curves
    if 'train_acc' in history and 'val_acc' in history:
        epochs = range(1, len(history['train_acc']) + 1)
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_model_architecture_diagram(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    save_path: Optional[str] = None
) -> str:
    """
    Create model architecture diagram
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        save_path: Path to save the diagram
        
    Returns:
        Path to saved diagram or diagram text
    """
    try:
        from torchviz import make_dot
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Forward pass to get computation graph
        output = model(dummy_input)
        
        # Create visualization
        if isinstance(output, dict):
            # Multi-output model
            dot = make_dot(output['classification'], params=dict(model.named_parameters()))
        else:
            dot = make_dot(output, params=dict(model.named_parameters()))
        
        if save_path:
            dot.render(save_path, format='png', cleanup=True)
            return f"{save_path}.png"
        else:
            return dot.source
            
    except ImportError:
        logger.warning("torchviz not available. Cannot create architecture diagram.")
        return "Architecture diagram requires torchviz package."


def visualize_feature_maps(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    layer_name: str,
    num_features: int = 16,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize intermediate feature maps from a specific layer
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor
        layer_name: Name of layer to visualize
        num_features: Number of feature maps to show
        save_path: Path to save the visualization
        
    Returns:
        Matplotlib figure
    """
    # Register forward hook to capture feature maps
    feature_maps = {}
    
    def hook_fn(module, input, output):
        feature_maps[layer_name] = output.detach()
    
    # Find and register hook
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook_fn)
            break
    else:
        raise ValueError(f"Layer '{layer_name}' not found in model")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Remove hook
    handle.remove()
    
    # Get feature maps
    if layer_name not in feature_maps:
        raise RuntimeError(f"Feature maps not captured for layer '{layer_name}'")
    
    features = feature_maps[layer_name][0]  # First sample in batch
    num_features = min(num_features, features.shape[0])
    
    # Create visualization
    cols = 4
    rows = (num_features + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_features):
        row = i // cols
        col = i % cols
        
        feature_map = features[i].cpu().numpy()
        axes[row, col].imshow(feature_map, cmap='viridis')
        axes[row, col].set_title(f'Feature {i}')
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(num_features, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Feature Maps from {layer_name}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_dataset_analysis_plots(
    dataset_analysis: Dict,
    save_dir: Optional[str] = None
) -> List[plt.Figure]:
    """
    Create analysis plots for dataset statistics
    
    Args:
        dataset_analysis: Dataset analysis results
        save_dir: Directory to save plots
        
    Returns:
        List of matplotlib figures
    """
    figures = []
    
    # 1. Overall class distribution
    if 'class_distribution' in dataset_analysis:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        classes = list(dataset_analysis['class_distribution'].keys())
        counts = list(dataset_analysis['class_distribution'].values())
        
        bars = ax1.bar(classes, counts, color=['green', 'red', 'orange'])
        ax1.set_title('Overall Class Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Number of Videos')
        
        # Add count labels
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + count*0.01,
                    str(count), ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_dir:
            fig1.savefig(f"{save_dir}/class_distribution.png", dpi=300, bbox_inches='tight')
        
        figures.append(fig1)
    
    # 2. Split-wise distribution
    if 'splits' in dataset_analysis:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        
        splits = list(dataset_analysis['splits'].keys())
        class_names = ['normal', 'drowsy', 'phone_distraction']
        
        x = np.arange(len(splits))
        width = 0.25
        
        for i, class_name in enumerate(class_names):
            counts = [
                dataset_analysis['splits'][split].get('classes', {}).get(class_name, 0)
                for split in splits
            ]
            color = 'green' if class_name == 'normal' else 'red' if class_name == 'drowsy' else 'orange'
            ax2.bar(x + i * width, counts, width, label=class_name, color=color, alpha=0.7)
        
        ax2.set_xlabel('Dataset Split')
        ax2.set_ylabel('Number of Videos')
        ax2.set_title('Class Distribution by Dataset Split')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(splits)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_dir:
            fig2.savefig(f"{save_dir}/split_distribution.png", dpi=300, bbox_inches='tight')
        
        figures.append(fig2)
    
    return figures


def create_performance_comparison_chart(
    model_results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'avg_inference_time_ms', 'model_size_mb'],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create performance comparison chart for different models
    
    Args:
        model_results: Dictionary of model results
        metrics: List of metrics to compare
        save_path: Path to save the chart
        
    Returns:
        Matplotlib figure
    """
    models = list(model_results.keys())
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        values = [model_results[model].get(metric, 0) for model in models]
        
        bars = axes[i].bar(models, values, alpha=0.7)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel(metric)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.2f}', ha='center', va='bottom')
        
        # Rotate x-axis labels if needed
        if len(max(models, key=len)) > 10:
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Test visualization functions
    
    # Test prediction overlay
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_prediction = {
        'predicted_class': 'phone_distraction',
        'confidence': 0.85,
        'class_probabilities': {
            'normal': 0.1,
            'drowsy': 0.05,
            'phone_distraction': 0.85
        },
        'auxiliary_predictions': {
            'phone_detected': True,
            'phone_confidence': 0.92,
            'seatbelt_worn': True,
            'seatbelt_confidence': 0.78
        },
        'inference_time': 0.045
    }
    
    overlay_frame = create_prediction_overlay(dummy_frame, dummy_prediction)
    print(f"Created overlay frame with shape: {overlay_frame.shape}")
    
    # Test confidence bar
    confidence_chart = create_confidence_bar(dummy_prediction['class_probabilities'])
    print(f"Created confidence chart with shape: {confidence_chart.shape}")
    
    print("Visualization utilities test completed!")
