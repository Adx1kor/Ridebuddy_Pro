"""
Metrics calculation utilities for RideBuddy model evaluation
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score
)
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """
    Comprehensive metrics calculator for multi-class classification
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or ['normal', 'drowsy', 'phone_distraction']
        self.num_classes = len(self.class_names)
    
    def calculate_metrics(
        self, 
        y_true: List[int], 
        y_pred: List[int], 
        y_probs: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_probs: Prediction probabilities (optional)
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        # Basic accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, recall, F1-score per class and averaged
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name}_precision'] = precision[i] if i < len(precision) else 0.0
            metrics[f'{class_name}_recall'] = recall[i] if i < len(recall) else 0.0
            metrics[f'{class_name}_f1'] = f1[i] if i < len(f1) else 0.0
            metrics[f'{class_name}_support'] = support[i] if i < len(support) else 0
        
        # Averaged metrics
        avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['weighted_precision'] = avg_precision
        metrics['weighted_recall'] = avg_recall
        metrics['weighted_f1'] = avg_f1
        
        # Macro averaged metrics
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        metrics['macro_precision'] = macro_precision
        metrics['macro_recall'] = macro_recall
        metrics['macro_f1'] = macro_f1
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Class-wise accuracy from confusion matrix
        if cm.shape[0] == cm.shape[1]:  # Square matrix
            for i, class_name in enumerate(self.class_names):
                if i < cm.shape[0] and cm[i].sum() > 0:
                    metrics[f'{class_name}_accuracy'] = cm[i, i] / cm[i].sum()
        
        # AUC-ROC if probabilities provided
        if y_probs is not None and y_probs.shape[1] == self.num_classes:
            try:
                # Multi-class AUC (one-vs-rest)
                auc_scores = []
                for i in range(self.num_classes):
                    y_true_binary = (np.array(y_true) == i).astype(int)
                    if len(np.unique(y_true_binary)) > 1:  # Check if both classes present
                        auc = roc_auc_score(y_true_binary, y_probs[:, i])
                        auc_scores.append(auc)
                        metrics[f'{self.class_names[i]}_auc'] = auc
                
                if auc_scores:
                    metrics['mean_auc'] = np.mean(auc_scores)
            except Exception as e:
                print(f"AUC calculation failed: {e}")
        
        return metrics
    
    def calculate_inference_metrics(
        self, 
        inference_times: List[float], 
        model_size_mb: float
    ) -> Dict[str, float]:
        """
        Calculate inference performance metrics
        
        Args:
            inference_times: List of inference times in seconds
            model_size_mb: Model size in megabytes
            
        Returns:
            Dictionary of performance metrics
        """
        if not inference_times:
            return {}
        
        times_ms = np.array(inference_times) * 1000  # Convert to milliseconds
        
        return {
            'avg_inference_time_ms': float(np.mean(times_ms)),
            'median_inference_time_ms': float(np.median(times_ms)),
            'min_inference_time_ms': float(np.min(times_ms)),
            'max_inference_time_ms': float(np.max(times_ms)),
            'std_inference_time_ms': float(np.std(times_ms)),
            'p95_inference_time_ms': float(np.percentile(times_ms, 95)),
            'p99_inference_time_ms': float(np.percentile(times_ms, 99)),
            'fps': 1000.0 / float(np.mean(times_ms)),
            'model_size_mb': model_size_mb,
            'throughput_fps': len(inference_times) / sum(inference_times)
        }
    
    def plot_confusion_matrix(
        self, 
        y_true: List[int], 
        y_pred: List[int], 
        save_path: Optional[str] = None,
        normalize: bool = True
    ) -> plt.Figure:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
            normalize: Whether to normalize the matrix
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_class_distribution(
        self, 
        y_true: List[int], 
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot class distribution
        
        Args:
            y_true: True labels
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        unique, counts = np.unique(y_true, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar([self.class_names[i] for i in unique], counts)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + count*0.01,
                    str(count), ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def print_classification_report(
        self, 
        y_true: List[int], 
        y_pred: List[int]
    ) -> str:
        """
        Generate and print detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report as string
        """
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=self.class_names,
            digits=4
        )
        print(report)
        return report


class ModelComparisonMetrics:
    """
    Compare multiple models' performance
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or ['normal', 'drowsy', 'phone_distraction']
        self.metrics_calculator = MetricsCalculator(class_names)
    
    def compare_models(
        self, 
        model_results: Dict[str, Dict[str, List]], 
        save_path: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Compare multiple models' performance
        
        Args:
            model_results: Dictionary mapping model names to their results
                Format: {'model_name': {'y_true': [...], 'y_pred': [...], 'y_probs': [...]}}
            save_path: Path to save comparison plot
            
        Returns:
            Dictionary of comparison metrics
        """
        comparison = {}
        
        # Calculate metrics for each model
        for model_name, results in model_results.items():
            metrics = self.metrics_calculator.calculate_metrics(
                results['y_true'], 
                results['y_pred'], 
                results.get('y_probs')
            )
            comparison[model_name] = metrics
        
        # Create comparison plot
        if len(model_results) > 1:
            self._plot_model_comparison(comparison, save_path)
        
        return comparison
    
    def _plot_model_comparison(
        self, 
        comparison: Dict[str, Dict], 
        save_path: Optional[str] = None
    ):
        """Plot model comparison chart"""
        models = list(comparison.keys())
        metrics_to_plot = ['accuracy', 'weighted_f1', 'macro_f1']
        
        x = np.arange(len(models))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics_to_plot):
            values = [comparison[model][metric] for model in models]
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for i, metric in enumerate(metrics_to_plot):
            values = [comparison[model][metric] for model in models]
            for j, v in enumerate(values):
                ax.text(j + i * width, v + 0.01, f'{v:.3f}', 
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def calculate_model_efficiency_score(
    accuracy: float,
    inference_time_ms: float,
    model_size_mb: float,
    target_accuracy: float = 0.95,
    target_inference_time_ms: float = 50.0,
    target_model_size_mb: float = 10.0
) -> float:
    """
    Calculate model efficiency score balancing accuracy, speed, and size
    
    Args:
        accuracy: Model accuracy (0-1)
        inference_time_ms: Average inference time in milliseconds
        model_size_mb: Model size in megabytes
        target_accuracy: Target accuracy threshold
        target_inference_time_ms: Target inference time threshold
        target_model_size_mb: Target model size threshold
        
    Returns:
        Efficiency score (higher is better)
    """
    # Normalize metrics (0-1, where 1 is best)
    accuracy_score = min(accuracy / target_accuracy, 1.0)
    speed_score = min(target_inference_time_ms / inference_time_ms, 1.0)
    size_score = min(target_model_size_mb / model_size_mb, 1.0)
    
    # Weighted combination (accuracy is most important)
    efficiency_score = 0.5 * accuracy_score + 0.3 * speed_score + 0.2 * size_score
    
    return efficiency_score


if __name__ == "__main__":
    # Test metrics calculator
    calculator = MetricsCalculator()
    
    # Dummy data for testing
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 1]
    y_pred = [0, 1, 1, 0, 1, 2, 0, 2, 2, 1]
    
    metrics = calculator.calculate_metrics(y_true, y_pred)
    print("Calculated metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Test classification report
    print("\nClassification Report:")
    calculator.print_classification_report(y_true, y_pred)
