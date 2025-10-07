"""
Training script for RideBuddy driver monitoring model
Supports multi-task learning with classification and object detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import wandb
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, List
import time
from tqdm import tqdm

from src.models.ridebuddy_model import create_lightweight_model, TemporalModel
from src.data.dataset import create_dataloaders
from src.utils.metrics import MetricsCalculator
from src.utils.early_stopping import EarlyStopping

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining classification and detection losses
    """
    
    def __init__(
        self, 
        classification_weight: float = 1.0,
        phone_detection_weight: float = 0.5,
        seatbelt_detection_weight: float = 0.3,
        focal_loss_alpha: float = 0.25,
        focal_loss_gamma: float = 2.0
    ):
        super().__init__()
        self.classification_weight = classification_weight
        self.phone_detection_weight = phone_detection_weight
        self.seatbelt_detection_weight = seatbelt_detection_weight
        
        # Focal loss for handling class imbalance
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss for handling class imbalance"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_loss_alpha * (1 - pt) ** self.focal_loss_gamma * ce_loss
        return focal_loss.mean()
    
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate multi-task loss
        
        Args:
            outputs: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Classification loss (primary task)
        if 'classification' in outputs and 'classification' in targets:
            losses['classification'] = self.focal_loss(
                outputs['classification'], 
                targets['classification']
            ) * self.classification_weight
        
        # Phone detection loss
        if 'phone_detection' in outputs and 'phone_detection' in targets:
            losses['phone_detection'] = self.bce_loss(
                outputs['phone_detection'], 
                targets['phone_detection'].float()
            ) * self.phone_detection_weight
        
        # Seatbelt detection loss
        if 'seatbelt_detection' in outputs and 'seatbelt_detection' in targets:
            losses['seatbelt_detection'] = self.bce_loss(
                outputs['seatbelt_detection'], 
                targets['seatbelt_detection'].float()
            ) * self.seatbelt_detection_weight
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


class RideBuddyTrainer:
    """
    Trainer class for RideBuddy model
    """
    
    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict[str, DataLoader],
        config: Dict,
        device: torch.device
    ):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.config = config
        self.device = device
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = MultiTaskLoss(**config.get('loss', {}))
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Metrics and monitoring
        self.metrics_calculator = MetricsCalculator()
        self.early_stopping = EarlyStopping(**config.get('early_stopping', {}))
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_path = Path(config['output_dir']) / 'best_model.pth'
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config"""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'AdamW')
        lr = optimizer_config.get('lr', 1e-3)
        weight_decay = optimizer_config.get('weight_decay', 1e-4)
        
        if optimizer_type == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_type == 'SGD':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'CosineAnnealingLR')
        
        if scheduler_type == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        pbar = tqdm(self.dataloaders['train'], desc='Training')
        
        for batch_idx, batch in enumerate(pbar):
            frames = batch['frames'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Create target dictionary (simplified for now)
            targets = {'classification': labels}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(frames)
                    losses = self.criterion(outputs, targets)
                    total_loss = losses['total']
            else:
                outputs = self.model(frames)
                losses = self.criterion(outputs, targets)
                total_loss = losses['total']
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()
            
            # Update metrics
            running_loss += total_loss.item()
            
            # Calculate accuracy
            if 'classification' in outputs:
                _, predicted = torch.max(outputs['classification'], 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Acc': f'{correct_predictions/total_predictions:.4f}' if total_predictions > 0 else '0.0000',
                'LR': f'{current_lr:.6f}'
            })
        
        epoch_loss = running_loss / len(self.dataloaders['train'])
        epoch_acc = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.dataloaders['val'], desc='Validation')
            
            for batch in pbar:
                frames = batch['frames'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Create target dictionary
                targets = {'classification': labels}
                
                # Forward pass
                outputs = self.model(frames)
                losses = self.criterion(outputs, targets)
                
                # Update metrics
                running_loss += losses['total'].item()
                
                if 'classification' in outputs:
                    _, predicted = torch.max(outputs['classification'], 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_predictions += labels.size(0)
                    
                    # Store for detailed metrics
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'Loss': f'{losses["total"].item():.4f}',
                    'Acc': f'{correct_predictions/total_predictions:.4f}' if total_predictions > 0 else '0.0000'
                })
        
        epoch_loss = running_loss / len(self.dataloaders['val'])
        epoch_acc = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        # Calculate detailed metrics
        if all_predictions and all_targets:
            detailed_metrics = self.metrics_calculator.calculate_metrics(
                all_targets, all_predictions
            )
        else:
            detailed_metrics = {}
        
        return {
            'loss': epoch_loss, 
            'accuracy': epoch_acc,
            'detailed_metrics': detailed_metrics
        }
    
    def train(self) -> Dict:
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            logger.info(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['accuracy'])
            else:
                self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Log metrics
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_model(self.best_model_path)
                logger.info(f"New best model saved with validation accuracy: {self.best_val_acc:.4f}")
            
            # Early stopping check
            if self.early_stopping(val_metrics['loss']):
                logger.info("Early stopping triggered")
                break
            
            # Log to wandb if configured
            if wandb.run:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return {
            'best_val_accuracy': self.best_val_acc,
            'training_time': training_time,
            'history': self.history
        }
    
    def save_model(self, path: Path):
        """Save model checkpoint"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'history': self.history
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train RideBuddy model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='models', help='Output directory')
    parser.add_argument('--wandb_project', type=str, default='ridebuddy', help='Wandb project name')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config['output_dir'] = args.output_dir
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize wandb if configured
    if config.get('use_wandb', False):
        wandb.init(
            project=args.wandb_project,
            config=config,
            name=config.get('experiment_name', 'ridebuddy_experiment')
        )
    
    # Create data loaders
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        **config.get('dataloader', {})
    )
    
    # Create model
    model = create_lightweight_model(**config.get('model', {}))
    
    # Initialize trainer
    trainer = RideBuddyTrainer(model, dataloaders, config, device)
    
    # Train model
    results = trainer.train()
    
    logger.info("Training completed successfully!")
    logger.info(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
    
    # Close wandb
    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
