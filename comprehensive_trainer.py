#!/usr/bin/env python3
"""
RideBuddy Pro v2.1.0 - Comprehensive Training Pipeline
Trains models on comprehensive datasets for production deployment.
"""

import os
import sys
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
import random
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveDriverDataset(Dataset):
    """Enhanced dataset class for comprehensive driver monitoring data"""
    
    def __init__(self, samples, base_dir, transform=None, target_size=(224, 224)):
        self.samples = samples
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.target_size = target_size
        
        # Create class mapping
        unique_classes = sorted(set(sample['class'] for sample in samples))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.num_classes = len(unique_classes)
        
        print(f"📊 Dataset Info:")
        print(f"   Samples: {len(self.samples):,}")
        print(f"   Classes: {self.num_classes} -> {list(self.class_to_idx.keys())}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image_path = self.base_dir / sample['image_path']
        
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                # Create fallback image if loading fails
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, self.target_size)
        except Exception as e:
            logger.warning(f"Failed to load {image_path}: {e}")
            image = np.zeros((*self.target_size, 3), dtype=np.uint8)
        
        # Convert to tensor
        image = image.astype(np.float32) / 255.0
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Get label
        label = self.class_to_idx[sample['class']]
        
        return image, label

class EnhancedDriverNet(nn.Module):
    """Enhanced CNN architecture for driver monitoring"""
    
    def __init__(self, num_classes=5, dropout_rate=0.3):
        super(EnhancedDriverNet, self).__init__()
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Global pooling
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class ComprehensiveTrainer:
    """Comprehensive training pipeline for driver monitoring"""
    
    def __init__(self, output_dir="trained_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {self.device}")
        
        # Training configuration
        self.config = {
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_epochs": 100,
            "patience": 15,
            "min_delta": 0.001,
            "weight_decay": 1e-4,
            "step_size": 30,
            "gamma": 0.1
        }
        
    def load_comprehensive_datasets(self):
        """Load comprehensive datasets"""
        
        datasets_dir = Path("comprehensive_datasets")
        
        if not datasets_dir.exists():
            print("❌ Comprehensive datasets not found!")
            print("📦 Please run: py comprehensive_dataset_downloader.py")
            return None, None, None
        
        # Load metadata
        metadata_path = datasets_dir / "combined_metadata.json"
        samples_path = datasets_dir / "all_samples_metadata.json"
        
        if not metadata_path.exists() or not samples_path.exists():
            print("❌ Dataset metadata not found!")
            print("📦 Please run: py comprehensive_dataset_downloader.py")
            return None, None, None
        
        print("📊 Loading comprehensive datasets...")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        with open(samples_path, 'r') as f:
            all_samples = json.load(f)
        
        print(f"✅ Loaded {metadata['total_samples']:,} samples from {metadata['total_datasets']} datasets")
        print("📈 Class Distribution:")
        for class_name, count in metadata['class_distribution'].items():
            percentage = (count / metadata['total_samples']) * 100
            print(f"   {class_name}: {count:,} samples ({percentage:.1f}%)")
        print()
        
        # Split datasets
        train_samples, val_samples, test_samples = self.create_dataset_splits(all_samples)
        
        return train_samples, val_samples, test_samples
    
    def create_dataset_splits(self, all_samples, train_ratio=0.7, val_ratio=0.2):
        """Create balanced dataset splits"""
        
        # Group by class
        class_groups = {}
        for sample in all_samples:
            class_name = sample['class']
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(sample)
        
        train_samples = []
        val_samples = []
        test_samples = []
        
        for class_name, samples in class_groups.items():
            # Shuffle within class
            random.shuffle(samples)
            
            n_samples = len(samples)
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            
            train_samples.extend(samples[:n_train])
            val_samples.extend(samples[n_train:n_train + n_val])
            test_samples.extend(samples[n_train + n_val:])
        
        # Final shuffle
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)
        
        print(f"📊 Dataset Splits:")
        print(f"   Training: {len(train_samples):,} samples")
        print(f"   Validation: {len(val_samples):,} samples")
        print(f"   Testing: {len(test_samples):,} samples")
        print()
        
        return train_samples, val_samples, test_samples
    
    def create_data_loaders(self, train_samples, val_samples, test_samples):
        """Create data loaders with appropriate transforms"""
        
        # Training transforms with augmentation
        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validation/test transforms (no augmentation)
        val_test_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = ComprehensiveDriverDataset(
            train_samples, "comprehensive_datasets", transform=train_transforms
        )
        val_dataset = ComprehensiveDriverDataset(
            val_samples, "comprehensive_datasets", transform=val_test_transforms
        )
        test_dataset = ComprehensiveDriverDataset(
            test_samples, "comprehensive_datasets", transform=val_test_transforms
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.config["batch_size"],
            shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config["batch_size"],
            shuffle=False, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config["batch_size"],
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader, train_dataset.class_to_idx
    
    def train_model(self, train_loader, val_loader, class_to_idx):
        """Train the enhanced model"""
        
        num_classes = len(class_to_idx)
        model = EnhancedDriverNet(num_classes=num_classes).to(self.device)
        
        print(f"🧠 Model Architecture:")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"   Classes: {num_classes}")
        print()
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config["step_size"],
            gamma=self.config["gamma"]
        )
        
        # Training tracking
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("🚀 Starting training...")
        print("=" * 60)
        
        for epoch in range(self.config["num_epochs"]):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']} [Train]")
            
            for batch_idx, (data, targets) in enumerate(train_pbar):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
                
                # Update progress bar
                train_acc = 100.0 * train_correct / train_total
                train_pbar.set_postfix({
                    'Loss': f'{train_loss/(batch_idx+1):.4f}',
                    'Acc': f'{train_acc:.2f}%'
                })
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']} [Val]", leave=False)
                
                for data, targets in val_pbar:
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()
            
            # Calculate epoch metrics
            epoch_train_loss = train_loss / len(train_loader)
            epoch_val_loss = val_loss / len(val_loader)
            epoch_train_acc = 100.0 * train_correct / train_total
            epoch_val_acc = 100.0 * val_correct / val_total
            
            # Store metrics
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            train_accuracies.append(epoch_train_acc)
            val_accuracies.append(epoch_val_acc)
            
            # Update scheduler
            scheduler.step()
            
            # Print epoch results
            print(f"Epoch {epoch+1:3d} | "
                  f"Train Loss: {epoch_train_loss:.4f} | "
                  f"Val Loss: {epoch_val_loss:.4f} | "
                  f"Train Acc: {epoch_train_acc:6.2f}% | "
                  f"Val Acc: {epoch_val_acc:6.2f}% | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if epoch_val_loss < best_val_loss - self.config["min_delta"]:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                
                # Save best model
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': epoch_val_loss,
                    'val_accuracy': epoch_val_acc,
                    'class_to_idx': class_to_idx,
                    'config': self.config
                }
                
                torch.save(checkpoint, self.output_dir / 'best_enhanced_model.pth')
                
            else:
                patience_counter += 1
                
                if patience_counter >= self.config["patience"]:
                    print(f"\n⏹️  Early stopping triggered at epoch {epoch+1}")
                    break
        
        print("\n🎯 Training completed!")
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'config': self.config,
            'class_to_idx': class_to_idx
        }
        
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        return model, history
    
    def evaluate_model(self, model, test_loader, class_to_idx):
        """Comprehensive model evaluation"""
        
        print("📊 Evaluating model on test set...")
        
        model.eval()
        all_predictions = []
        all_targets = []
        test_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, targets in tqdm(test_loader, desc="Testing"):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        test_accuracy = 100.0 * correct / total
        test_loss = test_loss / len(test_loader)
        
        print(f"\n🎯 Test Results:")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.2f}%")
        
        # Detailed classification report
        idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        
        report = classification_report(
            all_targets, all_predictions,
            target_names=class_names,
            output_dict=True
        )
        
        print(f"\n📈 Classification Report:")
        print(classification_report(all_targets, all_predictions, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Save evaluation results
        evaluation_results = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'class_names': class_names
        }
        
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, class_names)
        
        return evaluation_results
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot and save confusion matrix"""
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names
        )
        plt.title('Enhanced Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confusion matrix saved to: {self.output_dir / 'confusion_matrix.png'}")
    
    def create_deployment_package(self, class_to_idx):
        """Create deployment-ready model package"""
        
        print("📦 Creating deployment package...")
        
        deployment_dir = self.output_dir / "deployment_package"
        deployment_dir.mkdir(exist_ok=True)
        
        # Load best model
        checkpoint = torch.load(self.output_dir / 'best_enhanced_model.pth')
        
        # Create deployment info
        deployment_info = {
            'model_version': '2.1.0',
            'creation_date': datetime.now().isoformat(),
            'num_classes': len(class_to_idx),
            'class_to_idx': class_to_idx,
            'input_size': [224, 224],
            'model_architecture': 'EnhancedDriverNet',
            'training_config': checkpoint['config'],
            'best_val_accuracy': checkpoint['val_accuracy'],
            'deployment_notes': 'Enhanced model trained on comprehensive datasets for production deployment'
        }
        
        # Save deployment files
        torch.save(checkpoint, deployment_dir / 'enhanced_driver_model.pth')
        
        with open(deployment_dir / 'deployment_info.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        print(f"✅ Deployment package created at: {deployment_dir}")
        print("📁 Package contents:")
        print("   - enhanced_driver_model.pth")
        print("   - deployment_info.json")
        
        return deployment_dir

def main():
    """Main training pipeline execution"""
    
    print("🚀 RideBuddy Pro v2.1.0 - Comprehensive Training Pipeline")
    print("=" * 70)
    
    # Initialize trainer
    trainer = ComprehensiveTrainer()
    
    # Load datasets
    print("📊 Step 1: Loading comprehensive datasets...")
    train_samples, val_samples, test_samples = trainer.load_comprehensive_datasets()
    
    if train_samples is None:
        return 1
    
    # Create data loaders
    print("🔄 Step 2: Creating data loaders...")
    train_loader, val_loader, test_loader, class_to_idx = trainer.create_data_loaders(
        train_samples, val_samples, test_samples
    )
    
    # Train model
    print("🧠 Step 3: Training enhanced model...")
    model, history = trainer.train_model(train_loader, val_loader, class_to_idx)
    
    # Evaluate model
    print("📊 Step 4: Evaluating model...")
    evaluation_results = trainer.evaluate_model(model, test_loader, class_to_idx)
    
    # Create deployment package
    print("📦 Step 5: Creating deployment package...")
    deployment_dir = trainer.create_deployment_package(class_to_idx)
    
    # Final summary
    print("\n🎉 Training Pipeline Complete!")
    print("=" * 70)
    print(f"✅ Model trained on {len(train_samples):,} samples")
    print(f"📊 Test accuracy: {evaluation_results['test_accuracy']:.2f}%")
    print(f"📦 Deployment package: {deployment_dir}")
    print()
    print("🔄 Next Steps:")
    print("1. Deploy enhanced model:")
    print("   py model_integration.py")
    print()
    print("2. Test integrated system:")
    print("   py ridebuddy_optimized_gui.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())