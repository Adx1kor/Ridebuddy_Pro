"""
RideBuddy: Lightweight Multi-Task Model for Driver Monitoring
Combines drowsiness/distraction classification with object detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LightweightFeatureExtractor(nn.Module):
    """Lightweight feature extractor based on MobileNetV3 or EfficientNet"""
    
    def __init__(self, backbone: str = 'mobilenet_v3_small', pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        
        if backbone == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
            # Remove classifier layers
            self.backbone.classifier = nn.Identity()
            self.feature_dim = 576
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            self.backbone.classifier = nn.Identity()
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Adaptive pooling to ensure consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        # Handle different backbone architectures
        if len(features.shape) > 2:
            features = self.adaptive_pool(features)
            features = torch.flatten(features, 1)
        return features


class AttentionModule(nn.Module):
    """Lightweight attention mechanism for focusing on relevant regions"""
    
    def __init__(self, feature_dim: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // reduction, feature_dim, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiTaskHead(nn.Module):
    """Multi-task head for classification and detection tasks"""
    
    def __init__(self, feature_dim: int, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        
        # Classification head (drowsy, phone_distraction, normal)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout // 2),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        # Phone detection head (binary classification)
        self.phone_detector = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 2)  # phone present/absent
        )
        
        # Seatbelt detection head (binary classification)
        self.seatbelt_detector = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 2)  # seatbelt worn/not worn
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'classification': self.classifier(features),
            'phone_detection': self.phone_detector(features),
            'seatbelt_detection': self.seatbelt_detector(features)
        }


class RideBuddyModel(nn.Module):
    """
    Lightweight multi-task model for driver monitoring
    Combines feature extraction with multi-task prediction heads
    """
    
    def __init__(
        self,
        backbone: str = 'mobilenet_v3_small',
        num_classes: int = 3,
        dropout: float = 0.2,
        use_attention: bool = True,
        pretrained: bool = True
    ):
        super().__init__()
        
        # Feature extraction backbone
        self.feature_extractor = LightweightFeatureExtractor(backbone, pretrained)
        
        # Optional attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionModule(self.feature_extractor.feature_dim)
        
        # Multi-task prediction heads
        self.multi_task_head = MultiTaskHead(
            self.feature_extractor.feature_dim, 
            num_classes, 
            dropout
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for custom layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply multi-task heads
        outputs = self.multi_task_head(features)
        
        return outputs
    
    def get_model_size(self) -> float:
        """Calculate model size in MB"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
    
    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TemporalModel(nn.Module):
    """
    Temporal model for processing video sequences
    Uses LSTM to capture temporal dependencies in driver behavior
    """
    
    def __init__(
        self,
        base_model: RideBuddyModel,
        sequence_length: int = 16,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 1
    ):
        super().__init__()
        
        self.base_model = base_model
        self.sequence_length = sequence_length
        
        # Freeze base model for feature extraction
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=base_model.feature_extractor.feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0
        )
        
        # Temporal classification head
        self.temporal_classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(lstm_hidden_size // 2, 3)  # drowsy, distraction, normal
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for temporal model
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
        Returns:
            Classification logits
        """
        batch_size, seq_len, c, h, w = x.shape
        
        # Reshape for base model processing
        x = x.view(batch_size * seq_len, c, h, w)
        
        # Extract features using base model
        with torch.no_grad():
            features = self.base_model.feature_extractor(x)
        
        # Reshape back to sequence format
        features = features.view(batch_size, seq_len, -1)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(features)
        
        # Use final timestep output for classification
        final_output = lstm_out[:, -1, :]
        
        # Classification
        logits = self.temporal_classifier(final_output)
        
        return logits


def create_lightweight_model(
    backbone: str = 'mobilenet_v3_small',
    num_classes: int = 3,
    use_attention: bool = True,
    pretrained: bool = True
) -> RideBuddyModel:
    """
    Factory function to create a lightweight RideBuddy model
    
    Args:
        backbone: Backbone architecture name
        num_classes: Number of classification classes
        use_attention: Whether to use attention mechanism
        pretrained: Use pretrained weights
    
    Returns:
        Configured RideBuddy model
    """
    model = RideBuddyModel(
        backbone=backbone,
        num_classes=num_classes,
        use_attention=use_attention,
        pretrained=pretrained
    )
    
    logger.info(f"Created model with {model.count_parameters():,} parameters")
    logger.info(f"Model size: {model.get_model_size():.2f} MB")
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_lightweight_model()
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    outputs = model(dummy_input)
    
    print("Model outputs:")
    for task, output in outputs.items():
        print(f"  {task}: {output.shape}")
    
    print(f"\nModel parameters: {model.count_parameters():,}")
    print(f"Model size: {model.get_model_size():.2f} MB")
