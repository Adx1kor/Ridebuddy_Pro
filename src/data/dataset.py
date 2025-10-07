"""
Data preprocessing and augmentation utilities for RideBuddy
Handles video data loading, frame extraction, and augmentation
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from PIL import Image

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Utility class for video processing operations"""
    
    def __init__(self, target_fps: int = 5, frame_size: Tuple[int, int] = (224, 224)):
        self.target_fps = target_fps
        self.frame_size = frame_size
    
    def extract_frames(
        self, 
        video_path: str, 
        max_frames: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extract frames from video file
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of extracted frames as numpy arrays
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return frames
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling rate
        if fps > 0:
            frame_step = max(1, int(fps / self.target_fps))
        else:
            frame_step = 1
        
        frame_idx = 0
        extracted_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at target FPS
            if frame_idx % frame_step == 0:
                # Resize frame
                frame_resized = cv2.resize(frame, self.frame_size)
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                extracted_count += 1
                
                # Check max frames limit
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_idx += 1
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    
    def create_video_sequence(
        self, 
        frames: List[np.ndarray], 
        sequence_length: int = 16
    ) -> List[List[np.ndarray]]:
        """
        Create overlapping sequences from frames
        
        Args:
            frames: List of video frames
            sequence_length: Length of each sequence
            
        Returns:
            List of frame sequences
        """
        if len(frames) < sequence_length:
            # Pad with repeated frames if video is too short
            padding_needed = sequence_length - len(frames)
            if frames:
                frames.extend([frames[-1]] * padding_needed)
            else:
                return []
        
        sequences = []
        step_size = max(1, sequence_length // 4)  # 75% overlap
        
        for i in range(0, len(frames) - sequence_length + 1, step_size):
            sequence = frames[i:i + sequence_length]
            sequences.append(sequence)
        
        return sequences


class RideBuddyDataset(Dataset):
    """
    Dataset class for RideBuddy driver monitoring data
    Supports both single frame and video sequence processing
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        sequence_length: Optional[int] = None,
        transform: Optional[A.Compose] = None,
        target_fps: int = 5,
        frame_size: Tuple[int, int] = (224, 224)
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.transform = transform
        self.video_processor = VideoProcessor(target_fps, frame_size)
        
        # Class mapping
        self.class_to_idx = {
            'normal': 0,
            'drowsy': 1,
            'phone_distraction': 2
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Load dataset
        self.samples = self._load_dataset()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_dataset(self) -> List[Dict]:
        """Load dataset samples with metadata"""
        samples = []
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            logger.error(f"Split directory not found: {split_dir}")
            return samples
        
        # Iterate through class directories
        for class_name in self.class_to_idx.keys():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            # Find video files
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            for video_path in class_dir.glob('*'):
                if video_path.suffix.lower() in video_extensions:
                    sample = {
                        'video_path': str(video_path),
                        'class_name': class_name,
                        'class_idx': self.class_to_idx[class_name]
                    }
                    samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, str]]:
        sample = self.samples[idx]
        video_path = sample['video_path']
        class_idx = sample['class_idx']
        
        # Extract frames from video
        frames = self.video_processor.extract_frames(video_path)
        
        if not frames:
            # Return dummy data if video processing fails
            if self.sequence_length:
                frames_tensor = torch.zeros(self.sequence_length, 3, 224, 224)
            else:
                frames_tensor = torch.zeros(3, 224, 224)
            
            return {
                'frames': frames_tensor,
                'label': class_idx,
                'video_path': video_path
            }
        
        if self.sequence_length:
            # Process as video sequence
            sequences = self.video_processor.create_video_sequence(
                frames, self.sequence_length
            )
            
            if sequences:
                # Use first sequence (or random selection during training)
                selected_sequence = sequences[0]
                if self.split == 'train' and len(sequences) > 1:
                    selected_sequence = np.random.choice(sequences)
                
                # Apply transforms to each frame in sequence
                processed_frames = []
                for frame in selected_sequence:
                    if self.transform:
                        transformed = self.transform(image=frame)
                        frame_tensor = transformed['image']
                    else:
                        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    processed_frames.append(frame_tensor)
                
                frames_tensor = torch.stack(processed_frames)
            else:
                frames_tensor = torch.zeros(self.sequence_length, 3, 224, 224)
        else:
            # Process as single frame
            # Use middle frame or random frame during training
            if self.split == 'train':
                frame = np.random.choice(frames)
            else:
                frame = frames[len(frames) // 2]
            
            if self.transform:
                transformed = self.transform(image=frame)
                frames_tensor = transformed['image']
            else:
                frames_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        
        return {
            'frames': frames_tensor,
            'label': class_idx,
            'video_path': video_path
        }


def get_transforms(split: str = 'train', image_size: int = 224) -> A.Compose:
    """
    Get augmentation transforms for different splits
    
    Args:
        split: Dataset split (train/val/test)
        image_size: Target image size
        
    Returns:
        Albumentations compose object
    """
    if split == 'train':
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, 
                sat_shift_limit=20, 
                val_shift_limit=20, 
                p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    return transform


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    sequence_length: Optional[int] = None,
    image_size: int = 224
) -> Dict[str, DataLoader]:
    """
    Create data loaders for train, validation, and test splits
    
    Args:
        data_dir: Root directory containing data splits
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        sequence_length: Length of video sequences (None for single frames)
        image_size: Target image size
        
    Returns:
        Dictionary of data loaders for each split
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        transform = get_transforms(split, image_size)
        
        dataset = RideBuddyDataset(
            data_dir=data_dir,
            split=split,
            sequence_length=sequence_length,
            transform=transform
        )
        
        # Skip if no samples found
        if len(dataset) == 0:
            logger.warning(f"No samples found for {split} split")
            continue
        
        shuffle = (split == 'train')
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=shuffle
        )
        
        dataloaders[split] = dataloader
        logger.info(f"Created {split} dataloader with {len(dataset)} samples")
    
    return dataloaders


def analyze_dataset(data_dir: str) -> Dict:
    """
    Analyze dataset statistics and class distribution
    
    Args:
        data_dir: Root directory containing data splits
        
    Returns:
        Dataset analysis results
    """
    analysis = {
        'splits': {},
        'total_samples': 0,
        'class_distribution': {}
    }
    
    data_path = Path(data_dir)
    
    for split in ['train', 'val', 'test']:
        split_dir = data_path / split
        if not split_dir.exists():
            continue
        
        split_stats = {'total': 0, 'classes': {}}
        
        for class_name in ['normal', 'drowsy', 'phone_distraction']:
            class_dir = split_dir / class_name
            if class_dir.exists():
                video_count = len(list(class_dir.glob('*.mp4'))) + \
                             len(list(class_dir.glob('*.avi'))) + \
                             len(list(class_dir.glob('*.mov')))
                split_stats['classes'][class_name] = video_count
                split_stats['total'] += video_count
        
        analysis['splits'][split] = split_stats
        analysis['total_samples'] += split_stats['total']
    
    # Calculate overall class distribution
    for class_name in ['normal', 'drowsy', 'phone_distraction']:
        total_class = sum(
            analysis['splits'].get(split, {}).get('classes', {}).get(class_name, 0)
            for split in analysis['splits']
        )
        analysis['class_distribution'][class_name] = total_class
    
    return analysis


if __name__ == "__main__":
    # Test data loading
    data_dir = "data"  # Update with actual data directory
    
    if os.path.exists(data_dir):
        # Analyze dataset
        analysis = analyze_dataset(data_dir)
        print("Dataset Analysis:")
        print(json.dumps(analysis, indent=2))
        
        # Test data loaders
        dataloaders = create_dataloaders(
            data_dir=data_dir,
            batch_size=4,
            num_workers=0  # Set to 0 for testing
        )
        
        # Test a batch
        if 'train' in dataloaders:
            train_loader = dataloaders['train']
            batch = next(iter(train_loader))
            print(f"\nBatch shapes:")
            print(f"Frames: {batch['frames'].shape}")
            print(f"Labels: {batch['label'].shape}")
    else:
        print(f"Data directory not found: {data_dir}")
        print("Please update the data_dir path or create the dataset structure")
