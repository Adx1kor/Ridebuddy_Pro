"""
Data preparation script for RideBuddy dataset
Organizes videos into train/val/test splits with proper class structure
"""

import os
import shutil
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def organize_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, int]:
    """
    Organize raw video files into train/val/test splits
    
    Args:
        source_dir: Directory containing raw video files
        output_dir: Directory to create organized dataset
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation  
        test_ratio: Fraction of data for testing
        random_seed: Random seed for reproducible splits
        
    Returns:
        Dictionary with split statistics
    """
    random.seed(random_seed)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Create output directory structure
    splits = ['train', 'val', 'test']
    classes = ['normal', 'drowsy', 'phone_distraction']
    
    for split in splits:
        for class_name in classes:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(source_path.glob(f'**/*{ext}'))
    
    logger.info(f"Found {len(video_files)} video files")
    
    # Classify videos based on filename patterns or manual annotation
    classified_videos = classify_videos(video_files)
    
    # Split each class independently
    split_stats = {}
    
    for class_name, class_videos in classified_videos.items():
        if not class_videos:
            logger.warning(f"No videos found for class: {class_name}")
            continue
        
        # Shuffle videos
        random.shuffle(class_videos)
        
        # Calculate split sizes
        total_videos = len(class_videos)
        train_size = int(total_videos * train_ratio)
        val_size = int(total_videos * val_ratio)
        test_size = total_videos - train_size - val_size
        
        # Split videos
        train_videos = class_videos[:train_size]
        val_videos = class_videos[train_size:train_size + val_size]
        test_videos = class_videos[train_size + val_size:]
        
        split_stats[class_name] = {
            'train': len(train_videos),
            'val': len(val_videos),
            'test': len(test_videos)
        }
        
        # Copy videos to respective directories
        copy_videos(train_videos, output_path / 'train' / class_name)
        copy_videos(val_videos, output_path / 'val' / class_name)
        copy_videos(test_videos, output_path / 'test' / class_name)
        
        logger.info(f"Class '{class_name}': Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    
    # Save split information
    split_info = {
        'source_dir': str(source_dir),
        'output_dir': str(output_dir),
        'split_ratios': {'train': train_ratio, 'val': val_ratio, 'test': test_ratio},
        'random_seed': random_seed,
        'statistics': split_stats,
        'total_videos': sum(sum(class_stats.values()) for class_stats in split_stats.values())
    }
    
    with open(output_path / 'dataset_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    return split_stats


def classify_videos(video_files: List[Path]) -> Dict[str, List[Path]]:
    """
    Classify videos into categories based on filename patterns
    
    Args:
        video_files: List of video file paths
        
    Returns:
        Dictionary mapping class names to video file lists
    """
    classified = {
        'normal': [],
        'drowsy': [],
        'phone_distraction': []
    }
    
    # Keywords for classification (adjust based on your dataset naming)
    keywords = {
        'drowsy': ['drowsy', 'sleepy', 'tired', 'eyes_closed'],
        'phone_distraction': ['phone', 'mobile', 'texting', 'calling', 'distracted'],
        'normal': ['normal', 'alert', 'focused', 'attentive']
    }
    
    for video_file in video_files:
        filename_lower = video_file.name.lower()
        
        # Try to classify based on filename
        classified_flag = False
        
        for class_name, class_keywords in keywords.items():
            if any(keyword in filename_lower for keyword in class_keywords):
                classified[class_name].append(video_file)
                classified_flag = True
                break
        
        # If not classified by keywords, try directory structure
        if not classified_flag:
            parent_dir = video_file.parent.name.lower()
            
            for class_name, class_keywords in keywords.items():
                if any(keyword in parent_dir for keyword in class_keywords):
                    classified[class_name].append(video_file)
                    classified_flag = True
                    break
        
        # Default to normal if still not classified
        if not classified_flag:
            logger.warning(f"Could not classify video: {video_file}. Assigning to 'normal' class.")
            classified['normal'].append(video_file)
    
    return classified


def copy_videos(video_list: List[Path], destination_dir: Path):
    """
    Copy videos to destination directory
    
    Args:
        video_list: List of video file paths to copy
        destination_dir: Destination directory
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    
    for video_file in tqdm(video_list, desc=f"Copying to {destination_dir.name}"):
        destination_file = destination_dir / video_file.name
        
        # Handle filename conflicts
        counter = 1
        while destination_file.exists():
            stem = video_file.stem
            suffix = video_file.suffix
            destination_file = destination_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        shutil.copy2(video_file, destination_file)


def create_annotation_template(
    video_files: List[Path],
    output_file: str = 'annotation_template.json'
):
    """
    Create annotation template for manual labeling
    
    Args:
        video_files: List of video files to annotate
        output_file: Output JSON file for annotations
    """
    annotations = {}
    
    for video_file in video_files:
        annotations[str(video_file)] = {
            'classification': '',  # normal, drowsy, phone_distraction
            'phone_usage': {
                'present': False,
                'hand': '',  # left, right, both
                'position': ''  # ear, lap, dashboard
            },
            'seatbelt': {
                'worn': False,
                'visible': True
            },
            'lighting': '',  # day, night, mixed
            'driver_visibility': '',  # clear, partially_occluded, heavily_occluded
            'notes': ''
        }
    
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    logger.info(f"Annotation template created: {output_file}")


def validate_dataset(dataset_dir: str) -> Dict[str, Dict[str, int]]:
    """
    Validate dataset structure and count files
    
    Args:
        dataset_dir: Path to dataset directory
        
    Returns:
        Validation results
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    validation_results = {}
    splits = ['train', 'val', 'test']
    classes = ['normal', 'drowsy', 'phone_distraction']
    
    for split in splits:
        split_dir = dataset_path / split
        split_results = {}
        
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue
        
        for class_name in classes:
            class_dir = split_dir / class_name
            
            if class_dir.exists():
                video_files = list(class_dir.glob('*.mp4')) + \
                             list(class_dir.glob('*.avi')) + \
                             list(class_dir.glob('*.mov'))
                split_results[class_name] = len(video_files)
            else:
                split_results[class_name] = 0
        
        validation_results[split] = split_results
    
    # Print validation summary
    logger.info("Dataset Validation Summary:")
    for split, split_data in validation_results.items():
        total_split = sum(split_data.values())
        logger.info(f"  {split.upper()}: {total_split} videos")
        for class_name, count in split_data.items():
            logger.info(f"    {class_name}: {count}")
    
    total_videos = sum(sum(split_data.values()) for split_data in validation_results.values())
    logger.info(f"  TOTAL: {total_videos} videos")
    
    return validation_results


def main():
    parser = argparse.ArgumentParser(description='Prepare RideBuddy dataset')
    parser.add_argument('--source_dir', type=str, required=True, 
                       help='Directory containing raw video files')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory for organized dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Fraction of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Fraction of data for validation')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Fraction of data for testing')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducible splits')
    parser.add_argument('--validate_only', action='store_true',
                       help='Only validate existing dataset structure')
    parser.add_argument('--create_annotation_template', action='store_true',
                       help='Create annotation template for manual labeling')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    if args.validate_only:
        # Validate existing dataset
        validation_results = validate_dataset(args.output_dir)
        return
    
    if args.create_annotation_template:
        # Create annotation template
        source_path = Path(args.source_dir)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(source_path.glob(f'**/*{ext}'))
        
        create_annotation_template(video_files)
        return
    
    # Organize dataset
    logger.info("Starting dataset organization...")
    logger.info(f"Source directory: {args.source_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Split ratios - Train: {args.train_ratio}, Val: {args.val_ratio}, Test: {args.test_ratio}")
    
    split_stats = organize_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )
    
    # Validate the created dataset
    logger.info("Validating created dataset...")
    validation_results = validate_dataset(args.output_dir)
    
    logger.info("Dataset preparation completed successfully!")


if __name__ == "__main__":
    main()
