"""
RideBuddy Dataset Organization Script
Organizes the hackathon dataset into proper train/val/test structure
"""

import os
import shutil
import random
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def organize_hackathon_dataset():
    """
    Organize the hackathon dataset into the required structure for RideBuddy training
    """
    logger.info("Starting hackathon dataset organization...")
    
    # Define source paths
    base_path = Path("data")
    hackathon_path = base_path / "hackathon_videos" / "Hackathon_data_track5"
    true_drowsy_path = base_path / "true_drowsy" / "True_drowsy_detection"
    
    # Define target structure
    target_path = base_path / "organized"
    
    # Create target directories
    for split in ['train', 'val', 'test']:
        for class_name in ['normal', 'drowsy', 'phone_distraction']:
            (target_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Collect video files by category
    video_collections = {
        'drowsy': [],
        'phone_distraction': [],
        'normal': []
    }
    
    # 1. True drowsy videos
    if hackathon_path.exists():
        drowsy_folder = hackathon_path / "True_drowsy_detection"
        if drowsy_folder.exists():
            for video_file in drowsy_folder.glob("*.mp4"):
                video_collections['drowsy'].append(video_file)
                logger.info(f"Found drowsy video: {video_file.name}")
    
    # Additional true drowsy videos
    if true_drowsy_path.exists():
        for video_file in true_drowsy_path.glob("*.mp4"):
            video_collections['drowsy'].append(video_file)
            logger.info(f"Found additional drowsy video: {video_file.name}")
    
    # 2. Phone distraction videos (False detections)
    if hackathon_path.exists():
        false_detection_path = hackathon_path / "False_detection"
        
        # Phone related distractions
        phone_folders = [
            "driver_looking_at_phone",
            "driver_looking_down"  # Often phone related
        ]
        
        for folder_name in phone_folders:
            folder_path = false_detection_path / folder_name
            if folder_path.exists():
                for video_file in folder_path.glob("*.mp4"):
                    video_collections['phone_distraction'].append(video_file)
                    logger.info(f"Found phone distraction video: {video_file.name}")
        
        # 3. Normal driving videos (other false detections that are not phone related)
        normal_folders = [
            "driver_turning_left_or_right",
            "hand_movement"
        ]
        
        for folder_name in normal_folders:
            folder_path = false_detection_path / folder_name
            if folder_path.exists():
                for video_file in folder_path.glob("*.mp4"):
                    video_collections['normal'].append(video_file)
                    logger.info(f"Found normal driving video: {video_file.name}")
    
    # Print collection summary
    logger.info("\nDataset Summary:")
    for class_name, videos in video_collections.items():
        logger.info(f"  {class_name}: {len(videos)} videos")
    
    # Split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    # Organize each class
    dataset_stats = {}
    
    for class_name, videos in video_collections.items():
        if not videos:
            logger.warning(f"No videos found for class: {class_name}")
            continue
        
        # Shuffle videos for random split
        random.shuffle(videos)
        
        # Calculate split sizes
        total_videos = len(videos)
        train_size = int(total_videos * train_ratio)
        val_size = int(total_videos * val_ratio)
        test_size = total_videos - train_size - val_size
        
        # Split videos
        train_videos = videos[:train_size]
        val_videos = videos[train_size:train_size + val_size]
        test_videos = videos[train_size + val_size:]
        
        # Copy videos to target directories
        for split, split_videos in [('train', train_videos), ('val', val_videos), ('test', test_videos)]:
            target_dir = target_path / split / class_name
            
            for video_file in split_videos:
                target_file = target_dir / video_file.name
                shutil.copy2(video_file, target_file)
                logger.info(f"Copied {video_file.name} to {split}/{class_name}/")
        
        dataset_stats[class_name] = {
            'train': len(train_videos),
            'val': len(val_videos),
            'test': len(test_videos),
            'total': total_videos
        }
        
        logger.info(f"Class '{class_name}': Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    
    # Save dataset info
    dataset_info = {
        'dataset_name': 'RideBuddy Hackathon Dataset',
        'organization_date': str(Path().cwd()),
        'splits': {
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio
        },
        'statistics': dataset_stats,
        'total_videos': sum(stats['total'] for stats in dataset_stats.values()),
        'class_mapping': {
            'normal': 0,
            'drowsy': 1,
            'phone_distraction': 2
        }
    }
    
    # Save dataset info
    with open(target_path / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    logger.info(f"\nDataset organization complete!")
    logger.info(f"Organized dataset saved to: {target_path}")
    logger.info(f"Total videos processed: {dataset_info['total_videos']}")
    
    return str(target_path), dataset_info


def validate_organized_dataset(dataset_path):
    """
    Validate the organized dataset structure
    """
    dataset_path = Path(dataset_path)
    
    logger.info("Validating organized dataset...")
    
    validation_results = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = dataset_path / split
        split_results = {}
        
        if not split_dir.exists():
            logger.error(f"Split directory missing: {split_dir}")
            continue
        
        for class_name in ['normal', 'drowsy', 'phone_distraction']:
            class_dir = split_dir / class_name
            
            if class_dir.exists():
                video_files = list(class_dir.glob('*.mp4'))
                split_results[class_name] = len(video_files)
                logger.info(f"  {split}/{class_name}: {len(video_files)} videos")
            else:
                split_results[class_name] = 0
                logger.warning(f"  {split}/{class_name}: Missing directory")
        
        validation_results[split] = split_results
    
    # Calculate totals
    total_videos = sum(sum(split_data.values()) for split_data in validation_results.values())
    logger.info(f"\nTotal videos in organized dataset: {total_videos}")
    
    return validation_results


if __name__ == "__main__":
    # Set random seed for reproducible splits
    random.seed(42)
    
    # Organize dataset
    organized_path, dataset_info = organize_hackathon_dataset()
    
    # Validate organized dataset
    validation_results = validate_organized_dataset(organized_path)
    
    print("\n" + "="*60)
    print("DATASET ORGANIZATION COMPLETE!")
    print("="*60)
    print(f"Organized dataset location: {organized_path}")
    print(f"Total videos: {dataset_info['total_videos']}")
    print("\nClass distribution:")
    for class_name, stats in dataset_info['statistics'].items():
        print(f"  {class_name}: {stats['total']} videos")
        print(f"    Train: {stats['train']}, Val: {stats['val']}, Test: {stats['test']}")
    
    print(f"\nDataset is ready for RideBuddy training!")
    print("Next step: Install Python dependencies and start training")
