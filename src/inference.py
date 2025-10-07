"""
Inference script for RideBuddy driver monitoring model
Supports real-time video processing and batch inference
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json

from src.models.ridebuddy_model import create_lightweight_model, TemporalModel
from src.data.dataset import VideoProcessor, get_transforms
from src.utils.visualization import create_prediction_overlay
from src.utils.model_optimization import optimize_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RideBuddyInference:
    """
    Inference engine for RideBuddy driver monitoring
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cpu',
        confidence_threshold: float = 0.5,
        optimize_for_inference: bool = True
    ):
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Optimize model for inference
        if optimize_for_inference:
            self.model = optimize_model(self.model, device)
        
        # Class labels
        self.class_names = ['normal', 'drowsy', 'phone_distraction']
        
        # Video processor
        self.video_processor = VideoProcessor(target_fps=5, frame_size=(224, 224))
        
        # Transform for preprocessing
        self.transform = get_transforms('test', image_size=224)
        
        # Performance tracking
        self.inference_times = []
        
        logger.info("RideBuddy inference engine initialized")
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained model from checkpoint"""
        logger.info(f"Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        # Create model with same configuration
        model_config = config.get('model', {})
        model = create_lightweight_model(**model_config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess single frame for inference
        
        Args:
            frame: Input frame as numpy array (H, W, 3)
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Apply transforms
        transformed = self.transform(image=frame)
        frame_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        return frame_tensor.to(self.device)
    
    def predict_frame(self, frame: np.ndarray) -> Dict[str, Union[str, float, Dict]]:
        """
        Predict on single frame
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Prediction results dictionary
        """
        start_time = time.time()
        
        # Preprocess frame
        frame_tensor = self.preprocess_frame(frame)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(frame_tensor)
        
        # Process classification output
        classification_logits = outputs['classification']
        classification_probs = F.softmax(classification_logits, dim=1)
        predicted_class_idx = torch.argmax(classification_probs, dim=1).item()
        confidence = classification_probs[0, predicted_class_idx].item()
        predicted_class = self.class_names[predicted_class_idx]
        
        # Process auxiliary outputs if available
        auxiliary_predictions = {}
        
        if 'phone_detection' in outputs:
            phone_probs = torch.sigmoid(outputs['phone_detection'])
            phone_confidence = phone_probs[0, 1].item()  # Probability of phone present
            auxiliary_predictions['phone_detected'] = phone_confidence > self.confidence_threshold
            auxiliary_predictions['phone_confidence'] = phone_confidence
        
        if 'seatbelt_detection' in outputs:
            seatbelt_probs = torch.sigmoid(outputs['seatbelt_detection'])
            seatbelt_confidence = seatbelt_probs[0, 1].item()  # Probability of seatbelt worn
            auxiliary_predictions['seatbelt_worn'] = seatbelt_confidence > self.confidence_threshold
            auxiliary_predictions['seatbelt_confidence'] = seatbelt_confidence
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_probabilities': {
                name: prob for name, prob in zip(self.class_names, classification_probs[0].tolist())
            },
            'auxiliary_predictions': auxiliary_predictions,
            'inference_time': inference_time
        }
    
    def predict_video(
        self, 
        video_path: str, 
        output_path: Optional[str] = None,
        show_visualization: bool = False
    ) -> List[Dict]:
        """
        Predict on video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video with predictions
            show_visualization: Whether to show real-time visualization
            
        Returns:
            List of prediction results for each frame
        """
        logger.info(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Initialize video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        predictions = []
        frame_count = 0
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Make prediction
            prediction = self.predict_frame(frame_rgb)
            prediction['frame_number'] = frame_count
            predictions.append(prediction)
            
            # Create visualization
            if show_visualization or output_path:
                annotated_frame = create_prediction_overlay(frame, prediction)
                
                if show_visualization:
                    cv2.imshow('RideBuddy Predictions', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if writer:
                    writer.write(annotated_frame)
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if show_visualization:
            cv2.destroyAllWindows()
        
        logger.info(f"Video processing completed. Processed {len(predictions)} frames")
        
        return predictions
    
    def predict_webcam(self, camera_id: int = 0, save_output: bool = False):
        """
        Real-time prediction from webcam
        
        Args:
            camera_id: Camera device ID
            save_output: Whether to save output video
        """
        logger.info("Starting webcam inference...")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize video writer if saving output
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"webcam_output_{timestamp}.mp4"
            writer = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
        
        logger.info("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB for model
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Make prediction
                prediction = self.predict_frame(frame_rgb)
                
                # Create visualization
                annotated_frame = create_prediction_overlay(frame, prediction)
                
                # Display frame
                cv2.imshow('RideBuddy Live Monitoring', annotated_frame)
                
                # Save frame if requested
                if writer:
                    writer.write(annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    save_path = f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(save_path, annotated_frame)
                    logger.info(f"Frame saved to {save_path}")
                
                frame_count += 1
                
                # Log performance periodically
                if frame_count % 100 == 0:
                    avg_inference_time = np.mean(self.inference_times[-100:])
                    logger.info(f"Average inference time: {avg_inference_time*1000:.2f}ms")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            logger.info("Webcam inference stopped")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get inference performance statistics"""
        if not self.inference_times:
            return {}
        
        times_ms = np.array(self.inference_times) * 1000  # Convert to milliseconds
        
        return {
            'avg_inference_time_ms': float(np.mean(times_ms)),
            'min_inference_time_ms': float(np.min(times_ms)),
            'max_inference_time_ms': float(np.max(times_ms)),
            'std_inference_time_ms': float(np.std(times_ms)),
            'total_frames_processed': len(self.inference_times),
            'fps': 1000.0 / float(np.mean(times_ms))
        }
    
    def batch_inference(
        self, 
        input_dir: str, 
        output_dir: str, 
        save_predictions: bool = True
    ) -> Dict[str, List[Dict]]:
        """
        Run batch inference on multiple videos
        
        Args:
            input_dir: Directory containing input videos
            output_dir: Directory to save outputs
            save_predictions: Whether to save prediction results to JSON
            
        Returns:
            Dictionary mapping video names to their predictions
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_path.glob(f'*{ext}'))
        
        logger.info(f"Found {len(video_files)} video files for batch inference")
        
        all_predictions = {}
        
        for video_file in video_files:
            logger.info(f"Processing {video_file.name}...")
            
            try:
                # Run inference
                predictions = self.predict_video(
                    str(video_file),
                    output_path / f"{video_file.stem}_annotated.mp4"
                )
                
                all_predictions[video_file.name] = predictions
                
                # Save predictions to JSON
                if save_predictions:
                    pred_file = output_path / f"{video_file.stem}_predictions.json"
                    with open(pred_file, 'w') as f:
                        json.dump(predictions, f, indent=2)
                
                logger.info(f"Completed {video_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing {video_file.name}: {e}")
                continue
        
        # Save overall statistics
        stats = self.get_performance_stats()
        stats_file = output_path / "inference_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Batch inference completed")
        logger.info(f"Performance stats: {stats}")
        
        return all_predictions


def main():
    parser = argparse.ArgumentParser(description='RideBuddy Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input', type=str, help='Input video file or directory')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for real-time inference')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera device ID')
    parser.add_argument('--batch', action='store_true', help='Run batch inference on directory')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--save_output', action='store_true', help='Save output video')
    parser.add_argument('--show_viz', action='store_true', help='Show visualization')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference_engine = RideBuddyInference(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.confidence
    )
    
    if args.webcam:
        # Real-time webcam inference
        inference_engine.predict_webcam(
            camera_id=args.camera_id,
            save_output=args.save_output
        )
    
    elif args.batch and args.input:
        # Batch inference on directory
        if not args.output:
            args.output = Path(args.input).parent / "inference_outputs"
        
        inference_engine.batch_inference(
            input_dir=args.input,
            output_dir=args.output
        )
    
    elif args.input:
        # Single video inference
        predictions = inference_engine.predict_video(
            video_path=args.input,
            output_path=args.output,
            show_visualization=args.show_viz
        )
        
        # Save predictions
        if args.output:
            pred_file = Path(args.output).with_suffix('.json')
            with open(pred_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            logger.info(f"Predictions saved to {pred_file}")
    
    else:
        parser.print_help()
        return
    
    # Print performance statistics
    stats = inference_engine.get_performance_stats()
    logger.info("Performance Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()
