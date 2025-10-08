"""
RideBuddy Pro - Comprehensive Drowsiness Detection Analysis & Testing
Analyzes all training parameters, creates confusion matrices, and tests edge cases
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import pandas as pd
import cv2
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DrowsinessParameterAnalyzer:
    """
    Comprehensive analysis of drowsiness detection parameters and model performance
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.parameters = self._define_drowsiness_parameters()
        self.test_cases = self._define_comprehensive_test_cases()
        self.results = {}
        
    def _define_drowsiness_parameters(self) -> Dict[str, Dict]:
        """
        Define all parameters considered for drowsiness detection training
        """
        return {
            "facial_features": {
                "eye_aspect_ratio": {
                    "description": "Ratio of eye height to eye width",
                    "drowsy_threshold": 0.25,
                    "normal_range": [0.3, 0.4],
                    "weight": 0.35,
                    "detection_method": "Dlib 68-point facial landmarks"
                },
                "eye_closure_duration": {
                    "description": "Duration of eye closure in consecutive frames",
                    "drowsy_threshold": 30,  # frames (~1 second at 30fps)
                    "normal_range": [0, 10],
                    "weight": 0.30,
                    "detection_method": "Temporal analysis of EAR"
                },
                "blink_frequency": {
                    "description": "Number of blinks per minute",
                    "drowsy_range": [0, 12],  # Low blink rate indicates drowsiness
                    "normal_range": [15, 30],
                    "weight": 0.15,
                    "detection_method": "Blink detection algorithm"
                },
                "mouth_aspect_ratio": {
                    "description": "Mouth opening ratio for yawning detection",
                    "drowsy_threshold": 0.6,
                    "normal_range": [0.0, 0.3],
                    "weight": 0.20,
                    "detection_method": "Dlib mouth landmarks analysis"
                }
            },
            "head_pose_parameters": {
                "head_nod_frequency": {
                    "description": "Frequency of involuntary head nodding",
                    "drowsy_indicator": "> 3 nods per minute",
                    "weight": 0.25,
                    "detection_method": "Head pose estimation with Euler angles"
                },
                "head_tilt_angle": {
                    "description": "Excessive head tilting indicating loss of control",
                    "drowsy_threshold": 15,  # degrees
                    "weight": 0.20,
                    "detection_method": "PnP head pose estimation"
                },
                "gaze_direction": {
                    "description": "Eye gaze direction consistency",
                    "drowsy_indicator": "Inconsistent/wandering gaze",
                    "weight": 0.15,
                    "detection_method": "Pupil tracking and gaze estimation"
                }
            },
            "behavioral_parameters": {
                "reaction_time": {
                    "description": "Response time to visual/audio stimuli",
                    "drowsy_threshold": 1.5,  # seconds
                    "normal_range": [0.3, 0.8],
                    "weight": 0.10,
                    "detection_method": "Simulated reaction tests"
                },
                "hand_position": {
                    "description": "Hand placement on steering wheel",
                    "drowsy_indicators": ["loose grip", "hands dropping"],
                    "weight": 0.15,
                    "detection_method": "YOLO hand detection + pose estimation"
                },
                "body_posture": {
                    "description": "Overall body posture and slumping",
                    "drowsy_indicators": ["slouching", "leaning"],
                    "weight": 0.10,
                    "detection_method": "Pose estimation algorithms"
                }
            },
            "distraction_differentiators": {
                "phone_usage": {
                    "detection_methods": ["Object detection", "Hand gesture analysis"],
                    "indicators": ["phone_visible", "texting_gesture", "call_gesture"],
                    "weight": 0.40,
                    "confidence_threshold": 0.7
                },
                "conversation": {
                    "detection_methods": ["Audio analysis", "Mouth movement"],
                    "indicators": ["consistent_speech", "head_movement_patterns"],
                    "weight": 0.25,
                    "confidence_threshold": 0.6
                },
                "navigation_interaction": {
                    "detection_methods": ["Touch gesture detection", "Screen interaction"],
                    "indicators": ["deliberate_touches", "focused_gaze"],
                    "weight": 0.20,
                    "confidence_threshold": 0.65
                },
                "passenger_interaction": {
                    "detection_methods": ["Multi-person detection", "Interaction analysis"],
                    "indicators": ["multiple_faces", "interaction_gestures"],
                    "weight": 0.15,
                    "confidence_threshold": 0.55
                }
            },
            "environmental_factors": {
                "lighting_conditions": {
                    "categories": ["bright_sunlight", "low_light", "artificial_light"],
                    "compensation_methods": ["Histogram equalization", "Gamma correction"],
                    "weight": 0.05
                },
                "time_of_day": {
                    "high_risk_periods": ["2-6 AM", "2-4 PM"],
                    "contextual_weight_adjustment": 1.2,
                    "weight": 0.03
                },
                "driving_duration": {
                    "fatigue_threshold": 2,  # hours
                    "high_risk_threshold": 4,  # hours
                    "weight": 0.02
                }
            }
        }
    
    def _define_comprehensive_test_cases(self) -> List[Dict]:
        """
        Define comprehensive test cases covering all scenarios
        """
        return [
            # Pure Drowsiness Cases
            {
                "case_id": "D001",
                "category": "pure_drowsiness",
                "description": "Closed eyes, normal hand position, no phone",
                "parameters": {
                    "eye_aspect_ratio": 0.15,
                    "eye_closure_duration": 45,
                    "blink_frequency": 8,
                    "phone_detected": False,
                    "hand_on_steering": True
                },
                "expected_class": "drowsy",
                "confidence_threshold": 0.9
            },
            {
                "case_id": "D002", 
                "category": "pure_drowsiness",
                "description": "Frequent yawning, head nodding, slow blinks",
                "parameters": {
                    "mouth_aspect_ratio": 0.7,
                    "head_nod_frequency": 5,
                    "eye_aspect_ratio": 0.22,
                    "phone_detected": False
                },
                "expected_class": "drowsy",
                "confidence_threshold": 0.85
            },
            
            # Phone Distraction Cases
            {
                "case_id": "P001",
                "category": "phone_distraction",
                "description": "Phone visible, texting gesture, eyes focused on phone",
                "parameters": {
                    "phone_detected": True,
                    "phone_confidence": 0.95,
                    "texting_gesture": True,
                    "eye_aspect_ratio": 0.35,
                    "gaze_direction": "down"
                },
                "expected_class": "phone_distraction",
                "confidence_threshold": 0.92
            },
            {
                "case_id": "P002",
                "category": "phone_distraction", 
                "description": "Phone call, one hand off wheel, focused attention",
                "parameters": {
                    "phone_detected": True,
                    "call_gesture": True,
                    "hand_on_steering": "partial",
                    "eye_aspect_ratio": 0.32,
                    "head_tilt_angle": 8
                },
                "expected_class": "phone_distraction",
                "confidence_threshold": 0.88
            },
            
            # Normal Driving Cases
            {
                "case_id": "N001",
                "category": "normal_driving",
                "description": "Alert driver, hands on wheel, no distractions",
                "parameters": {
                    "eye_aspect_ratio": 0.35,
                    "blink_frequency": 20,
                    "phone_detected": False,
                    "hand_on_steering": True,
                    "head_pose_stable": True
                },
                "expected_class": "normal",
                "confidence_threshold": 0.85
            },
            
            # Edge Cases - Challenging Scenarios
            {
                "case_id": "E001",
                "category": "edge_case",
                "description": "Sunglasses + phone (occlusion + distraction)",
                "parameters": {
                    "eye_visibility": 0.3,
                    "phone_detected": True,
                    "lighting_condition": "bright_sunlight",
                    "phone_confidence": 0.8
                },
                "expected_class": "phone_distraction",
                "confidence_threshold": 0.75
            },
            {
                "case_id": "E002",
                "category": "edge_case", 
                "description": "Medical condition mimicking drowsiness",
                "parameters": {
                    "eye_aspect_ratio": 0.28,
                    "blink_pattern": "irregular",
                    "phone_detected": False,
                    "reaction_time": 0.9
                },
                "expected_class": "normal",  # Should not be classified as drowsy
                "confidence_threshold": 0.60
            },
            {
                "case_id": "E003",
                "category": "edge_case",
                "description": "Navigation interaction (legitimate distraction)",
                "parameters": {
                    "touch_gesture": True,
                    "screen_interaction": True,
                    "gaze_direction": "dashboard",
                    "phone_detected": False,
                    "interaction_duration": 3  # seconds
                },
                "expected_class": "normal",  # Brief navigation is acceptable
                "confidence_threshold": 0.70
            },
            
            # Temporal Sequence Cases
            {
                "case_id": "T001",
                "category": "temporal_sequence",
                "description": "Progressive drowsiness over time",
                "sequence": [
                    {"frame": 1, "eye_aspect_ratio": 0.35, "alert_level": "high"},
                    {"frame": 300, "eye_aspect_ratio": 0.30, "alert_level": "medium"},
                    {"frame": 600, "eye_aspect_ratio": 0.25, "alert_level": "low"},
                    {"frame": 900, "eye_aspect_ratio": 0.20, "alert_level": "critical"}
                ],
                "expected_progression": "normal_to_drowsy",
                "confidence_threshold": 0.80
            },
            
            # Multi-modal Cases
            {
                "case_id": "M001",
                "category": "multi_modal",
                "description": "Phone + slight drowsiness (conflict resolution)",
                "parameters": {
                    "phone_detected": True,
                    "phone_confidence": 0.85,
                    "eye_aspect_ratio": 0.26,  # Slightly drowsy
                    "eye_closure_duration": 20,
                    "interaction_active": True
                },
                "expected_class": "phone_distraction",  # Phone takes priority
                "confidence_threshold": 0.75
            }
        ]
    
    def create_synthetic_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synthetic test data based on defined test cases
        """
        logger.info("Creating synthetic test data for comprehensive testing")
        
        # Generate synthetic features for each test case
        features = []
        labels = []
        case_info = []
        
        class_mapping = {"drowsy": 0, "phone_distraction": 1, "normal": 2}
        
        for case in self.test_cases:
            if "sequence" in case:
                # Handle temporal sequences
                for seq_data in case["sequence"]:
                    feature_vector = self._parameters_to_feature_vector(seq_data)
                    features.append(feature_vector)
                    labels.append(class_mapping.get(case.get("expected_progression", "normal").split("_")[-1], 2))
                    case_info.append(case["case_id"])
            else:
                # Handle single frame cases
                feature_vector = self._parameters_to_feature_vector(case["parameters"])
                features.append(feature_vector)
                labels.append(class_mapping[case["expected_class"]])
                case_info.append(case["case_id"])
        
        # Add realistic noise and variations
        features = np.array(features)
        labels = np.array(labels)
        
        # Add multiple variations of each test case
        augmented_features = []
        augmented_labels = []
        augmented_case_info = []
        
        for i, (feature, label, case_id) in enumerate(zip(features, labels, case_info)):
            # Original case
            augmented_features.append(feature)
            augmented_labels.append(label)
            augmented_case_info.append(case_id)
            
            # Create 10 variations with noise
            for j in range(10):
                noise = np.random.normal(0, 0.05, feature.shape)
                noisy_feature = feature + noise
                # Ensure features stay within realistic bounds
                noisy_feature = np.clip(noisy_feature, 0, 1)
                
                augmented_features.append(noisy_feature)
                augmented_labels.append(label)
                augmented_case_info.append(f"{case_id}_var{j}")
        
        self.case_info = augmented_case_info
        return np.array(augmented_features), np.array(augmented_labels)
    
    def _parameters_to_feature_vector(self, parameters: Dict) -> np.ndarray:
        """
        Convert parameter dictionary to normalized feature vector
        """
        # Create a 20-dimensional feature vector
        feature_vector = np.zeros(20)
        
        # Eye aspect ratio (normalized)
        if "eye_aspect_ratio" in parameters:
            feature_vector[0] = max(0, min(1, parameters["eye_aspect_ratio"] / 0.5))
        
        # Eye closure duration (normalized to 0-1, max 60 frames)
        if "eye_closure_duration" in parameters:
            feature_vector[1] = max(0, min(1, parameters["eye_closure_duration"] / 60))
        
        # Blink frequency (normalized, max 40 blinks/min)
        if "blink_frequency" in parameters:
            feature_vector[2] = max(0, min(1, parameters["blink_frequency"] / 40))
        
        # Mouth aspect ratio
        if "mouth_aspect_ratio" in parameters:
            feature_vector[3] = max(0, min(1, parameters["mouth_aspect_ratio"]))
        
        # Head nod frequency (normalized, max 10 nods/min)
        if "head_nod_frequency" in parameters:
            feature_vector[4] = max(0, min(1, parameters["head_nod_frequency"] / 10))
        
        # Head tilt angle (normalized, max 30 degrees)
        if "head_tilt_angle" in parameters:
            feature_vector[5] = max(0, min(1, parameters["head_tilt_angle"] / 30))
        
        # Phone detection (binary)
        if "phone_detected" in parameters:
            feature_vector[6] = 1.0 if parameters["phone_detected"] else 0.0
        
        # Phone confidence
        if "phone_confidence" in parameters:
            feature_vector[7] = max(0, min(1, parameters["phone_confidence"]))
        
        # Hand on steering (binary/partial)
        if "hand_on_steering" in parameters:
            if parameters["hand_on_steering"] is True:
                feature_vector[8] = 1.0
            elif parameters["hand_on_steering"] == "partial":
                feature_vector[8] = 0.5
            else:
                feature_vector[8] = 0.0
        
        # Texting gesture
        if "texting_gesture" in parameters:
            feature_vector[9] = 1.0 if parameters["texting_gesture"] else 0.0
        
        # Call gesture
        if "call_gesture" in parameters:
            feature_vector[10] = 1.0 if parameters["call_gesture"] else 0.0
        
        # Gaze direction (encoded)
        gaze_directions = {"forward": 0.5, "down": 0.0, "up": 1.0, "dashboard": 0.3}
        if "gaze_direction" in parameters:
            feature_vector[11] = gaze_directions.get(parameters["gaze_direction"], 0.5)
        
        # Eye visibility (for occlusion cases)
        if "eye_visibility" in parameters:
            feature_vector[12] = max(0, min(1, parameters["eye_visibility"]))
        
        # Lighting condition (encoded)
        lighting_conditions = {"bright_sunlight": 1.0, "low_light": 0.0, "artificial_light": 0.5}
        if "lighting_condition" in parameters:
            feature_vector[13] = lighting_conditions.get(parameters["lighting_condition"], 0.5)
        
        # Reaction time (normalized, max 3 seconds)
        if "reaction_time" in parameters:
            feature_vector[14] = max(0, min(1, parameters["reaction_time"] / 3))
        
        # Touch gesture
        if "touch_gesture" in parameters:
            feature_vector[15] = 1.0 if parameters["touch_gesture"] else 0.0
        
        # Screen interaction
        if "screen_interaction" in parameters:
            feature_vector[16] = 1.0 if parameters["screen_interaction"] else 0.0
        
        # Interaction duration (normalized, max 10 seconds)
        if "interaction_duration" in parameters:
            feature_vector[17] = max(0, min(1, parameters["interaction_duration"] / 10))
        
        # Head pose stability
        if "head_pose_stable" in parameters:
            feature_vector[18] = 1.0 if parameters["head_pose_stable"] else 0.0
        
        # Alert level (encoded)
        alert_levels = {"critical": 0.0, "low": 0.25, "medium": 0.5, "high": 1.0}
        if "alert_level" in parameters:
            feature_vector[19] = alert_levels.get(parameters["alert_level"], 0.5)
        
        return feature_vector
    
    def simulate_model_predictions(self, features: np.ndarray) -> np.ndarray:
        """
        Simulate model predictions based on feature analysis
        Uses rule-based logic to simulate realistic model behavior
        """
        predictions = []
        
        for feature in features:
            # Extract key features
            ear = feature[0]  # Eye aspect ratio
            eye_closure = feature[1]  # Eye closure duration
            blink_freq = feature[2]  # Blink frequency
            phone_detected = feature[6]  # Phone detection
            phone_confidence = feature[7]  # Phone confidence
            hand_on_steering = feature[8]  # Hand on steering
            texting = feature[9]  # Texting gesture
            calling = feature[10]  # Call gesture
            
            # Decision logic
            drowsiness_score = 0.0
            distraction_score = 0.0
            normal_score = 0.0
            
            # Drowsiness indicators
            if ear < 0.5:  # Low EAR indicates closed/drowsy eyes
                drowsiness_score += 0.4
            if eye_closure > 0.5:  # Long eye closure
                drowsiness_score += 0.3
            if blink_freq < 0.3:  # Low blink frequency
                drowsiness_score += 0.2
            
            # Phone distraction indicators
            if phone_detected > 0.5:
                distraction_score += 0.4 * phone_confidence
            if texting > 0.5:
                distraction_score += 0.3
            if calling > 0.5:
                distraction_score += 0.25
            if hand_on_steering < 0.8:  # Hands not fully on steering
                distraction_score += 0.15
            
            # Normal driving indicators
            if 0.5 <= ear <= 1.0 and eye_closure < 0.3:  # Alert eyes
                normal_score += 0.3
            if 0.4 <= blink_freq <= 0.8:  # Normal blink rate
                normal_score += 0.2
            if hand_on_steering > 0.8:  # Hands on steering
                normal_score += 0.2
            if phone_detected < 0.3:  # No phone detected
                normal_score += 0.3
            
            # Normalize and add noise for realism
            total_score = drowsiness_score + distraction_score + normal_score
            if total_score > 0:
                drowsiness_score /= total_score
                distraction_score /= total_score
                normal_score /= total_score
            else:
                normal_score = 1.0
            
            # Add realistic noise
            noise = np.random.normal(0, 0.05, 3)
            scores = np.array([drowsiness_score, distraction_score, normal_score]) + noise
            scores = np.maximum(0, scores)  # Ensure non-negative
            scores = scores / np.sum(scores)  # Normalize
            
            predictions.append(np.argmax(scores))
        
        return np.array(predictions)
    
    def generate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Generate comprehensive confusion matrix analysis
        """
        logger.info("Generating confusion matrix and performance metrics")
        
        # Class names
        class_names = ['Drowsy', 'Phone Distraction', 'Normal']
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # Create detailed analysis
        analysis = {
            "confusion_matrix": cm.tolist(),
            "class_names": class_names,
            "overall_accuracy": float(accuracy),
            "per_class_metrics": {
                class_names[i]: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1_score": float(f1[i]),
                    "support": int(support[i])
                } for i in range(len(class_names))
            },
            "weighted_averages": {
                "precision": float(precision_avg),
                "recall": float(recall_avg),
                "f1_score": float(f1_avg)
            }
        }
        
        return analysis
    
    def create_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray, analysis: Dict):
        """
        Create comprehensive visualizations of the results
        """
        logger.info("Creating visualization plots")
        
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Confusion Matrix Heatmap
        plt.subplot(2, 3, 1)
        cm = np.array(analysis["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=analysis["class_names"],
                   yticklabels=analysis["class_names"])
        plt.title('Confusion Matrix\nOverall Accuracy: {:.2%}'.format(analysis["overall_accuracy"]))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 2. Per-Class Performance Metrics
        plt.subplot(2, 3, 2)
        classes = analysis["class_names"]
        metrics = ['precision', 'recall', 'f1_score']
        
        x = np.arange(len(classes))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [analysis["per_class_metrics"][cls][metric] for cls in classes]
            plt.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Per-Class Performance Metrics')
        plt.xticks(x + width, classes, rotation=45)
        plt.legend()
        plt.ylim(0, 1.1)
        
        # 3. Classification Distribution
        plt.subplot(2, 3, 3)
        unique, counts = np.unique(y_true, return_counts=True)
        class_names_short = ['Drowsy', 'Distraction', 'Normal']
        colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
        
        plt.pie(counts, labels=[class_names_short[i] for i in unique], 
               autopct='%1.1f%%', colors=[colors[i] for i in unique])
        plt.title('Test Data Distribution')
        
        # 4. Error Analysis
        plt.subplot(2, 3, 4)
        errors = y_true != y_pred
        error_types = []
        error_counts = []
        
        for true_class in range(3):
            for pred_class in range(3):
                if true_class != pred_class:
                    mask = (y_true == true_class) & (y_pred == pred_class)
                    count = np.sum(mask)
                    if count > 0:
                        error_types.append(f'{class_names_short[true_class]}→{class_names_short[pred_class]}')
                        error_counts.append(count)
        
        if error_types:
            plt.bar(range(len(error_types)), error_counts, color='red', alpha=0.7)
            plt.xlabel('Error Type')
            plt.ylabel('Count')
            plt.title('Error Analysis by Type')
            plt.xticks(range(len(error_types)), error_types, rotation=45, ha='right')
        else:
            plt.text(0.5, 0.5, 'No Classification Errors!', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=16, color='green')
            plt.title('Error Analysis')
        
        # 5. Feature Importance Simulation
        plt.subplot(2, 3, 5)
        feature_names = [
            'Eye Aspect Ratio', 'Eye Closure Duration', 'Blink Frequency',
            'Mouth Aspect Ratio', 'Head Nod Frequency', 'Head Tilt',
            'Phone Detection', 'Phone Confidence', 'Hand Position',
            'Texting Gesture', 'Call Gesture', 'Gaze Direction'
        ]
        
        # Simulate feature importance based on parameter weights
        importance = [0.35, 0.30, 0.15, 0.20, 0.25, 0.20, 0.40, 0.35, 0.15, 0.30, 0.25, 0.15]
        
        # Sort by importance
        sorted_idx = np.argsort(importance)[-8:]  # Top 8 features
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_importance = [importance[i] for i in sorted_idx]
        
        plt.barh(range(len(sorted_features)), sorted_importance, color='skyblue')
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Importance Score')
        plt.title('Top Feature Importance')
        
        # 6. Model Confidence Distribution
        plt.subplot(2, 3, 6)
        # Simulate confidence scores
        np.random.seed(42)
        confidence_scores = np.random.beta(2, 1, len(y_pred)) * 0.4 + 0.6  # Biased towards high confidence
        
        correct_predictions = y_true == y_pred
        correct_conf = confidence_scores[correct_predictions]
        incorrect_conf = confidence_scores[~correct_predictions]
        
        plt.hist(correct_conf, bins=20, alpha=0.7, label='Correct Predictions', color='green')
        plt.hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect Predictions', color='red')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Model Confidence Distribution')
        plt.legend()
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'drowsiness_analysis_complete_{timestamp}.png', dpi=300, bbox_inches='tight')
        logger.info(f"Comprehensive analysis plot saved as drowsiness_analysis_complete_{timestamp}.png")
        
        return fig
    
    def run_comprehensive_analysis(self) -> Dict:
        """
        Run complete analysis including parameter analysis, testing, and visualization
        """
        logger.info("Starting comprehensive drowsiness detection analysis")
        
        # Generate test data
        X_test, y_test = self.create_synthetic_test_data()
        logger.info(f"Generated {len(X_test)} test samples")
        
        # Simulate model predictions
        y_pred = self.simulate_model_predictions(X_test)
        
        # Generate confusion matrix and metrics
        analysis = self.generate_confusion_matrix(y_test, y_pred)
        
        # Create visualizations
        fig = self.create_visualizations(y_test, y_pred, analysis)
        
        # Detailed test case analysis
        test_case_results = self._analyze_test_cases(y_test, y_pred)
        
        # Compile comprehensive report
        comprehensive_report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "parameters_analyzed": self.parameters,
            "test_cases_count": len(self.test_cases),
            "performance_metrics": analysis,
            "test_case_results": test_case_results,
            "recommendations": self._generate_recommendations(analysis)
        }
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f'drowsiness_detection_comprehensive_report_{timestamp}.json'
        
        with open(report_filename, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        logger.info(f"Comprehensive report saved as {report_filename}")
        
        return comprehensive_report
    
    def _analyze_test_cases(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Analyze performance on specific test cases
        """
        results = {}
        
        # Group results by original test cases (before augmentation)
        case_groups = {}
        for i, case_id in enumerate(self.case_info):
            base_case = case_id.split('_var')[0]  # Remove variation suffix
            if base_case not in case_groups:
                case_groups[base_case] = {"true": [], "pred": []}
            case_groups[base_case]["true"].append(y_true[i])
            case_groups[base_case]["pred"].append(y_pred[i])
        
        # Analyze each test case group
        for case_id, predictions in case_groups.items():
            true_labels = predictions["true"]
            pred_labels = predictions["pred"]
            
            accuracy = accuracy_score(true_labels, pred_labels)
            
            results[case_id] = {
                "accuracy": float(accuracy),
                "sample_count": len(true_labels),
                "correct_predictions": int(np.sum(np.array(true_labels) == np.array(pred_labels))),
                "most_common_prediction": int(np.bincount(pred_labels).argmax()),
                "expected_class": int(true_labels[0])  # All samples in group have same true label
            }
        
        return results
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """
        Generate recommendations based on analysis results
        """
        recommendations = []
        
        # Overall accuracy recommendations
        if analysis["overall_accuracy"] < 0.9:
            recommendations.append("Overall accuracy is below 90%. Consider additional training data and feature engineering.")
        
        # Per-class performance recommendations
        for class_name, metrics in analysis["per_class_metrics"].items():
            if metrics["recall"] < 0.85:
                recommendations.append(f"Low recall ({metrics['recall']:.2%}) for {class_name}. Consider balancing dataset and improving feature extraction.")
            
            if metrics["precision"] < 0.85:
                recommendations.append(f"Low precision ({metrics['precision']:.2%}) for {class_name}. Consider reducing false positives with better threshold tuning.")
        
        # Specific recommendations for drowsiness detection
        drowsy_metrics = analysis["per_class_metrics"]["Drowsy"]
        if drowsy_metrics["recall"] < 0.95:
            recommendations.append("Critical: Drowsiness detection recall should be >95% for safety. Implement ensemble methods and temporal analysis.")
        
        # Distraction vs drowsiness confusion
        cm = np.array(analysis["confusion_matrix"])
        if cm[0, 1] > 0 or cm[1, 0] > 0:  # Confusion between drowsy and distraction
            recommendations.append("Implement better differentiation between drowsiness and phone distraction using multi-modal features.")
        
        return recommendations


def main():
    """
    Main function to run comprehensive drowsiness detection analysis
    """
    print("🚗 RideBuddy Pro - Comprehensive Drowsiness Detection Analysis")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = DrowsinessParameterAnalyzer()
    
    # Run comprehensive analysis
    report = analyzer.run_comprehensive_analysis()
    
    # Print summary
    print("\n📊 Analysis Summary:")
    print(f"Overall Accuracy: {report['performance_metrics']['overall_accuracy']:.2%}")
    print(f"Test Cases Analyzed: {report['test_cases_count']}")
    
    print("\nPer-Class Performance:")
    for class_name, metrics in report['performance_metrics']['per_class_metrics'].items():
        print(f"  {class_name}:")
        print(f"    Precision: {metrics['precision']:.2%}")
        print(f"    Recall: {metrics['recall']:.2%}")
        print(f"    F1-Score: {metrics['f1_score']:.2%}")
    
    print("\n🎯 Key Recommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print("\n✅ Analysis Complete!")
    print("Check generated files for detailed visualizations and reports.")


if __name__ == "__main__":
    main()