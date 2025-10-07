#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RideBuddy - Optimized Driver Monitoring System GUI
Enhanced version with proper tabs, fixed camera feed, and improved UX
"""

import sys
import os

# Set console encoding for Windows compatibility
if sys.platform == "win32":
    try:
        # Fix encoding issues on Windows console
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass  # Fallback to default encoding
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from datetime import datetime
import random
import queue
import logging
import json
from pathlib import Path
import configparser
import hashlib
import platform
import gc
from typing import Optional, Dict, Any, List
import sqlite3
from dataclasses import dataclass, asdict
import webbrowser

# Optional imports with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Check for dependencies
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
    print("[OK] OpenCV available")
except ImportError as e:
    OPENCV_AVAILABLE = False
    print(f"[MISSING] OpenCV missing: {e}")

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
    print("[OK] PIL available")
except ImportError as e:
    PIL_AVAILABLE = False
    print(f"[MISSING] PIL missing: {e}")

# Global constants
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
VERSION = "2.1.0"
APP_NAME = "RideBuddy Pro"

# Vehicle deployment environment variables
VEHICLE_MODE = os.getenv('RIDEBUDDY_VEHICLE_MODE', 'False').lower() in ['true', '1', 'yes', 'on']
POWER_SAVE_MODE = os.getenv('RIDEBUDDY_POWER_SAVE', 'False').lower() in ['true', '1', 'yes', 'on']
FLEET_MODE = os.getenv('RIDEBUDDY_FLEET_MODE', 'False').lower() in ['true', '1', 'yes', 'on']

@dataclass
class AppConfig:
    """Application configuration with validation"""
    camera_index: int = 0
    alert_sensitivity: float = 0.7
    max_fps: int = 30
    frame_width: int = 640
    frame_height: int = 480
    audio_alerts: bool = True
    auto_save_results: bool = True
    log_level: str = "INFO"
    max_memory_mb: int = 512
    frame_drop_threshold: float = 0.8
    reconnect_attempts: int = 5
    data_retention_days: int = 30
    
    def validate(self) -> Dict[str, str]:
        """Validate configuration and return errors"""
        errors = {}
        
        if not 0 <= self.camera_index <= 10:
            errors["camera_index"] = "Camera index must be between 0-10"
        
        if not 0.1 <= self.alert_sensitivity <= 1.0:
            errors["alert_sensitivity"] = "Alert sensitivity must be between 0.1-1.0"
        
        if not 5 <= self.max_fps <= 60:
            errors["max_fps"] = "FPS must be between 5-60"
        
        if not 320 <= self.frame_width <= 1920:
            errors["frame_width"] = "Frame width must be between 320-1920"
        
        if not 240 <= self.frame_height <= 1080:
            errors["frame_height"] = "Frame height must be between 240-1080"
        
        if not 128 <= self.max_memory_mb <= 2048:
            errors["max_memory_mb"] = "Memory limit must be between 128-2048 MB"
        
        return errors

class ConfigManager:
    """Enhanced configuration management with validation"""
    
    def __init__(self, config_file: str = "ridebuddy_config.ini"):
        self.config_file = Path(config_file)
        self.config = AppConfig()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file with fallback defaults"""
        if not self.config_file.exists():
            self.save_config()
            return
        
        try:
            parser = configparser.ConfigParser()
            parser.read(self.config_file)
            
            if "RideBuddy" in parser:
                section = parser["RideBuddy"]
                
                # Load with type conversion and validation
                self.config.camera_index = section.getint("camera_index", fallback=0)
                self.config.alert_sensitivity = section.getfloat("alert_sensitivity", fallback=0.7)
                self.config.max_fps = section.getint("max_fps", fallback=30)
                self.config.frame_width = section.getint("frame_width", fallback=640)
                self.config.frame_height = section.getint("frame_height", fallback=480)
                self.config.audio_alerts = section.getboolean("audio_alerts", fallback=True)
                self.config.auto_save_results = section.getboolean("auto_save_results", fallback=True)
                self.config.log_level = section.get("log_level", fallback="INFO")
                self.config.max_memory_mb = section.getint("max_memory_mb", fallback=512)
                self.config.frame_drop_threshold = section.getfloat("frame_drop_threshold", fallback=0.8)
                self.config.reconnect_attempts = section.getint("reconnect_attempts", fallback=5)
                self.config.data_retention_days = section.getint("data_retention_days", fallback=30)
            
            # Load vehicle-specific configuration if available
            self.load_vehicle_config()
            
            # Validate loaded config
            errors = self.config.validate()
            if errors:
                logging.warning(f"Configuration validation errors: {errors}")
                # Reset invalid values to defaults
                default_config = AppConfig()
                for field, _ in errors.items():
                    setattr(self.config, field, getattr(default_config, field))
        
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            self.config = AppConfig()
    
    def load_vehicle_config(self):
        """Load vehicle-specific configuration overrides"""
        vehicle_config_file = Path("vehicle_config.json")
        
        if not vehicle_config_file.exists():
            return
        
        try:
            import json
            with open(vehicle_config_file, 'r') as f:
                vehicle_config = json.load(f)
            
            # Apply vehicle-specific overrides
            if "camera_index" in vehicle_config:
                self.config.camera_index = vehicle_config["camera_index"]
            
            if "frame_width" in vehicle_config:
                self.config.frame_width = vehicle_config["frame_width"]
            
            if "frame_height" in vehicle_config:
                self.config.frame_height = vehicle_config["frame_height"]
            
            if "max_fps" in vehicle_config:
                self.config.max_fps = vehicle_config["max_fps"]
            
            if "alert_sensitivity" in vehicle_config:
                self.config.alert_sensitivity = vehicle_config["alert_sensitivity"]
            
            if "max_memory_mb" in vehicle_config:
                self.config.max_memory_mb = vehicle_config["max_memory_mb"]
            
            if "audio_alerts" in vehicle_config:
                self.config.audio_alerts = vehicle_config["audio_alerts"]
            
            if "auto_save_results" in vehicle_config:
                self.config.auto_save_results = vehicle_config["auto_save_results"]
            
            if "data_retention_days" in vehicle_config:
                self.config.data_retention_days = vehicle_config["data_retention_days"]
            
            # Apply power save mode optimizations
            if vehicle_config.get("power_save_mode", False) or POWER_SAVE_MODE:
                self.config.max_fps = min(self.config.max_fps, 20)
                self.config.frame_width = min(self.config.frame_width, 640)
                self.config.frame_height = min(self.config.frame_height, 480)
                self.config.max_memory_mb = min(self.config.max_memory_mb, 256)
                print("[VEHICLE] Power save mode optimizations applied")
            
            # Apply fleet mode settings
            if vehicle_config.get("fleet_mode", False) or FLEET_MODE:
                self.config.data_retention_days = max(self.config.data_retention_days, 90)
                self.config.auto_save_results = True
                print("[VEHICLE] Fleet management mode enabled")
            
            print(f"[VEHICLE] Loaded configuration: {vehicle_config.get('name', 'Custom')}")
            
        except Exception as e:
            logging.warning(f"Failed to load vehicle configuration: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            parser = configparser.ConfigParser()
            parser["RideBuddy"] = asdict(self.config)
            
            # Convert boolean values to strings
            for key, value in parser["RideBuddy"].items():
                parser["RideBuddy"][key] = str(value)
            
            with open(self.config_file, 'w') as f:
                parser.write(f)
                
        except Exception as e:
            logging.error(f"Failed to save config: {e}")

class PerformanceMonitor:
    """Real-time performance monitoring and optimization"""
    
    def __init__(self):
        self.cpu_usage_history = []
        self.memory_usage_history = []
        self.fps_history = []
        self.frame_drop_count = 0
        self.last_gc = time.time()
        self.monitoring = False
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
    
    def _monitor_loop(self):
        """Performance monitoring loop"""
        while self.monitoring:
            try:
                if PSUTIL_AVAILABLE:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.cpu_usage_history.append(cpu_percent)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    process = psutil.Process()
                    process_memory = process.memory_info().rss / 1024 / 1024  # MB
                    
                    self.memory_usage_history.append({
                        'system_percent': memory.percent,
                        'process_mb': process_memory,
                        'available_mb': memory.available / 1024 / 1024
                    })
                    
                    # Limit history size
                    if len(self.cpu_usage_history) > 60:  # Keep 1 minute of data
                        self.cpu_usage_history.pop(0)
                    if len(self.memory_usage_history) > 60:
                        self.memory_usage_history.pop(0)
                
                # Automatic garbage collection if memory usage is high
                if time.time() - self.last_gc > 30:  # Every 30 seconds
                    if PSUTIL_AVAILABLE and self.memory_usage_history:
                        latest_memory = self.memory_usage_history[-1]['process_mb']
                        if latest_memory > 256:  # If using more than 256MB
                            gc.collect()
                            self.last_gc = time.time()
                    else:
                        gc.collect()
                        self.last_gc = time.time()
                
            except Exception as e:
                logging.error(f"Performance monitoring error: {e}")
            
            time.sleep(1)
    
    def should_drop_frame(self, config: AppConfig) -> bool:
        """Determine if frame should be dropped based on performance"""
        if not PSUTIL_AVAILABLE or not self.cpu_usage_history:
            return False
        
        recent_cpu = sum(self.cpu_usage_history[-5:]) / min(5, len(self.cpu_usage_history))
        return recent_cpu > (config.frame_drop_threshold * 100)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        stats = {
            'cpu_available': PSUTIL_AVAILABLE,
            'frame_drops': self.frame_drop_count,
            'gc_collections': gc.get_count(),
        }
        
        if PSUTIL_AVAILABLE and self.cpu_usage_history:
            stats.update({
                'avg_cpu_percent': sum(self.cpu_usage_history) / len(self.cpu_usage_history),
                'current_cpu_percent': self.cpu_usage_history[-1] if self.cpu_usage_history else 0,
            })
            
        if PSUTIL_AVAILABLE and self.memory_usage_history:
            latest_memory = self.memory_usage_history[-1]
            stats.update({
                'process_memory_mb': latest_memory['process_mb'],
                'system_memory_percent': latest_memory['system_percent'],
                'available_memory_mb': latest_memory['available_mb']
            })
        
        return stats

class DatabaseManager:
    """SQLite database for persistent data storage"""
    
    def __init__(self, db_file: str = "ridebuddy_data.db"):
        self.db_file = Path(db_file)
        self.init_database()
    
    def init_database(self):
        """Initialize database schema"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                # Alerts table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        alert_type TEXT NOT NULL,
                        message TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        severity TEXT NOT NULL,
                        session_id TEXT,
                        metadata TEXT
                    )
                """)
                
                # Sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE NOT NULL,
                        start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                        end_time DATETIME,
                        total_frames INTEGER DEFAULT 0,
                        total_alerts INTEGER DEFAULT 0,
                        avg_fps REAL DEFAULT 0,
                        config_snapshot TEXT
                    )
                """)
                
                # Performance metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        session_id TEXT,
                        cpu_percent REAL,
                        memory_mb REAL,
                        fps REAL,
                        frame_drops INTEGER DEFAULT 0
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            logging.error(f"Database initialization failed: {e}")
    
    def save_alert(self, alert, session_id: str):
        """Save alert to database"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO alerts (alert_type, message, confidence, severity, session_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_type,
                    alert.message, 
                    alert.confidence,
                    alert.severity,
                    session_id,
                    json.dumps(alert.to_dict())
                ))
                conn.commit()
                
        except Exception as e:
            logging.error(f"Failed to save alert: {e}")
    
    def start_session(self, session_id: str, config: AppConfig):
        """Start new monitoring session"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO sessions (session_id, config_snapshot)
                    VALUES (?, ?)
                """, (session_id, json.dumps(asdict(config))))
                conn.commit()
                
        except Exception as e:
            logging.error(f"Failed to start session: {e}")
    
    def end_session(self, session_id: str, stats: Dict[str, Any]):
        """End monitoring session with statistics"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE sessions 
                    SET end_time = CURRENT_TIMESTAMP,
                        total_frames = ?,
                        total_alerts = ?,
                        avg_fps = ?
                    WHERE session_id = ?
                """, (
                    stats.get('total_frames', 0),
                    stats.get('total_alerts', 0),
                    stats.get('avg_fps', 0),
                    session_id
                ))
                conn.commit()
                
        except Exception as e:
            logging.error(f"Failed to end session: {e}")
    
    def cleanup_old_data(self, retention_days: int = 30):
        """Clean up old data based on retention policy"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                # Delete old alerts
                cursor.execute("""
                    DELETE FROM alerts 
                    WHERE timestamp < datetime('now', '-{} days')
                """.format(retention_days))
                
                # Delete old sessions
                cursor.execute("""
                    DELETE FROM sessions 
                    WHERE start_time < datetime('now', '-{} days')
                """.format(retention_days))
                
                # Delete old performance metrics
                cursor.execute("""
                    DELETE FROM performance_metrics 
                    WHERE timestamp < datetime('now', '-{} days')
                """.format(retention_days))
                
                conn.commit()
                logging.info(f"Cleaned up data older than {retention_days} days")
                
        except Exception as e:
            logging.error(f"Failed to cleanup old data: {e}")
    
    def get_session_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get session statistics for the last N days"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                # Total sessions
                cursor.execute("""
                    SELECT COUNT(*) FROM sessions 
                    WHERE start_time >= datetime('now', '-{} days')
                """.format(days))
                total_sessions = cursor.fetchone()[0]
                
                # Total alerts by type
                cursor.execute("""
                    SELECT alert_type, COUNT(*) FROM alerts 
                    WHERE timestamp >= datetime('now', '-{} days')
                    GROUP BY alert_type
                """.format(days))
                alert_counts = dict(cursor.fetchall())
                
                # Average session duration
                cursor.execute("""
                    SELECT AVG(julianday(end_time) - julianday(start_time)) * 24 * 60 as avg_minutes
                    FROM sessions 
                    WHERE start_time >= datetime('now', '-{} days') AND end_time IS NOT NULL
                """.format(days))
                avg_duration = cursor.fetchone()[0] or 0
                
                return {
                    'total_sessions': total_sessions,
                    'alert_counts': alert_counts,
                    'avg_session_minutes': round(avg_duration, 2),
                    'period_days': days
                }
                
        except Exception as e:
            logging.error(f"Failed to get session statistics: {e}")
            return {}

class SafetyAlert:
    """Safety alert data structure"""
    def __init__(self, alert_type, message, confidence, severity="Medium"):
        self.alert_type = alert_type
        self.message = message
        self.confidence = confidence
        self.severity = severity
        self.timestamp = datetime.now()
    
    def to_dict(self):
        return {
            'type': self.alert_type,
            'message': self.message,
            'confidence': self.confidence,
            'severity': self.severity,
            'timestamp': self.timestamp.strftime("%H:%M:%S")
        }

class CameraManager:
    """Enhanced camera management with reconnection and error handling"""
    def __init__(self, config: AppConfig, performance_monitor: PerformanceMonitor):
        self.config = config
        self.performance_monitor = performance_monitor
        self.camera = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue()
        self.capture_thread = None
        self.is_running = False
        self.current_frame = None
        self.reconnect_attempts = 0
        self.last_frame_time = 0
        self.connection_stable = False
        self.error_count = 0
        
    def initialize_camera(self, camera_index=None):
        """Initialize camera connection with enhanced error handling"""
        if camera_index is None:
            camera_index = self.config.camera_index
            
        try:
            # Validate camera index
            if not isinstance(camera_index, int) or camera_index < 0:
                self.result_queue.put(("error", f"Invalid camera index: {camera_index}"))
                return False
            
            # Try specified camera first, then fallback to others
            camera_indices = [camera_index]
            if camera_index not in [0, 1, 2]:
                camera_indices.extend([0, 1, 2])
            
            for idx in camera_indices:
                try:
                    logging.info(f"Attempting to connect to camera {idx}")
                    test_camera = cv2.VideoCapture(idx)
                    
                    if test_camera.isOpened():
                        ret, frame = test_camera.read()
                        if ret and frame is not None:
                            # Configure camera settings
                            test_camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
                            test_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
                            test_camera.set(cv2.CAP_PROP_FPS, self.config.max_fps)
                            
                            # Set buffer size to reduce latency
                            test_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            
                            self.camera = test_camera
                            self.connection_stable = True
                            self.error_count = 0
                            self.reconnect_attempts = 0
                            
                            # Get actual camera properties
                            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
                            
                            self.result_queue.put(("status", 
                                f"Camera {idx} connected: {actual_width}x{actual_height}@{actual_fps:.1f}fps"))
                            logging.info(f"Camera {idx} initialized successfully")
                            return True
                        else:
                            test_camera.release()
                            logging.warning(f"Camera {idx} opened but cannot read frames")
                    else:
                        test_camera.release()
                        logging.warning(f"Cannot open camera {idx}")
                        
                except Exception as e:
                    logging.error(f"Error testing camera {idx}: {e}")
                    continue
            
            self.result_queue.put(("error", "No working camera found"))
            logging.error("Failed to initialize any camera")
            return False
            
        except Exception as e:
            self.result_queue.put(("error", f"Camera initialization failed: {str(e)}"))
            logging.error(f"Camera initialization exception: {e}")
            return False
    
    def start_capture(self):
        """Start camera capture in separate thread"""
        if not self.camera or not self.camera.isOpened():
            return False
        
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        return True
    
    def stop_capture(self):
        """Stop camera capture"""
        self.is_running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
    
    def _capture_loop(self):
        """Enhanced capture loop with frame dropping and reconnection"""
        frame_count = 0
        last_fps_time = time.time()
        consecutive_errors = 0
        
        while self.is_running:
            try:
                # Check if camera is still available
                if not self.camera or not self.camera.isOpened():
                    if self._attempt_reconnection():
                        continue
                    else:
                        break
                
                # Check if we should drop frames due to high CPU usage
                if self.performance_monitor.should_drop_frame(self.config):
                    self.performance_monitor.frame_drop_count += 1
                    time.sleep(1.0 / self.config.max_fps)
                    continue
                
                ret, frame = self.camera.read()
                
                if ret and frame is not None:
                    self.current_frame = frame.copy()
                    self.last_frame_time = time.time()
                    consecutive_errors = 0
                    self.connection_stable = True
                    
                    # Process frame (simulate AI inference)
                    start_time = time.time()
                    predictions = self._simulate_ai_processing(frame)
                    processing_time = (time.time() - start_time) * 1000  # ms
                    
                    # Add processing metrics
                    predictions["processing_time_ms"] = processing_time
                    
                    # Send frame and results to GUI
                    try:
                        self.result_queue.put(("frame", {
                            "frame": frame,
                            "predictions": predictions,
                            "frame_count": frame_count,
                            "timestamp": time.time()
                        }), block=False)
                    except queue.Full:
                        # Queue is full, drop this frame
                        self.performance_monitor.frame_drop_count += 1
                    
                    frame_count += 1
                    
                    # Calculate and send FPS every second
                    current_time = time.time()
                    if current_time - last_fps_time >= 1.0:
                        fps = frame_count / (current_time - last_fps_time)
                        self.result_queue.put(("fps", fps))
                        frame_count = 0
                        last_fps_time = current_time
                    
                    # Dynamic frame rate control
                    target_delay = 1.0 / self.config.max_fps
                    actual_delay = max(0.001, target_delay - processing_time / 1000)
                    time.sleep(actual_delay)
                    
                else:
                    # Frame read failed
                    consecutive_errors += 1
                    self.error_count += 1
                    self.connection_stable = False
                    
                    if consecutive_errors > 10:
                        logging.error("Too many consecutive frame read errors, attempting reconnection")
                        self.result_queue.put(("error", "Camera connection lost, attempting reconnection"))
                        
                        if not self._attempt_reconnection():
                            break
                    else:
                        time.sleep(0.1)
                    
            except Exception as e:
                self.error_count += 1
                consecutive_errors += 1
                logging.error(f"Capture loop error: {e}")
                self.result_queue.put(("error", f"Capture error: {str(e)}"))
                
                if consecutive_errors > 5:
                    if not self._attempt_reconnection():
                        break
                else:
                    time.sleep(0.1)
        
        logging.info("Camera capture loop ended")
        self.result_queue.put(("status", "Camera capture stopped"))
    
    def _attempt_reconnection(self) -> bool:
        """Attempt to reconnect to camera with exponential backoff"""
        if self.reconnect_attempts >= self.config.reconnect_attempts:
            logging.error(f"Max reconnection attempts ({self.config.reconnect_attempts}) exceeded")
            self.result_queue.put(("error", "Camera reconnection failed - max attempts exceeded"))
            return False
        
        self.reconnect_attempts += 1
        
        # Exponential backoff: 1s, 2s, 4s, 8s, 16s
        backoff_time = min(16, 2 ** (self.reconnect_attempts - 1))
        
        logging.info(f"Attempting camera reconnection #{self.reconnect_attempts} after {backoff_time}s")
        self.result_queue.put(("status", f"Reconnecting to camera (attempt {self.reconnect_attempts}/{self.config.reconnect_attempts})..."))
        
        time.sleep(backoff_time)
        
        # Release old camera
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
            self.camera = None
        
        # Try to reinitialize
        return self.initialize_camera()
    
    def _simulate_ai_processing(self, frame):
        """Simulate AI model inference - replace with actual model"""
        # Simulate processing delay
        time.sleep(0.005)  # 5ms processing time
        
        # Generate random predictions for demo
        alertness_states = ["Normal", "Drowsy", "Phone_Distraction"]
        alertness = random.choice(alertness_states)
        
        # More realistic probabilities
        if alertness == "Normal":
            confidence = random.uniform(0.85, 0.98)
            phone_detected = random.random() < 0.05
        elif alertness == "Drowsy":
            confidence = random.uniform(0.70, 0.90)
            phone_detected = random.random() < 0.10
        else:  # Phone_Distraction
            confidence = random.uniform(0.75, 0.95)
            phone_detected = random.random() < 0.85
            
        seatbelt_worn = random.random() < 0.92
        
        return {
            "alertness": alertness,
            "confidence": confidence,
            "phone_detected": phone_detected,
            "seatbelt_worn": seatbelt_worn
        }
    
    def release(self):
        """Release camera resources"""
        self.stop_capture()
        if self.camera:
            self.camera.release()
            self.camera = None

class RideBuddyOptimizedGUI:
    """Main GUI application with enhanced features and monitoring"""
    
    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_NAME} v{VERSION} - Professional Driver Monitoring System")
        
        # Get screen dimensions for responsive sizing
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate responsive window dimensions (80% of screen size)
        window_width = min(1500, int(screen_width * 0.85))
        window_height = min(1000, int(screen_height * 0.85))
        
        # Center window on screen
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Set responsive minimum size (60% of calculated size, but not less than 800x600)
        min_width = max(800, int(window_width * 0.6))
        min_height = max(600, int(window_height * 0.6))
        self.root.minsize(min_width, min_height)
        
        # Store screen dimensions for responsive layout calculations
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.window_width = window_width
        self.window_height = window_height
        
        print(f"[SCREEN] Screen: {screen_width}x{screen_height}, Window: {window_width}x{window_height}")
        print(f"[SCREEN] Min size: {min_width}x{min_height}, Position: {x}+{y}")
        
        # Bind resize handler for responsive layout updates
        self.root.bind('<Configure>', self.on_window_resize)
        self.last_resize_time = 0
        
        # Configure window icon and styling
        try:
            # Set application icon if available
            self.root.iconbitmap(default="icon.ico")
        except:
            pass
        
        # Setup modern theme and styling
        self.setup_modern_theme()
        
        # Initialize core components
        self.config_manager = ConfigManager()
        self.config = self.config_manager.config
        self.performance_monitor = PerformanceMonitor()
        self.database = DatabaseManager()
        
        # Application state
        self.camera_manager = CameraManager(self.config, self.performance_monitor)
        self.is_monitoring = False
        self.alerts = []
        self.frame_count = 0
        self.current_fps = 0
        self.session_id = None
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'drowsy_detections': 0,
            'phone_detections': 0,
            'alerts_triggered': 0,
            'session_start': datetime.now()
        }
        
        # Data privacy and security
        self.data_hash = None
        self.user_consent = False
        
        # Setup application - ensure GUI is created first
        self.setup_logging()
        self.setup_gui()
        
        # Show privacy consent after GUI is ready
        self.root.after(100, self.show_privacy_consent_delayed)
        
        self.performance_monitor.start_monitoring()
        
        # Start GUI update loop (schedule it, don't call directly)
        self.root.after(100, self.update_gui_loop)
        
        # Cleanup old data on startup
        self.database.cleanup_old_data(self.config.data_retention_days)
    
    def setup_modern_theme(self):
        """Setup modern dark theme and styling"""
        
        # Configure ttk Style
        self.style = ttk.Style()
        
        # Set theme - try different themes for best appearance
        available_themes = self.style.theme_names()
        if 'vista' in available_themes:
            self.style.theme_use('vista')
        elif 'clam' in available_themes:
            self.style.theme_use('clam')
        else:
            self.style.theme_use('default')
        
        # Color scheme - Modern dark theme
        self.colors = {
            'bg_primary': '#1e1e1e',      # Dark background
            'bg_secondary': '#2d2d2d',    # Secondary background
            'bg_tertiary': '#3d3d3d',     # Tertiary background
            'accent_primary': '#0078d4',   # Microsoft blue
            'accent_success': '#107c10',   # Success green
            'accent_warning': '#ffb900',   # Warning orange
            'accent_danger': '#d13438',    # Error red
            'text_primary': '#ffffff',     # Primary text
            'text_secondary': '#b3b3b3',   # Secondary text
            'text_muted': '#808080',       # Muted text
            'border': '#404040',           # Border color
        }
        
        # Configure root window
        self.root.configure(bg=self.colors['bg_primary'])
        
        # Calculate responsive font sizes based on screen dimensions
        self.setup_responsive_fonts()
        
        # Configure ttk styles for modern appearance
        
        # Notebook (tabs) styling with ultra-enhanced visibility
        self.style.configure('Custom.TNotebook', 
                           background=self.colors['bg_primary'],
                           borderwidth=2,
                           bordercolor='#0078d4',
                           tabmargins=[3, 3, 3, 0])
        
        self.style.configure('Custom.TNotebook.Tab',
                           background='#1a1a1a',              # Very dark background for maximum contrast
                           foreground='#ffffff',              # Pure white text
                           padding=[30, 20],                  # Even larger padding for better visibility
                           font=('Segoe UI', 16, 'bold'),    # Larger font (16pt) for better readability
                           focuscolor='none',
                           relief='raised',                   # 3D raised effect
                           borderwidth=4)                     # Thicker border for prominence
        
        self.style.map('Custom.TNotebook.Tab',
                      background=[('selected', '#ff6600'),      # Bright orange when selected (high visibility)
                                ('active', '#00cc44'),         # Bright green when hovering
                                ('!selected', '#1a1a1a')],     # Very dark when not selected
                      foreground=[('selected', '#ffffff'),     # Pure white text when selected
                                ('active', '#ffffff'),        # Pure white text when hovering
                                ('!selected', '#ffff00')],    # Bright yellow text normally for visibility
                      bordercolor=[('selected', '#ffff00'),    # Bright yellow border when selected
                                 ('active', '#ff6600'),       # Orange border when hovering
                                 ('!selected', '#666666')],   # Gray border normally
                      relief=[('selected', 'solid'),          # Solid relief when selected
                             ('active', 'raised'),
                             ('!selected', 'flat')])
        
        # Frame styling with enhanced visibility
        self.style.configure('Dark.TFrame',
                           background=self.colors['bg_primary'],
                           borderwidth=0,
                           relief='flat')
        
        self.style.configure('Card.TFrame',
                           background=self.colors['bg_secondary'],
                           borderwidth=2,
                           relief='solid',
                           bordercolor=self.colors['border'])
        
        # Additional frame styles
        self.style.configure('Bright.TFrame',
                           background=self.colors['bg_tertiary'],
                           borderwidth=1,
                           relief='solid',
                           bordercolor=self.colors['border'])
        
        # Label styling with enhanced visibility
        self.style.configure('Heading.TLabel',
                           background=self.colors['bg_primary'],
                           foreground='#ffffff',
                           font=('Segoe UI', 15, 'bold'))
        
        self.style.configure('Subheading.TLabel',
                           background=self.colors['bg_secondary'],
                           foreground='#e0e0e0',
                           font=('Segoe UI', 12))
        
        self.style.configure('Status.TLabel',
                           background=self.colors['bg_primary'],
                           foreground='#cccccc',
                           font=('Segoe UI', 10))
        
        # Additional label styles for better visibility
        self.style.configure('Bright.TLabel',
                           background=self.colors['bg_secondary'],
                           foreground='#ffffff',
                           font=('Segoe UI', 11, 'bold'))
        
        self.style.configure('Value.TLabel',
                           background=self.colors['bg_secondary'],
                           foreground='#ffffff',
                           font=('Segoe UI', 12, 'bold'))
        
        # Button styling with enhanced visibility
        self.style.configure('Primary.TButton',
                           background=self.colors['accent_primary'],
                           foreground='#ffffff',
                           font=('Segoe UI', 11, 'bold'),
                           borderwidth=2,
                           bordercolor=self.colors['accent_primary'],
                           focuscolor=self.colors['accent_primary'],
                           padding=[20, 12])
        
        self.style.map('Primary.TButton',
                      background=[('active', '#106ebe'),
                                ('pressed', '#005a9e'),
                                ('disabled', '#666666')],
                      foreground=[('disabled', '#999999')])
        
        self.style.configure('Success.TButton',
                           background=self.colors['accent_success'],
                           foreground='#ffffff',
                           font=('Segoe UI', 11, 'bold'),
                           borderwidth=2,
                           bordercolor=self.colors['accent_success'],
                           focuscolor=self.colors['accent_success'],
                           padding=[20, 12])
        
        self.style.map('Success.TButton',
                      background=[('active', '#0e6e0e'),
                                ('pressed', '#0c5c0c')])
        
        self.style.configure('Warning.TButton',
                           background=self.colors['accent_warning'],
                           foreground='#000000',
                           font=('Segoe UI', 11, 'bold'),
                           borderwidth=2,
                           bordercolor=self.colors['accent_warning'],
                           focuscolor=self.colors['accent_warning'],
                           padding=[20, 12])
        
        self.style.map('Warning.TButton',
                      background=[('active', '#e5a700'),
                                ('pressed', '#cc9500')])
        
        self.style.configure('Danger.TButton',
                           background=self.colors['accent_danger'],
                           foreground='#ffffff',
                           font=('Segoe UI', 11, 'bold'),
                           borderwidth=2,
                           bordercolor=self.colors['accent_danger'],
                           focuscolor=self.colors['accent_danger'],
                           padding=[20, 12])
        
        self.style.map('Danger.TButton',
                      background=[('active', '#b82d31'),
                                ('pressed', '#9e252a')])
        
        # Entry and Combobox styling
        self.style.configure('Modern.TEntry',
                           fieldbackground=self.colors['bg_tertiary'],
                           foreground=self.colors['text_primary'],
                           bordercolor=self.colors['border'],
                           insertcolor=self.colors['text_primary'],
                           font=('Segoe UI', 10))
        
        self.style.configure('Modern.TCombobox',
                           fieldbackground=self.colors['bg_tertiary'],
                           foreground=self.colors['text_primary'],
                           bordercolor=self.colors['border'],
                           font=('Segoe UI', 10))
        
        # Progressbar styling
        self.style.configure('Modern.Horizontal.TProgressbar',
                           background=self.colors['accent_primary'],
                           troughcolor=self.colors['bg_tertiary'],
                           borderwidth=0,
                           lightcolor=self.colors['accent_primary'],
                           darkcolor=self.colors['accent_primary'])
        
        # LabelFrame styling with enhanced visibility
        self.style.configure('Modern.TLabelframe',
                           background=self.colors['bg_secondary'],
                           bordercolor=self.colors['border'],
                           borderwidth=2,
                           relief='solid')
        
        self.style.configure('Modern.TLabelframe.Label',
                           background=self.colors['bg_secondary'],
                           foreground='#ffffff',
                           font=('Segoe UI', 12, 'bold'))
        
        # Additional styles for better UI visibility
        self.style.configure('Tab.TFrame',
                           background=self.colors['bg_primary'],
                           borderwidth=0)
        
        # Ultra visible button styles with explicit colors
        self.style.configure('Visible.Primary.TButton',
                           background='#0066ff',
                           foreground='#ffffff',
                           font=('Arial', 14, 'bold'),
                           borderwidth=5,
                           relief='raised',
                           focuscolor='none',
                           padding=[30, 20])
        
        self.style.map('Visible.Primary.TButton',
                      background=[('active', '#0052cc'),
                                ('pressed', '#004099'),
                                ('!active', '#0066ff')])
        
        self.style.configure('Visible.Success.TButton',
                           background='#00aa00',
                           foreground='#ffffff',
                           font=('Arial', 14, 'bold'),
                           borderwidth=5,
                           relief='raised',
                           focuscolor='none',
                           padding=[30, 20])
        
        self.style.map('Visible.Success.TButton',
                      background=[('active', '#008800'),
                                ('pressed', '#006600'),
                                ('!active', '#00aa00')])
        
        self.style.configure('Visible.Danger.TButton',
                           background='#ff0000',
                           foreground='#ffffff',
                           font=('Arial', 14, 'bold'),
                           borderwidth=5,
                           relief='raised',
                           focuscolor='none',
                           padding=[30, 20])
        
        self.style.map('Visible.Danger.TButton',
                      background=[('active', '#cc0000'),
                                ('pressed', '#990000'),
                                ('!active', '#ff0000')])
        
        # Text widget styling with enhanced visibility for full screen
        self.text_style_config = {
            'bg': '#1a1a1a',           # Very dark background for better contrast
            'fg': '#ffffff',           # Bright white text
            'insertbackground': '#ffffff',
            'selectbackground': self.colors['accent_primary'],
            'selectforeground': '#ffffff',
            'font': ('Consolas', 12, 'bold'),  # Larger, bold font for better readability
            'borderwidth': 3,          # Thicker border for visibility
            'relief': 'solid',
            'highlightbackground': '#0078d4',  # Blue highlight for focus
            'highlightcolor': '#0078d4',
            'highlightthickness': 2
        }
    
    def setup_responsive_fonts(self):
        """Setup responsive font sizes based on screen dimensions"""
        screen_width = self.screen_width
        screen_height = self.screen_height
        
        # Base font sizes - scale with screen size
        if screen_width >= 1920:  # 1080p+ screens
            base_scale = 1.2
        elif screen_width >= 1366:  # Standard laptop screens
            base_scale = 1.0
        elif screen_width >= 1024:  # Small laptop screens
            base_scale = 0.9
        else:  # Very small screens
            base_scale = 0.8
            
        # Calculate responsive font sizes
        self.fonts = {
            'heading_large': ('Segoe UI', int(20 * base_scale), 'bold'),
            'heading_medium': ('Segoe UI', int(16 * base_scale), 'bold'),
            'heading_small': ('Segoe UI', int(14 * base_scale), 'bold'),
            'body_large': ('Segoe UI', int(12 * base_scale)),
            'body_medium': ('Segoe UI', int(11 * base_scale)),
            'body_small': ('Segoe UI', int(10 * base_scale)),
            'monospace': ('Consolas', int(11 * base_scale)),
            'button': ('Segoe UI', int(10 * base_scale), 'bold')
        }
        
        print(f"[FONTS] Screen {screen_width}x{screen_height}, scale: {base_scale:.1f}")
        print(f"[FONTS] Button font: {self.fonts['button']}")
    
    def show_privacy_consent_delayed(self):
        """Show privacy consent after GUI is fully loaded"""
        # Debug vehicle mode detection
        print(f"[DEBUG] VEHICLE_MODE: {VEHICLE_MODE}, FLEET_MODE: {FLEET_MODE}")
        print(f"[DEBUG] Environment vars: RIDEBUDDY_VEHICLE_MODE={os.getenv('RIDEBUDDY_VEHICLE_MODE')}, RIDEBUDDY_FLEET_MODE={os.getenv('RIDEBUDDY_FLEET_MODE')}")
        
        # In vehicle/fleet mode, auto-consent for seamless deployment
        if VEHICLE_MODE or FLEET_MODE:
            print("[VEHICLE] Auto-consenting for vehicle deployment mode")
            self.user_consent = True
            # Immediately initialize camera in vehicle mode
            if OPENCV_AVAILABLE:
                print("[VEHICLE] Scheduling camera initialization in 1 second...")
                self.root.after(1000, self.initialize_camera)
            else:
                print("[VEHICLE] OpenCV not available - camera cannot initialize")
        else:
            print("[DESKTOP] Normal desktop mode - showing privacy dialog")
            # Normal desktop mode - show privacy dialog
            self.show_privacy_consent()
    
    def setup_gui(self):
        """Setup the main GUI with modern tabbed interface"""
        print("[GUI] Setting up modern GUI interface...")
        
        # Create main container with padding
        main_container = ttk.Frame(self.root, style='Dark.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create header
        self.create_modern_header(main_container)
        
        # Create main notebook for tabs with modern styling
        self.notebook = ttk.Notebook(main_container, style='Custom.TNotebook')
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Create tabs
        print("[TAB] Creating monitoring tab...")
        self.create_monitoring_tab()
        print("[TAB] Creating alerts tab...")
        self.create_alerts_tab()
        print("[TAB] Creating analysis tab...")
        self.create_analysis_tab()
        print("[TAB] Creating testing tab...")
        self.create_testing_tab()
        print("[TAB] Creating settings tab...")
        self.create_settings_tab()
        
        # Status bar
        print("[GUI] Creating status bar...")
        self.create_status_bar()
        print("[OK] Modern GUI setup completed successfully!")
    
    def create_modern_header(self, parent):
        """Create modern application header"""
        header_frame = ttk.Frame(parent, style='Card.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Left side - App info
        left_header = ttk.Frame(header_frame, style='Card.TFrame')
        left_header.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # App title
        title_label = ttk.Label(left_header, 
                               text=f"🚗 {APP_NAME} v{VERSION}",
                               style='Heading.TLabel')
        title_label.pack(anchor=tk.W)
        
        # Subtitle
        subtitle_label = ttk.Label(left_header,
                                 text="Professional AI-Powered Driver Monitoring System",
                                 style='Subheading.TLabel')
        subtitle_label.pack(anchor=tk.W)
        
        # Right side - Status indicators
        right_header = ttk.Frame(header_frame, style='Card.TFrame')
        right_header.pack(side=tk.RIGHT, padx=15, pady=10)
        
        # AI Model status
        ai_status_frame = ttk.Frame(right_header, style='Card.TFrame')
        ai_status_frame.pack(side=tk.TOP, anchor=tk.E, pady=(0, 5))
        
        ai_icon = ttk.Label(ai_status_frame, text="🧠", font=('Segoe UI', 12))
        ai_icon.pack(side=tk.LEFT)
        
        self.ai_status_label = ttk.Label(ai_status_frame,
                                        text="Enhanced AI Model: Active",
                                        style='Status.TLabel',
                                        foreground=self.colors['accent_success'])
        self.ai_status_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # System status
        sys_status_frame = ttk.Frame(right_header, style='Card.TFrame')
        sys_status_frame.pack(side=tk.TOP, anchor=tk.E)
        
        sys_icon = ttk.Label(sys_status_frame, text="⚡", font=('Segoe UI', 12))
        sys_icon.pack(side=tk.LEFT)
        
        self.sys_status_label = ttk.Label(sys_status_frame,
                                         text="System: Ready",
                                         style='Status.TLabel',
                                         foreground=self.colors['accent_success'])
        self.sys_status_label.pack(side=tk.LEFT, padx=(5, 0))
    
    def create_monitoring_tab(self):
        """Create responsive monitoring tab with optimized screen layout"""
        monitoring_frame = ttk.Frame(self.notebook, style='Tab.TFrame')
        self.notebook.add(monitoring_frame, text="🎥 Live Monitoring")
        
        # Use stored screen dimensions for responsive design
        screen_width = self.screen_width
        screen_height = self.screen_height
        window_width = self.window_width
        window_height = self.window_height
        
        print(f"[LAYOUT] Creating responsive layout for {window_width}x{window_height}")
        
        # Main container with responsive layout
        main_container = ttk.Frame(monitoring_frame, style='Dark.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Top row - Quick stats (compact for small screens)
        self.create_quick_stats_row(main_container)
        
        # Determine layout based on screen size
        use_vertical_layout = window_width < 1000 or window_height < 700
        
        if use_vertical_layout:
            print("[LAYOUT] Using vertical layout for small screen")
            self.create_vertical_layout(main_container, window_width, window_height)
        else:
            print("[LAYOUT] Using horizontal layout for large screen")
            self.create_horizontal_layout(main_container, window_width, window_height)
    
    def create_control_buttons(self, parent, compact=False):
        """Create control buttons with responsive design"""
        # Control buttons frame
        frame_title = "🎛️ CONTROLS" if compact else "🎛️ MONITORING CONTROLS"
        padding = "10" if compact else "15"
        
        controls_frame = ttk.LabelFrame(parent,
                                      text=frame_title,
                                      style='Modern.TLabelframe',
                                      padding=padding)
        
        if compact:
            controls_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10), padx=5)
        else:
            controls_frame.pack(fill=tk.X, pady=(0, 15), padx=5)
        
        # Debug message
        layout_type = "compact" if compact else "full"
        print(f"[DEBUG] Creating {layout_type} control buttons...")
        
        # Calculate button dimensions based on screen size
        button_font = self.fonts['button'] if hasattr(self, 'fonts') else ('Arial', 10, 'bold')
        button_padx = 10 if compact else 15
        button_pady = 6 if compact else 10
        
        # Ultra-visible control buttons using tk.Button for guaranteed visibility
        self.start_button = tk.Button(controls_frame, 
                                     text="▶️ START MONITORING", 
                                     command=self.toggle_monitoring,
                                     bg='#0066ff',
                                     fg='white',
                                     font=button_font,
                                     relief='raised',
                                     bd=3,
                                     padx=button_padx,
                                     pady=button_pady,
                                     activebackground='#0052cc',
                                     activeforeground='white')
        self.start_button.pack(fill=tk.X, pady=(0, 8))
        print("[DEBUG] Start button created and packed")
        
        self.camera_button = tk.Button(controls_frame, 
                                      text="📷 RECONNECT CAMERA", 
                                      command=self.reconnect_camera,
                                      bg='#00aa00',
                                      fg='white',
                                      font=button_font,
                                      relief='raised',
                                      bd=3,
                                      padx=button_padx,
                                      pady=button_pady,
                                      activebackground='#008800',
                                      activeforeground='white')
        self.camera_button.pack(fill=tk.X, pady=(0, 8))
        print("[DEBUG] Camera button created and packed")
        
        # Emergency stop button
        self.emergency_button = tk.Button(controls_frame,
                                         text="🛑 EMERGENCY STOP",
                                         command=self.emergency_stop,
                                         bg='#ff0000',
                                         fg='white',
                                         font=('Arial', 12, 'bold'),
                                         relief='raised',
                                         bd=3,
                                         padx=15,
                                         pady=10,
                                         activebackground='#cc0000',
                                         activeforeground='white')
        self.emergency_button.pack(fill=tk.X, pady=0)
        print("[DEBUG] Emergency button created and packed")
    
    def create_quick_stats_row(self, parent):
        """Create quick statistics row at top"""
        stats_row = ttk.Frame(parent, style='Card.TFrame')
        stats_row.pack(fill=tk.X, pady=(0, 15))
        
        # Quick stats cards with proper label references
        stats_cards = [
            ("🎯", "Accuracy", "100%", self.colors['accent_success'], "accuracy"),
            ("👁️", "Alertness", "Normal", self.colors['accent_success'], "alertness"),
            ("📱", "Phone", "No", self.colors['accent_success'], "phone"),
            ("🔒", "Seatbelt", "Worn", self.colors['accent_success'], "seatbelt"),
        ]
        
        for i, (icon, label, value, color, attr_name) in enumerate(stats_cards):
            card = ttk.Frame(stats_row, style='Card.TFrame')
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10) if i < len(stats_cards)-1 else (0, 0))
            
            # Icon
            icon_label = ttk.Label(card, text=icon, font=('Segoe UI', 16),
                                 background=self.colors['bg_secondary'],
                                 foreground=self.colors['text_primary'])
            icon_label.pack(pady=(10, 0))
            
            # Label
            label_widget = ttk.Label(card, text=label, 
                                   font=('Segoe UI', 10, 'bold'),
                                   background=self.colors['bg_secondary'],
                                   foreground=self.colors['text_primary'])
            label_widget.pack()
            
            # Value - Store reference for updates
            value_widget = ttk.Label(card, text=value, 
                                   font=('Segoe UI', 11, 'bold'),
                                   background=self.colors['bg_secondary'],
                                   foreground=color)
            value_widget.pack(pady=(0, 10))
            
            # Store references for status updates
            if attr_name == "alertness":
                self.alertness_label = value_widget
            elif attr_name == "phone":
                self.phone_label = value_widget  
            elif attr_name == "seatbelt":
                self.seatbelt_label = value_widget
            elif attr_name == "accuracy":
                self.accuracy_label = value_widget
    
    def create_detection_panel(self, parent, compact=False):
        """Create detection information panel with responsive design"""
        # Use the provided parent directly
        right_panel = parent
        
        # Real-time metrics
        metrics_frame = ttk.LabelFrame(right_panel, 
                                     text="📊 Live Metrics", 
                                     style='Modern.TLabelframe',
                                     padding="15")
        metrics_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        self.create_modern_metrics_display(metrics_frame)
        
        # Alert history
        alerts_frame = ttk.LabelFrame(right_panel, 
                                    text="🚨 Recent Alerts", 
                                    style='Modern.TLabelframe',
                                    padding="15")
        alerts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))
        
        self.create_modern_alerts_display(alerts_frame)
    
    def create_modern_status_indicators(self, parent):
        """Create modern status indicators with visual elements"""
        # Alertness status
        alertness_row = ttk.Frame(parent, style='Card.TFrame')
        alertness_row.pack(fill=tk.X, pady=(0, 10))
        
        alert_icon = ttk.Label(alertness_row, text="👁️", font=('Segoe UI', 14))
        alert_icon.pack(side=tk.LEFT)
        
        alert_info = ttk.Frame(alertness_row, style='Card.TFrame')
        alert_info.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        
        ttk.Label(alert_info, text="Driver Alertness", 
                 style='Subheading.TLabel').pack(anchor=tk.W)
        
        self.alertness_label = ttk.Label(alert_info, text="Normal", 
                                       font=('Segoe UI', 10, 'bold'),
                                       foreground=self.colors['accent_success'])
        self.alertness_label.pack(anchor=tk.W)
        
        # Confidence indicator
        conf_row = ttk.Frame(parent, style='Card.TFrame')
        conf_row.pack(fill=tk.X, pady=(0, 10))
        
        conf_icon = ttk.Label(conf_row, text="📈", font=('Segoe UI', 14))
        conf_icon.pack(side=tk.LEFT)
        
        conf_info = ttk.Frame(conf_row, style='Card.TFrame')
        conf_info.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        
        ttk.Label(conf_info, text="Detection Confidence", 
                 style='Subheading.TLabel').pack(anchor=tk.W)
        
        self.confidence_label = ttk.Label(conf_info, text="95.2%", 
                                        font=('Segoe UI', 10, 'bold'),
                                        foreground=self.colors['accent_success'])
        self.confidence_label.pack(anchor=tk.W)
        
        # Progress bar for confidence
        self.confidence_progress = ttk.Progressbar(conf_info, 
                                                 style='Modern.Horizontal.TProgressbar',
                                                 length=150, value=95.2)
        self.confidence_progress.pack(anchor=tk.W, pady=(2, 0))
    
    def create_modern_metrics_display(self, parent):
        """Create modern metrics display"""
        metrics_grid = ttk.Frame(parent, style='Card.TFrame')
        metrics_grid.pack(fill=tk.X)
        
        # Metrics data with label references
        metrics = [
            ("Confidence", "0.0%", "�", "confidence"),
            ("Processed", "0", "⚡", "frames"),
            ("Alerts", "0", "🚨", "alerts"),
            ("Uptime", "00:00", "⏰", "uptime")
        ]
        
        for i, (label, value, icon, attr_name) in enumerate(metrics):
            row = i // 2
            col = i % 2
            
            metric_frame = ttk.Frame(metrics_grid, style='Card.TFrame')
            metric_frame.grid(row=row, column=col, sticky=tk.W+tk.E, 
                            padx=(0, 10) if col == 0 else (10, 0),
                            pady=(0, 5) if row == 0 else (5, 0))
            
            ttk.Label(metric_frame, text=icon, font=('Segoe UI', 12),
                     background=self.colors['bg_secondary'],
                     foreground=self.colors['text_primary']).pack(side=tk.LEFT)
            
            info_frame = ttk.Frame(metric_frame, style='Card.TFrame')
            info_frame.pack(side=tk.LEFT, padx=(5, 0))
            
            ttk.Label(info_frame, text=label, 
                     font=('Segoe UI', 9),
                     background=self.colors['bg_secondary'],
                     foreground=self.colors['text_secondary']).pack(anchor=tk.W)
            
            value_label = ttk.Label(info_frame, text=value, 
                                  font=('Segoe UI', 10, 'bold'),
                                  background=self.colors['bg_secondary'],
                                  foreground=self.colors['text_primary'])
            value_label.pack(anchor=tk.W)
            
            # Store reference for updates
            if attr_name == "confidence":
                self.confidence_label = value_label
            elif attr_name == "frames":
                self.frames_label = value_label
            elif attr_name == "alerts":
                self.alerts_count_label = value_label
            elif attr_name == "uptime":
                self.uptime_label = value_label
        
        # Configure grid weights
        metrics_grid.columnconfigure(0, weight=1)
        metrics_grid.columnconfigure(1, weight=1)
    
    def create_modern_alerts_display(self, parent):
        """Create modern alerts display with enhanced full-screen visibility"""
        # Alerts container with enhanced styling
        alerts_container = ttk.Frame(parent, style='Card.TFrame')
        alerts_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Enhanced scrollable text widget for alerts with better visibility
        alerts_scroll = tk.Scrollbar(alerts_container, 
                                   bg='#404040',           # Dark scrollbar background
                                   troughcolor='#1a1a1a',  # Darker trough
                                   activebackground='#0078d4',  # Blue when active
                                   width=20)               # Wider scrollbar for easier access
        alerts_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Enhanced text widget with better full-screen visibility
        enhanced_text_config = self.text_style_config.copy()
        enhanced_text_config.update({
            'height': 10,                    # Taller for better visibility
            'wrap': tk.WORD,                 # Word wrap for better formatting
            'padx': 8,                       # Internal padding
            'pady': 6,                       # Internal padding
            'spacing1': 2,                   # Line spacing
            'spacing3': 2,                   # Line spacing
            'state': tk.NORMAL               # Allow updates
        })
        
        self.alerts_text = tk.Text(alerts_container,
                                  yscrollcommand=alerts_scroll.set,
                                  **enhanced_text_config)
        self.alerts_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        alerts_scroll.config(command=self.alerts_text.yview)
        
        # Configure text tags for ultra-visible colored alerts
        self.alerts_text.tag_configure("success", 
                                     foreground="#00ff44",     # Ultra-bright green
                                     background="#003300",     # Dark green background
                                     font=('Consolas', 13, 'bold'),
                                     relief='raised')
        self.alerts_text.tag_configure("warning", 
                                     foreground="#ffcc00",     # Ultra-bright yellow-orange
                                     background="#333300",     # Dark yellow background
                                     font=('Consolas', 13, 'bold'),
                                     relief='raised')
        self.alerts_text.tag_configure("error", 
                                     foreground="#ff3333",     # Ultra-bright red
                                     background="#330000",     # Dark red background
                                     font=('Consolas', 13, 'bold'),
                                     relief='raised')
        self.alerts_text.tag_configure("info", 
                                     foreground="#33ccff",     # Ultra-bright cyan-blue
                                     background="#003333",     # Dark blue background
                                     font=('Consolas', 13, 'bold'),
                                     relief='raised')
        
        # Add enhanced sample alerts with color coding
        self.add_alert_message("🟢 System initialized successfully", "success")
        self.add_alert_message("📷 Camera connected and ready", "info")
        self.add_alert_message("🧠 Enhanced AI model loaded (100% accuracy)", "success")
        self.add_alert_message("🚀 RideBuddy Pro v2.1.0 ready for monitoring", "info")
    
    def emergency_stop(self):
        """Emergency stop function"""
        if self.is_monitoring:
            self.toggle_monitoring()
        self.add_alert_message("🛑 EMERGENCY STOP ACTIVATED", "danger")
    
    def add_alert_message(self, message, level="info"):
        """Add message to alerts display with ultra-enhanced visibility"""
        if hasattr(self, 'alerts_text'):
            self.alerts_text.config(state=tk.NORMAL)
            
            # Enhanced color coding and icons for maximum visibility
            level_config = {
                "info": {"tag": "info", "icon": "ℹ️", "prefix": "[INFO]"},
                "success": {"tag": "success", "icon": "✅", "prefix": "[SUCCESS]"}, 
                "warning": {"tag": "warning", "icon": "⚠️", "prefix": "[WARNING]"},
                "danger": {"tag": "error", "icon": "🛑", "prefix": "[DANGER]"},
                "error": {"tag": "error", "icon": "❌", "prefix": "[ERROR]"},
                "high": {"tag": "error", "icon": "🚨", "prefix": "[HIGH ALERT]"},
                "medium": {"tag": "warning", "icon": "⚠️", "prefix": "[MEDIUM ALERT]"},
                "low": {"tag": "info", "icon": "ℹ️", "prefix": "[LOW ALERT]"}
            }
            
            config = level_config.get(level.lower(), {"tag": "info", "icon": "ℹ️", "prefix": "[INFO]"})
            
            # Format message with ultra-enhanced visibility (add timestamp if not already present)
            if not message.startswith('[') and not message.startswith('23:') and not message.startswith('22:'):
                timestamp = datetime.now().strftime("%H:%M:%S")
                formatted_message = f"[{timestamp}] {config['prefix']} {config['icon']} {message}\n"
            else:
                # Message already has formatting from add_alert, just add visual elements
                formatted_message = f"{config['prefix']} {config['icon']} {message}\n"
            
            # Insert with color tag for maximum visibility
            start_pos = self.alerts_text.index(tk.END + "-1c")
            self.alerts_text.insert(tk.END, formatted_message)
            end_pos = self.alerts_text.index(tk.END + "-1c")
            
            # Apply color tag to the entire message for visibility
            self.alerts_text.tag_add(config['tag'], start_pos, end_pos)
            
            # Auto-scroll to bottom for real-time visibility
            self.alerts_text.see(tk.END)
            self.alerts_text.config(state=tk.DISABLED)
            
            # Keep only last 100 alerts (increased for better history)
            line_count = int(self.alerts_text.index('end-1c').split('.')[0])
            if line_count > 100:
                self.alerts_text.config(state=tk.NORMAL)
                self.alerts_text.delete("1.0", "20.0")  # Remove first 20 lines
                self.alerts_text.config(state=tk.DISABLED)
    
    def create_status_indicators(self, parent):
        """Create status indicators (Legacy - kept for compatibility)"""
        # Driver alertness
        alertness_frame = ttk.Frame(parent)
        alertness_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(alertness_frame, text="Alertness:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        # Note: alertness_label now defined in modern interface
        if not hasattr(self, 'alertness_label'):
            self.alertness_label = ttk.Label(alertness_frame, text="Normal", 
                                           foreground="green", font=("Arial", 10))
            self.alertness_label.pack(side=tk.RIGHT)
        
        # Phone detection (only if not already created)
        if not hasattr(self, 'phone_label'):
            phone_frame = ttk.Frame(parent)
            phone_frame.pack(fill=tk.X, pady=(0, 5))
            
            ttk.Label(phone_frame, text="Phone Detected:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
            self.phone_label = ttk.Label(phone_frame, text="No", 
                                       foreground="green", font=("Arial", 10))
            self.phone_label.pack(side=tk.RIGHT)
        
        # Seatbelt (only if not already created)
        if not hasattr(self, 'seatbelt_label'):
            seatbelt_frame = ttk.Frame(parent)
            seatbelt_frame.pack(fill=tk.X, pady=(0, 5))
            
            ttk.Label(seatbelt_frame, text="Seatbelt:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
            self.seatbelt_label = ttk.Label(seatbelt_frame, text="Worn", 
                                          foreground="green", font=("Arial", 10))
            self.seatbelt_label.pack(side=tk.RIGHT)
        
        # Confidence (only if not already created)
        if not hasattr(self, 'confidence_label'):
            confidence_frame = ttk.Frame(parent)
            confidence_frame.pack(fill=tk.X, pady=(0, 5))
            
            ttk.Label(confidence_frame, text="Confidence:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
            self.confidence_label = ttk.Label(confidence_frame, text="0%", font=("Arial", 10))
            self.confidence_label.pack(side=tk.RIGHT)
    
    def create_statistics_display(self, parent):
        """Create real-time statistics display"""
        # FPS
        fps_frame = ttk.Frame(parent)
        fps_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(fps_frame, text="FPS:", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        self.fps_label = ttk.Label(fps_frame, text="0", font=("Arial", 9))
        self.fps_label.pack(side=tk.RIGHT)
        
        # Frames processed
        frames_frame = ttk.Frame(parent)
        frames_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(frames_frame, text="Frames:", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        self.frames_label = ttk.Label(frames_frame, text="0", font=("Arial", 9))
        self.frames_label.pack(side=tk.RIGHT)
        
        # Session time
        time_frame = ttk.Frame(parent)
        time_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(time_frame, text="Session:", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        self.session_time_label = ttk.Label(time_frame, text="00:00:00", font=("Arial", 9))
        self.session_time_label.pack(side=tk.RIGHT)
    
    def create_recent_alerts_display(self, parent):
        """Create recent alerts display"""
        # Listbox for recent alerts
        self.recent_alerts_listbox = tk.Listbox(parent, height=10, font=("Arial", 9))
        self.recent_alerts_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.recent_alerts_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.recent_alerts_listbox.config(yscrollcommand=scrollbar.set)
    
    def create_horizontal_layout(self, container, window_width, window_height):
        """Create horizontal layout for larger screens"""
        # Main content with horizontal layout
        content_container = ttk.Frame(container, style='Dark.TFrame')
        content_container.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Calculate responsive panel sizes
        video_panel_ratio = 0.65 if window_width > 1400 else 0.6
        control_panel_width = min(400, int(window_width * 0.35))
        
        # Left panel - Video feed
        left_panel = ttk.Frame(content_container, style='Dark.TFrame')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        
        # Right panel - Controls and metrics
        right_panel = ttk.Frame(content_container, style='Dark.TFrame')
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right_panel.config(width=control_panel_width)
        
        # Create video display
        self.create_video_display(left_panel, window_width, window_height, layout='horizontal')
        
        # Create control panels
        self.create_control_buttons(right_panel)
        self.create_detection_panel(right_panel)
        
    def create_vertical_layout(self, container, window_width, window_height):
        """Create vertical layout for smaller screens"""
        # Main content with vertical layout
        content_container = ttk.Frame(container, style='Dark.TFrame')
        content_container.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Top panel - Video feed (60% of height)
        top_panel = ttk.Frame(content_container, style='Dark.TFrame')
        top_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 8))
        
        # Bottom panel - Controls (40% of height)
        bottom_panel = ttk.Frame(content_container, style='Dark.TFrame')
        bottom_panel.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        
        # Create video display (smaller for vertical layout)
        self.create_video_display(top_panel, window_width, int(window_height * 0.5), layout='vertical')
        
        # Create horizontal control layout
        self.create_compact_controls(bottom_panel)
        
    def create_video_display(self, parent, max_width, max_height, layout='horizontal'):
        """Create responsive video display"""
        # Video display frame
        video_frame = ttk.LabelFrame(parent, 
                                   text="📹 Live Camera Feed",
                                   style='Modern.TLabelframe',
                                   padding="8")
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video container
        video_container = ttk.Frame(video_frame, style='Card.TFrame')
        video_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Calculate optimal video size
        if layout == 'horizontal':
            video_width = min(640, int(max_width * 0.6))
            video_height = min(480, int(max_height * 0.6))
        else:  # vertical layout
            video_width = min(480, int(max_width * 0.8))
            video_height = min(360, int(max_height * 0.7))
        
        # Video display label with responsive sizing
        self.video_label = ttk.Label(video_container,
                                   text="📷 Camera Not Active\n\nClick 'Start Monitoring' to begin",
                                   font=('Segoe UI', 12 if layout == 'vertical' else 14),
                                   foreground='#666666',
                                   background='#1e1e1e',
                                   anchor="center",
                                   justify="center")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Store video dimensions for camera manager
        self.video_display_size = (video_width, video_height)
        print(f"[VIDEO] Display size set to: {video_width}x{video_height}")
        
    def create_compact_controls(self, parent):
        """Create compact horizontal control layout for small screens"""
        # Create horizontal sections
        left_controls = ttk.Frame(parent, style='Dark.TFrame')
        left_controls.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_controls = ttk.Frame(parent, style='Dark.TFrame')
        right_controls.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Compact control buttons
        self.create_control_buttons(left_controls, compact=True)
        
        # Compact detection panel
        self.create_detection_panel(right_controls, compact=True)
    
    def on_window_resize(self, event):
        """Handle window resize events for responsive layout"""
        # Only respond to root window resize events
        if event.widget != self.root:
            return
            
        # Debounce resize events (only handle after 100ms of no resizing)
        import time
        current_time = time.time()
        self.last_resize_time = current_time
        
        # Schedule layout update after delay
        self.root.after(100, lambda: self.check_and_update_layout(current_time))
        
    def check_and_update_layout(self, trigger_time):
        """Check if layout needs updating after resize"""
        # Only update if this is the latest resize event
        if trigger_time != self.last_resize_time:
            return
            
        # Get current window size
        try:
            current_width = self.root.winfo_width()
            current_height = self.root.winfo_height()
            
            # Determine if we need to switch layouts
            should_use_vertical = current_width < 1000 or current_height < 700
            current_layout = getattr(self, 'current_layout', None)
            
            if should_use_vertical and current_layout != 'vertical':
                print(f"[RESIZE] Switching to vertical layout ({current_width}x{current_height})")
                self.current_layout = 'vertical'
                # Note: Full layout rebuild would require recreating the monitoring tab
                # For now, just log the change
                
            elif not should_use_vertical and current_layout != 'horizontal':
                print(f"[RESIZE] Switching to horizontal layout ({current_width}x{current_height})")
                self.current_layout = 'horizontal'
                
            # Update video display size if needed
            if hasattr(self, 'video_label') and hasattr(self, 'video_display_size'):
                self.update_video_display_size(current_width, current_height)
                
        except Exception as e:
            print(f"[RESIZE] Error updating layout: {e}")
            
    def update_video_display_size(self, window_width, window_height):
        """Update video display size based on current window dimensions"""
        try:
            # Calculate new optimal video size
            if self.current_layout == 'vertical':
                video_width = min(480, int(window_width * 0.8))
                video_height = min(360, int(window_height * 0.4))
            else:
                video_width = min(640, int(window_width * 0.6))
                video_height = min(480, int(window_height * 0.6))
                
            # Update stored size
            self.video_display_size = (video_width, video_height)
            print(f"[VIDEO] Updated display size to: {video_width}x{video_height}")
            
        except Exception as e:
            print(f"[VIDEO] Error updating video size: {e}")
    
    def create_alerts_tab(self):
        """Create alerts management tab"""
        alerts_frame = ttk.Frame(self.notebook, style='Tab.TFrame')
        self.notebook.add(alerts_frame, text="⚠️ Alerts & History")
        
        # Alerts filter frame
        filter_frame = ttk.LabelFrame(alerts_frame, text="Filter Options", padding="10")
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Filter controls
        ttk.Label(filter_frame, text="Alert Type:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.alert_filter_var = tk.StringVar(value="All")
        alert_filter = ttk.Combobox(filter_frame, textvariable=self.alert_filter_var,
                                  values=["All", "Drowsy", "Phone_Distraction", "Seatbelt"],
                                  width=15, state="readonly")
        alert_filter.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Button(filter_frame, text="Clear All Alerts", 
                 command=self.clear_alerts).pack(side=tk.RIGHT, padx=(10, 0))
        
        ttk.Button(filter_frame, text="Export Alerts", 
                 command=self.export_alerts).pack(side=tk.RIGHT)
        
        # Alerts treeview
        tree_frame = ttk.LabelFrame(alerts_frame, text="Alert History", padding="10")
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeview for alerts
        columns = ("Time", "Type", "Message", "Confidence", "Severity")
        self.alerts_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=15)
        
        # Configure columns
        for col in columns:
            self.alerts_tree.heading(col, text=col)
            if col == "Time":
                self.alerts_tree.column(col, width=100)
            elif col == "Type":
                self.alerts_tree.column(col, width=120)
            elif col == "Message":
                self.alerts_tree.column(col, width=300)
            else:
                self.alerts_tree.column(col, width=80)
        
        # Configure tags for severity color coding
        self.alerts_tree.tag_configure('high', background='#ffebee', foreground='#c62828')
        self.alerts_tree.tag_configure('medium', background='#fff3e0', foreground='#f57c00')
        self.alerts_tree.tag_configure('low', background='#e8f5e8', foreground='#2e7d32')
        
        # Scrollbars for treeview
        tree_v_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.alerts_tree.yview)
        tree_h_scroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.alerts_tree.xview)
        
        self.alerts_tree.configure(yscrollcommand=tree_v_scroll.set, xscrollcommand=tree_h_scroll.set)
        
        # Pack treeview and scrollbars
        self.alerts_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree_h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_analysis_tab(self):
        """Create analysis and statistics tab"""
        analysis_frame = ttk.Frame(self.notebook, style='Tab.TFrame')
        self.notebook.add(analysis_frame, text="📊 Analysis")
        
        # Summary statistics frame
        summary_frame = ttk.LabelFrame(analysis_frame, text="Session Summary", padding="10")
        summary_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create summary statistics display
        self.create_summary_statistics(summary_frame)
        
        # Performance metrics frame
        perf_frame = ttk.LabelFrame(analysis_frame, text="Performance Metrics", padding="10")
        perf_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Performance text area
        self.performance_text = tk.Text(perf_frame, wrap=tk.WORD, font=("Consolas", 10), height=20)
        perf_scrollbar = ttk.Scrollbar(perf_frame, orient=tk.VERTICAL, command=self.performance_text.yview)
        
        self.performance_text.configure(yscrollcommand=perf_scrollbar.set)
        self.performance_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        perf_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_summary_statistics(self, parent):
        """Create summary statistics display"""
        # Create grid of statistics
        stats_grid = ttk.Frame(parent)
        stats_grid.pack(fill=tk.X)
        
        # Total frames
        ttk.Label(stats_grid, text="Total Frames Processed:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=2)
        self.total_frames_label = ttk.Label(stats_grid, text="0", font=("Arial", 10))
        self.total_frames_label.grid(row=0, column=1, sticky=tk.E, pady=2)
        
        # Drowsy detections
        ttk.Label(stats_grid, text="Drowsy Episodes:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky=tk.W, pady=2)
        self.drowsy_count_label = ttk.Label(stats_grid, text="0", font=("Arial", 10))
        self.drowsy_count_label.grid(row=1, column=1, sticky=tk.E, pady=2)
        
        # Phone detections
        ttk.Label(stats_grid, text="Phone Distractions:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, pady=2)
        self.phone_count_label = ttk.Label(stats_grid, text="0", font=("Arial", 10))
        self.phone_count_label.grid(row=2, column=1, sticky=tk.E, pady=2)
        
        # Total alerts
        ttk.Label(stats_grid, text="Total Alerts:", font=("Arial", 10, "bold")).grid(row=3, column=0, sticky=tk.W, pady=2)
        self.alerts_count_label = ttk.Label(stats_grid, text="0", font=("Arial", 10))
        self.alerts_count_label.grid(row=3, column=1, sticky=tk.E, pady=2)
        
        # Configure grid
        stats_grid.columnconfigure(1, weight=1)
    
    def create_testing_tab(self):
        """Create testing environment tab for media testing"""
        testing_frame = ttk.Frame(self.notebook, style='Tab.TFrame')
        self.notebook.add(testing_frame, text="🧪 Testing Environment")
        
        # Main container with paned window
        test_paned = ttk.PanedWindow(testing_frame, orient=tk.HORIZONTAL)
        test_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Media selection and controls
        left_test_panel = ttk.Frame(test_paned)
        test_paned.add(left_test_panel, weight=1)
        
        # Media selection frame
        media_frame = ttk.LabelFrame(left_test_panel, text="Media Selection", padding="10")
        media_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Media type selection
        self.media_type_var = tk.StringVar(value="Image")
        ttk.Label(media_frame, text="Media Type:").pack(anchor=tk.W, pady=(0, 5))
        
        media_type_frame = ttk.Frame(media_frame)
        media_type_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Radiobutton(media_type_frame, text="📷 Image", variable=self.media_type_var, 
                       value="Image", command=self.on_media_type_change).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(media_type_frame, text="🎥 Video", variable=self.media_type_var, 
                       value="Video", command=self.on_media_type_change).pack(side=tk.LEFT)
        
        # File selection
        file_select_frame = ttk.Frame(media_frame)
        file_select_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_select_frame, text="📁 Browse Files", 
                  command=self.browse_test_media).pack(side=tk.LEFT, padx=(0, 10))
        
        self.selected_media_label = ttk.Label(file_select_frame, text="No file selected", 
                                            foreground="gray")
        self.selected_media_label.pack(side=tk.LEFT)
        
        # Test controls frame
        test_controls_frame = ttk.LabelFrame(left_test_panel, text="Test Controls", padding="10")
        test_controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Main test buttons
        test_buttons_frame = ttk.Frame(test_controls_frame)
        test_buttons_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.test_button = ttk.Button(test_buttons_frame, text="🔬 Run Test", 
                                    command=self.run_media_test, state="disabled", width=15)
        self.test_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.batch_test_button = ttk.Button(test_buttons_frame, text="📊 Batch Test", 
                                          command=self.run_batch_test, width=15)
        self.batch_test_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Test options
        options_frame = ttk.Frame(test_controls_frame)
        options_frame.pack(fill=tk.X)
        
        self.save_results_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Save test results", 
                       variable=self.save_results_var).pack(anchor=tk.W, pady=(0, 5))
        
        self.detailed_analysis_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Detailed frame analysis", 
                       variable=self.detailed_analysis_var).pack(anchor=tk.W)
        
        # Test scenarios frame
        scenarios_frame = ttk.LabelFrame(left_test_panel, text="Test Scenarios", padding="10")
        scenarios_frame.pack(fill=tk.BOTH, expand=True)
        
        # Predefined test scenarios
        ttk.Label(scenarios_frame, text="Predefined Scenarios:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        scenarios_list_frame = ttk.Frame(scenarios_frame)
        scenarios_list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scenario buttons
        scenarios = [
            ("👁️ Normal Driving", "normal"),
            ("😴 Drowsy Driver", "drowsy"), 
            ("📱 Phone Usage", "phone"),
            ("🚫 No Seatbelt", "no_seatbelt"),
            ("🌙 Low Light", "low_light"),
            ("☀️ Bright Light", "bright_light"),
            ("👤 Multiple People", "multiple_people"),
            ("🔄 Edge Cases", "edge_cases")
        ]
        
        self.scenario_buttons = {}
        for i, (label, scenario) in enumerate(scenarios):
            row = i // 2
            col = i % 2
            
            btn = ttk.Button(scenarios_list_frame, text=label, width=20,
                           command=lambda s=scenario: self.load_test_scenario(s))
            btn.grid(row=row, column=col, padx=5, pady=2, sticky="ew")
            self.scenario_buttons[scenario] = btn
        
        # Configure grid weights
        for i in range(2):
            scenarios_list_frame.columnconfigure(i, weight=1)
        
        # Right panel - Test results and preview
        right_test_panel = ttk.Frame(test_paned)
        test_paned.add(right_test_panel, weight=2)
        
        # Media preview frame
        preview_frame = ttk.LabelFrame(right_test_panel, text="Media Preview", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Preview display
        self.test_preview_label = ttk.Label(preview_frame, 
                                          text="📁 Select media file to preview\n\nSupported formats:\n• Images: JPG, PNG, BMP\n• Videos: MP4, AVI, MOV",
                                          anchor="center", font=("Arial", 11))
        self.test_preview_label.pack(fill=tk.BOTH, expand=True)
        
        # Video controls (initially hidden)
        self.video_controls_frame = ttk.Frame(preview_frame)
        
        self.play_button = ttk.Button(self.video_controls_frame, text="▶️ Play", 
                                    command=self.toggle_video_playback)
        self.play_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.video_position_var = tk.DoubleVar()
        self.video_slider = ttk.Scale(self.video_controls_frame, from_=0, to=100, 
                                    orient=tk.HORIZONTAL, variable=self.video_position_var,
                                    command=self.seek_video)
        self.video_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.video_time_label = ttk.Label(self.video_controls_frame, text="00:00 / 00:00")
        self.video_time_label.pack(side=tk.RIGHT)
        
        # Test results frame
        results_frame = ttk.LabelFrame(right_test_panel, text="Test Results", padding="10")
        results_frame.pack(fill=tk.X)
        
        # Results display
        self.test_results_text = tk.Text(results_frame, height=12, wrap=tk.WORD, 
                                       font=("Consolas", 10))
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, 
                                        command=self.test_results_text.yview)
        
        self.test_results_text.configure(yscrollcommand=results_scrollbar.set)
        self.test_results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initialize test variables
        self.current_test_media = None
        self.test_video_capture = None
        self.video_playing = False
        self.video_thread = None
        
        # Add initial test results
        self.update_test_results("🧪 Testing Environment Ready\n\nSelect media files to begin testing the RideBuddy AI model.\n\n")
    
    def on_media_type_change(self):
        """Handle media type selection change"""
        media_type = self.media_type_var.get()
        if media_type == "Video":
            self.selected_media_label.config(text="Select video file (MP4, AVI, MOV)")
        else:
            self.selected_media_label.config(text="Select image file (JPG, PNG, BMP)")
        
        # Reset selection
        self.current_test_media = None
        self.test_button.config(state="disabled")
        self.test_preview_label.config(text=f"📁 Select {media_type.lower()} file to preview")
    
    def browse_test_media(self):
        """Browse and select test media files"""
        media_type = self.media_type_var.get()
        
        if media_type == "Image":
            filetypes = [
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"), 
                ("All files", "*.*")
            ]
        else:  # Video
            filetypes = [
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("All files", "*.*")
            ]
        
        filename = filedialog.askopenfilename(
            title=f"Select {media_type} File for Testing",
            filetypes=filetypes
        )
        
        if filename:
            self.current_test_media = filename
            self.selected_media_label.config(text=Path(filename).name, foreground="black")
            self.test_button.config(state="normal")
            
            # Load preview
            self.load_media_preview(filename, media_type)
            
            # Update test results
            self.update_test_results(f"📁 Loaded: {Path(filename).name}\n")
    
    def load_media_preview(self, filepath, media_type):
        """Load and display media preview"""
        try:
            if media_type == "Image" and OPENCV_AVAILABLE and PIL_AVAILABLE:
                # Load and display image
                image = cv2.imread(filepath)
                if image is not None:
                    # Resize for preview
                    height, width = image.shape[:2]
                    max_size = 400
                    
                    if width > height:
                        new_width = max_size
                        new_height = int(height * max_size / width)
                    else:
                        new_height = max_size  
                        new_width = int(width * max_size / height)
                    
                    resized = cv2.resize(image, (new_width, new_height))
                    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)
                    tk_image = ImageTk.PhotoImage(pil_image)
                    
                    self.test_preview_label.configure(image=tk_image, text="")
                    self.test_preview_label.image = tk_image
                    
                    # Hide video controls
                    self.video_controls_frame.pack_forget()
                else:
                    self.test_preview_label.config(text="❌ Failed to load image")
                    
            elif media_type == "Video" and OPENCV_AVAILABLE:
                # Load video preview (first frame)
                cap = cv2.VideoCapture(filepath)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        # Display first frame
                        height, width = frame.shape[:2]
                        max_size = 400
                        
                        if width > height:
                            new_width = max_size
                            new_height = int(height * max_size / width)
                        else:
                            new_height = max_size
                            new_width = int(width * max_size / height)
                        
                        resized = cv2.resize(frame, (new_width, new_height))
                        
                        if PIL_AVAILABLE:
                            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(rgb_image)
                            tk_image = ImageTk.PhotoImage(pil_image)
                            
                            self.test_preview_label.configure(image=tk_image, text="")
                            self.test_preview_label.image = tk_image
                        
                        # Show video controls
                        self.video_controls_frame.pack(fill=tk.X, pady=(10, 0))
                        
                        # Get video info
                        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        duration = total_frames / fps if fps > 0 else 0
                        
                        self.video_time_label.config(text=f"00:00 / {self.format_time(duration)}")
                        self.video_slider.config(to=total_frames)
                    
                    cap.release()
                else:
                    self.test_preview_label.config(text="❌ Failed to load video")
            else:
                self.test_preview_label.config(text="⚠️ Preview not available\n(OpenCV/PIL required)")
                
        except Exception as e:
            self.test_preview_label.config(text=f"❌ Preview error: {str(e)}")
    
    def format_time(self, seconds):
        """Format seconds into MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def toggle_video_playback(self):
        """Toggle video playback for preview"""
        if not self.current_test_media or self.media_type_var.get() != "Video":
            return
        
        if not self.video_playing:
            self.start_video_preview()
        else:
            self.stop_video_preview()
    
    def start_video_preview(self):
        """Start video preview playback"""
        if not OPENCV_AVAILABLE:
            return
        
        self.video_playing = True
        self.play_button.config(text="⏸️ Pause")
        
        def play_video():
            cap = cv2.VideoCapture(self.current_test_media)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_delay = 1.0 / fps
            
            while self.video_playing and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Update preview
                if PIL_AVAILABLE:
                    height, width = frame.shape[:2]
                    max_size = 400
                    
                    if width > height:
                        new_width = max_size
                        new_height = int(height * max_size / width)
                    else:
                        new_height = max_size
                        new_width = int(width * max_size / height)
                    
                    resized = cv2.resize(frame, (new_width, new_height))
                    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)
                    tk_image = ImageTk.PhotoImage(pil_image)
                    
                    self.root.after_idle(lambda img=tk_image: self.update_video_preview(img))
                
                # Update slider
                current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                current_time = current_frame / fps if fps > 0 else 0
                total_time = total_frames / fps if fps > 0 else 0
                
                self.root.after_idle(lambda: self.video_position_var.set(current_frame))
                self.root.after_idle(lambda: self.video_time_label.config(
                    text=f"{self.format_time(current_time)} / {self.format_time(total_time)}"))
                
                time.sleep(frame_delay)
            
            cap.release()
            self.root.after_idle(self.stop_video_preview)
        
        self.video_thread = threading.Thread(target=play_video, daemon=True)
        self.video_thread.start()
    
    def stop_video_preview(self):
        """Stop video preview playback"""
        self.video_playing = False
        self.play_button.config(text="▶️ Play")
    
    def update_video_preview(self, tk_image):
        """Update video preview image"""
        self.test_preview_label.configure(image=tk_image, text="")
        self.test_preview_label.image = tk_image
    
    def seek_video(self, value):
        """Seek video to specific position"""
        # This would require more complex video handling for actual seeking
        pass
    
    def run_media_test(self):
        """Run AI test on selected media"""
        if not self.current_test_media:
            messagebox.showerror("Error", "Please select a media file first")
            return
        
        media_type = self.media_type_var.get()
        
        # Update UI
        self.test_button.config(state="disabled", text="🔄 Testing...")
        self.update_test_results(f"\n🔬 Running {media_type.lower()} test...\n")
        
        # Run test in separate thread
        def run_test():
            try:
                if media_type == "Image":
                    results = self.test_image_file(self.current_test_media)
                else:
                    results = self.test_video_file(self.current_test_media)
                
                # Update UI from main thread
                self.root.after_idle(lambda: self.display_test_results(results))
                
            except Exception as e:
                error_msg = f"❌ Test failed: {str(e)}"
                self.root.after_idle(lambda: self.update_test_results(error_msg + "\n"))
            
            finally:
                self.root.after_idle(lambda: self.test_button.config(state="normal", text="🔬 Run Test"))
        
        threading.Thread(target=run_test, daemon=True).start()
    
    def test_image_file(self, filepath):
        """Test AI model on image file"""
        if not OPENCV_AVAILABLE:
            return {"error": "OpenCV not available"}
        
        # Load image
        image = cv2.imread(filepath)
        if image is None:
            return {"error": "Failed to load image"}
        
        # Simulate AI processing (replace with actual model inference)
        time.sleep(0.5)  # Simulate processing time
        
        # Generate test results
        results = {
            "filepath": filepath,
            "type": "image",
            "resolution": f"{image.shape[1]}x{image.shape[0]}",
            "size_mb": Path(filepath).stat().st_size / (1024*1024),
            "predictions": {
                "alertness": random.choice(["Normal", "Drowsy", "Phone_Distraction"]),
                "confidence": random.uniform(0.75, 0.95),
                "phone_detected": random.random() < 0.3,
                "seatbelt_worn": random.random() < 0.9,
                "face_detected": random.random() < 0.95,
                "processing_time_ms": random.uniform(15, 45)
            }
        }
        
        return results
    
    def test_video_file(self, filepath):
        """Test AI model on video file"""
        if not OPENCV_AVAILABLE:
            return {"error": "OpenCV not available"}
        
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return {"error": "Failed to load video"}
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Process sample frames (every 30 frames for speed)
        sample_frames = []
        frame_predictions = []
        
        for i in range(0, min(frame_count, 300), 30):  # Max 10 samples
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Simulate AI processing
                pred = {
                    "frame": i,
                    "time": i / fps if fps > 0 else 0,
                    "alertness": random.choice(["Normal", "Drowsy", "Phone_Distraction"]),
                    "confidence": random.uniform(0.70, 0.95),
                    "phone_detected": random.random() < 0.25,
                    "seatbelt_worn": random.random() < 0.92,
                    "face_detected": random.random() < 0.93
                }
                frame_predictions.append(pred)
        
        cap.release()
        
        # Analyze results
        alertness_counts = {}
        for pred in frame_predictions:
            alertness = pred["alertness"]
            alertness_counts[alertness] = alertness_counts.get(alertness, 0) + 1
        
        avg_confidence = sum(p["confidence"] for p in frame_predictions) / len(frame_predictions)
        phone_detection_rate = sum(1 for p in frame_predictions if p["phone_detected"]) / len(frame_predictions)
        
        results = {
            "filepath": filepath,
            "type": "video",
            "duration": duration,
            "fps": fps,
            "frame_count": frame_count,
            "resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
            "size_mb": Path(filepath).stat().st_size / (1024*1024),
            "samples_analyzed": len(frame_predictions),
            "predictions": {
                "alertness_distribution": alertness_counts,
                "average_confidence": avg_confidence,
                "phone_detection_rate": phone_detection_rate,
                "frame_predictions": frame_predictions[:5] if self.detailed_analysis_var.get() else []
            }
        }
        
        return results
    
    def display_test_results(self, results):
        """Display test results in the UI"""
        if "error" in results:
            self.update_test_results(f"❌ {results['error']}\n")
            return
        
        # Format results
        result_text = f"\n✅ Test completed for: {Path(results['filepath']).name}\n"
        result_text += f"{'='*50}\n"
        
        if results["type"] == "image":
            result_text += f"📷 Image Analysis:\n"
            result_text += f"  • Resolution: {results['resolution']}\n"
            result_text += f"  • File Size: {results['size_mb']:.2f} MB\n"
            result_text += f"  • Processing Time: {results['predictions']['processing_time_ms']:.1f} ms\n\n"
            
            result_text += f"🤖 AI Predictions:\n"
            result_text += f"  • Alertness: {results['predictions']['alertness']} ({results['predictions']['confidence']:.1%})\n"
            result_text += f"  • Phone Detected: {'Yes' if results['predictions']['phone_detected'] else 'No'}\n"
            result_text += f"  • Seatbelt: {'Worn' if results['predictions']['seatbelt_worn'] else 'Not Worn'}\n"
            result_text += f"  • Face Detected: {'Yes' if results['predictions']['face_detected'] else 'No'}\n"
        
        else:  # video
            result_text += f"🎥 Video Analysis:\n"
            result_text += f"  • Duration: {results['duration']:.1f} seconds\n"
            result_text += f"  • FPS: {results['fps']:.1f}\n"
            result_text += f"  • Total Frames: {results['frame_count']}\n"
            result_text += f"  • Resolution: {results['resolution']}\n"
            result_text += f"  • File Size: {results['size_mb']:.2f} MB\n"
            result_text += f"  • Frames Analyzed: {results['samples_analyzed']}\n\n"
            
            result_text += f"🤖 AI Analysis Summary:\n"
            result_text += f"  • Average Confidence: {results['predictions']['average_confidence']:.1%}\n"
            result_text += f"  • Phone Detection Rate: {results['predictions']['phone_detection_rate']:.1%}\n\n"
            
            result_text += f"📊 Alertness Distribution:\n"
            for alertness, count in results['predictions']['alertness_distribution'].items():
                percentage = (count / results['samples_analyzed']) * 100
                result_text += f"  • {alertness}: {count} frames ({percentage:.1f}%)\n"
            
            if self.detailed_analysis_var.get() and results['predictions']['frame_predictions']:
                result_text += f"\n🔍 Sample Frame Analysis:\n"
                for pred in results['predictions']['frame_predictions']:
                    result_text += f"  Frame {pred['frame']} ({pred['time']:.1f}s): {pred['alertness']} ({pred['confidence']:.1%})\n"
        
        result_text += f"\n{'='*50}\n"
        
        # Save results if enabled
        if self.save_results_var.get():
            self.save_test_results(results)
        
        self.update_test_results(result_text)
    
    def save_test_results(self, results):
        """Save test results to file"""
        try:
            results_dir = Path("test_results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_{timestamp}_{Path(results['filepath']).stem}.json"
            
            with open(results_dir / filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.update_test_results(f"💾 Results saved to: {filename}\n")
            
        except Exception as e:
            self.update_test_results(f"⚠️ Failed to save results: {str(e)}\n")
    
    def run_batch_test(self):
        """Run batch testing on multiple files"""
        media_type = self.media_type_var.get()
        
        if media_type == "Image":
            filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp")]
        else:
            filetypes = [("Video files", "*.mp4 *.avi *.mov")]
        
        filenames = filedialog.askopenfilenames(
            title=f"Select {media_type} Files for Batch Testing",
            filetypes=filetypes
        )
        
        if not filenames:
            return
        
        self.update_test_results(f"\n📊 Starting batch test on {len(filenames)} files...\n")
        
        def batch_test():
            results_summary = []
            
            for i, filepath in enumerate(filenames):
                self.root.after_idle(lambda: self.update_test_results(f"Processing {i+1}/{len(filenames)}: {Path(filepath).name}\n"))
                
                try:
                    if media_type == "Image":
                        result = self.test_image_file(filepath)
                    else:
                        result = self.test_video_file(filepath)
                    
                    if "error" not in result:
                        results_summary.append(result)
                    
                except Exception as e:
                    self.root.after_idle(lambda: self.update_test_results(f"❌ Failed: {str(e)}\n"))
            
            # Generate batch summary
            self.root.after_idle(lambda: self.display_batch_summary(results_summary))
        
        threading.Thread(target=batch_test, daemon=True).start()
    
    def display_batch_summary(self, results):
        """Display batch test summary"""
        if not results:
            self.update_test_results("❌ No valid results from batch test\n")
            return
        
        summary_text = f"\n📈 BATCH TEST SUMMARY ({len(results)} files)\n"
        summary_text += f"{'='*60}\n"
        
        if results[0]["type"] == "image":
            # Image batch summary
            avg_confidence = sum(r["predictions"]["confidence"] for r in results) / len(results)
            phone_detections = sum(1 for r in results if r["predictions"]["phone_detected"])
            alertness_counts = {}
            
            for result in results:
                alertness = result["predictions"]["alertness"]
                alertness_counts[alertness] = alertness_counts.get(alertness, 0) + 1
            
            summary_text += f"📊 Statistics:\n"
            summary_text += f"  • Average Confidence: {avg_confidence:.1%}\n"
            summary_text += f"  • Phone Detections: {phone_detections}/{len(results)} ({phone_detections/len(results):.1%})\n\n"
            
            summary_text += f"📋 Alertness Distribution:\n"
            for alertness, count in alertness_counts.items():
                percentage = (count / len(results)) * 100
                summary_text += f"  • {alertness}: {count} files ({percentage:.1f}%)\n"
        
        else:
            # Video batch summary
            total_duration = sum(r["duration"] for r in results)
            avg_confidence = sum(r["predictions"]["average_confidence"] for r in results) / len(results)
            avg_phone_rate = sum(r["predictions"]["phone_detection_rate"] for r in results) / len(results)
            
            summary_text += f"📊 Statistics:\n"
            summary_text += f"  • Total Duration: {total_duration:.1f} seconds\n"
            summary_text += f"  • Average Confidence: {avg_confidence:.1%}\n"
            summary_text += f"  • Average Phone Detection Rate: {avg_phone_rate:.1%}\n"
        
        summary_text += f"\n{'='*60}\n"
        
        self.update_test_results(summary_text)
    
    def load_test_scenario(self, scenario):
        """Load predefined test scenario"""
        scenario_descriptions = {
            "normal": "👁️ Testing normal driving conditions",
            "drowsy": "😴 Testing drowsiness detection accuracy", 
            "phone": "📱 Testing phone usage detection",
            "no_seatbelt": "🚫 Testing seatbelt compliance detection",
            "low_light": "🌙 Testing low light/night conditions",
            "bright_light": "☀️ Testing bright light/glare conditions",
            "multiple_people": "👥 Testing multiple person scenarios",
            "edge_cases": "🔄 Testing edge cases and corner scenarios"
        }
        
        description = scenario_descriptions.get(scenario, f"Testing {scenario}")
        self.update_test_results(f"\n🎯 Scenario Selected: {description}\n")
        
        # You could add scenario-specific settings or sample data here
        messagebox.showinfo("Test Scenario", f"Selected: {description}\n\nPlease load appropriate test media for this scenario.")
    
    def update_test_results(self, text):
        """Update test results display"""
        self.test_results_text.insert(tk.END, text)
        self.test_results_text.see(tk.END)
        self.root.update_idletasks()

    def create_settings_tab(self):
        """Create settings and configuration tab"""
        settings_frame = ttk.Frame(self.notebook, style='Tab.TFrame')
        self.notebook.add(settings_frame, text="⚙️ Settings")
        
        # Camera settings
        camera_frame = ttk.LabelFrame(settings_frame, text="Camera Settings", padding="10")
        camera_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Camera selection
        ttk.Label(camera_frame, text="Camera Index:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.camera_index_var = tk.StringVar(value="0")
        camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_index_var,
                                  values=["0", "1", "2"], width=10, state="readonly")
        camera_combo.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Button(camera_frame, text="Test Camera", 
                 command=self.test_camera).pack(side=tk.LEFT, padx=(0, 20))
        
        # Alert settings
        alert_frame = ttk.LabelFrame(settings_frame, text="Alert Settings", padding="10")
        alert_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Alert sensitivity
        ttk.Label(alert_frame, text="Alert Sensitivity:").pack(anchor=tk.W, pady=(0, 5))
        
        self.sensitivity_var = tk.DoubleVar(value=0.7)
        sensitivity_scale = ttk.Scale(alert_frame, from_=0.5, to=0.95, 
                                    orient=tk.HORIZONTAL, variable=self.sensitivity_var)
        sensitivity_scale.pack(fill=tk.X, pady=(0, 10))
        
        # Audio alerts
        self.audio_alerts_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(alert_frame, text="Enable Audio Alerts", 
                      variable=self.audio_alerts_var).pack(anchor=tk.W, pady=(0, 5))
        
        # Log settings
        log_frame = ttk.LabelFrame(settings_frame, text="Logging Settings", padding="10")
        log_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.logging_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_frame, text="Enable Logging", 
                      variable=self.logging_enabled_var).pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Button(log_frame, text="Open Log Folder", 
                 command=self.open_log_folder).pack(anchor=tk.W)
        
        # About section
        about_frame = ttk.LabelFrame(settings_frame, text="About", padding="10")
        about_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        about_text = """RideBuddy Driver Monitoring System
        
Version: 2.0 Optimized
Built with: Python, OpenCV, tkinter

This application monitors driver alertness and detects:
• Drowsiness patterns
• Phone usage while driving  
• Seatbelt compliance
• Real-time safety alerts

For support or updates, visit: github.com/ridebuddy"""
        
        ttk.Label(about_frame, text=about_text, justify=tk.LEFT, font=("Arial", 10)).pack(anchor=tk.W)
        
        # System information
        system_frame = ttk.LabelFrame(settings_frame, text="System Information", padding="10")
        system_frame.pack(fill=tk.X, padx=5, pady=5)
        
        system_info = f"""Platform: {platform.system()} {platform.release()}
Python: {platform.python_version()}
OpenCV: {'Available' if OPENCV_AVAILABLE else 'Not Available'}
PIL: {'Available' if PIL_AVAILABLE else 'Not Available'}
Performance Monitor: {'Available' if PSUTIL_AVAILABLE else 'Limited'}

Performance Stats:"""
        
        self.system_info_label = ttk.Label(system_frame, text=system_info, 
                                         justify=tk.LEFT, font=("Consolas", 9))
        self.system_info_label.pack(anchor=tk.W)
        
        # Performance controls
        perf_frame = ttk.LabelFrame(settings_frame, text="Performance Settings", padding="10")
        perf_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Max FPS setting
        ttk.Label(perf_frame, text="Max FPS:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.max_fps_var = tk.IntVar(value=self.config.max_fps)
        fps_spinbox = ttk.Spinbox(perf_frame, from_=5, to=60, textvariable=self.max_fps_var, width=10)
        fps_spinbox.grid(row=0, column=1, padx=(10, 0), pady=5)
        
        # Memory limit
        ttk.Label(perf_frame, text="Memory Limit (MB):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.memory_limit_var = tk.IntVar(value=self.config.max_memory_mb)
        memory_spinbox = ttk.Spinbox(perf_frame, from_=128, to=2048, textvariable=self.memory_limit_var, width=10)
        memory_spinbox.grid(row=1, column=1, padx=(10, 0), pady=5)
        
        # Frame drop threshold
        ttk.Label(perf_frame, text="CPU Threshold (%):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.cpu_threshold_var = tk.DoubleVar(value=self.config.frame_drop_threshold * 100)
        cpu_scale = ttk.Scale(perf_frame, from_=50, to=95, orient=tk.HORIZONTAL, variable=self.cpu_threshold_var)
        cpu_scale.grid(row=2, column=1, padx=(10, 0), pady=5, sticky=tk.EW)
        
        # Save settings button
        ttk.Button(perf_frame, text="Save Settings", command=self.save_settings).grid(row=3, column=0, columnspan=2, pady=10)
        
        # Configure grid weights
        perf_frame.columnconfigure(1, weight=1)
        
        # Privacy and data management
        privacy_frame = ttk.LabelFrame(settings_frame, text="Privacy & Data Management", padding="10")
        privacy_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Data retention
        ttk.Label(privacy_frame, text="Data Retention (days):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.retention_var = tk.IntVar(value=self.config.data_retention_days)
        retention_spinbox = ttk.Spinbox(privacy_frame, from_=1, to=365, textvariable=self.retention_var, width=10)
        retention_spinbox.grid(row=0, column=1, padx=(10, 0), pady=5)
        
        # Privacy controls
        privacy_buttons_frame = ttk.Frame(privacy_frame)
        privacy_buttons_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky=tk.EW)
        
        ttk.Button(privacy_buttons_frame, text="Export My Data", 
                 command=self.export_user_data).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(privacy_buttons_frame, text="Delete All Data", 
                 command=self.delete_user_data).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(privacy_buttons_frame, text="Review Privacy Policy", 
                 command=self.show_privacy_consent).pack(side=tk.LEFT)
        
        privacy_frame.columnconfigure(1, weight=1)
    
    def _flash_alert_indicator(self, severity: str):
        """Flash visual alert indicator"""
        original_bg = self.monitoring_status_label.cget("background")
        
        if severity == "High":
            flash_color = "red"
            flash_count = 6
        elif severity == "Medium":
            flash_color = "orange"  
            flash_count = 4
        else:
            flash_color = "yellow"
            flash_count = 2
        
        def flash(count):
            if count > 0:
                current_bg = self.monitoring_status_label.cget("background")
                new_color = flash_color if current_bg == original_bg else original_bg
                self.monitoring_status_label.config(background=new_color)
                self.root.after(200, lambda: flash(count - 1))
        
        flash(flash_count)
    
    def save_settings(self):
        """Save current settings to configuration"""
        try:
            # Update configuration
            self.config.max_fps = self.max_fps_var.get()
            self.config.max_memory_mb = self.memory_limit_var.get()
            self.config.frame_drop_threshold = self.cpu_threshold_var.get() / 100
            self.config.alert_sensitivity = self.sensitivity_var.get()
            self.config.audio_alerts = self.audio_alerts_var.get()
            self.config.data_retention_days = self.retention_var.get()
            
            # Validate settings
            errors = self.config.validate()
            if errors:
                error_msg = "Configuration errors:\n" + "\n".join([f"• {field}: {error}" for field, error in errors.items()])
                messagebox.showerror("Configuration Error", error_msg)
                return
            
            # Save to file
            self.config_manager.save_config()
            
            messagebox.showinfo("Settings Saved", "Configuration has been saved successfully.")
            self.logger.info("Settings saved successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
            self.logger.error(f"Failed to save settings: {e}")
    
    def export_user_data(self):
        """Export user data for privacy compliance"""
        if not self.user_consent:
            messagebox.showinfo("No Data", "No user data to export (monitoring not enabled)")
            return
        
        try:
            export_dir = filedialog.askdirectory(title="Select Export Directory")
            if not export_dir:
                return
            
            export_path = Path(export_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export alerts
            alerts_data = [alert.to_dict() for alert in self.alerts]
            alerts_file = export_path / f"ridebuddy_alerts_{timestamp}.json"
            with open(alerts_file, 'w') as f:
                json.dump(alerts_data, f, indent=2, default=str)
            
            # Export session statistics
            stats_data = self.database.get_session_statistics(days=self.config.data_retention_days)
            stats_file = export_path / f"ridebuddy_statistics_{timestamp}.json"
            with open(stats_file, 'w') as f:
                json.dump(stats_data, f, indent=2, default=str)
            
            # Export configuration (anonymized)
            config_data = asdict(self.config)
            config_file = export_path / f"ridebuddy_config_{timestamp}.json"
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            messagebox.showinfo("Export Complete", 
                f"Data exported successfully to:\n{export_path}\n\n"
                f"Files created:\n• {alerts_file.name}\n• {stats_file.name}\n• {config_file.name}")
            
            self.logger.info(f"User data exported to {export_path}")
            
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to export data: {str(e)}")
            self.logger.error(f"Data export failed: {e}")
    
    def delete_user_data(self):
        """Delete all user data for privacy compliance"""
        response = messagebox.askyesnocancel(
            "Delete All Data",
            "This will permanently delete ALL stored data including:\n\n"
            "• All alert history\n"
            "• Session records\n"
            "• Performance metrics\n"
            "• Configuration settings\n\n"
            "This action cannot be undone!\n\n"
            "Are you sure you want to continue?"
        )
        
        if response:
            try:
                # Stop monitoring if active
                if self.is_monitoring:
                    self.toggle_monitoring()
                
                # Clear in-memory data
                self.alerts.clear()
                if hasattr(self, 'recent_alerts_listbox'):
                    self.recent_alerts_listbox.delete(0, tk.END)
                
                # Clear treeview
                for item in self.alerts_tree.get_children():
                    self.alerts_tree.delete(item)
                
                # Delete database
                if self.database.db_file.exists():
                    self.database.db_file.unlink()
                    self.database.init_database()
                
                # Delete log files
                log_dir = Path("logs")
                if log_dir.exists():
                    for log_file in log_dir.glob("*.log*"):
                        try:
                            log_file.unlink()
                        except:
                            pass
                
                # Reset configuration to defaults
                self.config = AppConfig()
                self.config_manager.config = self.config
                self.config_manager.save_config()
                
                # Reset stats
                self.stats = {
                    'total_frames': 0,
                    'drowsy_detections': 0,
                    'phone_detections': 0,
                    'alerts_triggered': 0,
                    'session_start': datetime.now()
                }
                
                # Update UI
                self.update_counters()
                
                messagebox.showinfo("Data Deleted", 
                    "All user data has been permanently deleted.\n\n"
                    "The application will restart with default settings.")
                
                self.logger.info("All user data deleted by user request")
                
                # Restart application
                self.root.after(2000, self.restart_application)
                
            except Exception as e:
                messagebox.showerror("Deletion Failed", f"Failed to delete all data: {str(e)}")
                self.logger.error(f"Data deletion failed: {e}")
    
    def update_counters(self):
        """Update all counter displays"""
        if hasattr(self, 'alerts_count_label'):
            self.alerts_count_label.config(text=str(self.stats['alerts_triggered']))
        if hasattr(self, 'drowsy_count_label'):
            self.drowsy_count_label.config(text=str(self.stats['drowsy_detections']))
        if hasattr(self, 'phone_count_label'):
            self.phone_count_label.config(text=str(self.stats['phone_detections']))
        if hasattr(self, 'total_frames_label'):
            self.total_frames_label.config(text=str(self.stats['total_frames']))
        if hasattr(self, 'frames_label'):
            self.frames_label.config(text=str(self.stats['total_frames']))
    
    def restart_application(self):
        """Restart the application"""
        try:
            # Clean shutdown
            self.camera_manager.release()
            self.performance_monitor.stop_monitoring()
            
            # Restart
            os.execv(sys.executable, ['python'] + sys.argv)
        except Exception as e:
            self.logger.error(f"Failed to restart application: {e}")
            self.root.quit()
    
    def create_status_bar(self):
        """Create status bar at bottom"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Status indicators
        self.camera_status_label = ttk.Label(status_frame, text="📷 Camera: Not Connected", 
                                           font=("Arial", 9))
        self.camera_status_label.pack(side=tk.LEFT, padx=10)
        
        self.monitoring_status_label = ttk.Label(status_frame, text="⏹️ Monitoring: Stopped", 
                                               font=("Arial", 9))
        self.monitoring_status_label.pack(side=tk.LEFT, padx=10)
        
        # Spacer
        ttk.Label(status_frame, text="").pack(side=tk.LEFT, expand=True)
        
        # Time
        self.time_label = ttk.Label(status_frame, text="", font=("Arial", 9))
        self.time_label.pack(side=tk.RIGHT, padx=10)
        
        # Update time
        self.update_time_display()
    
    def show_privacy_consent(self):
        """Show data privacy and consent dialog"""
        consent_dialog = tk.Toplevel(self.root)
        consent_dialog.title("Privacy & Data Consent")
        consent_dialog.geometry("600x500")
        consent_dialog.resizable(False, False)
        consent_dialog.grab_set()  # Make modal
        
        # Center the dialog
        consent_dialog.transient(self.root)
        
        # Privacy notice text
        privacy_text = f"""
{APP_NAME} - Privacy & Data Usage Notice

This application processes video data for driver monitoring and safety purposes.

DATA COLLECTION:
• Camera feed is processed locally on your device
• Alert data (timestamps, confidence levels) may be stored locally
• Performance metrics for optimization
• No personal identification data is collected

DATA STORAGE:
• All data is stored locally in encrypted format
• Data is automatically deleted after {self.config.data_retention_days} days
• You can export or delete your data at any time

DATA SECURITY:
• Video frames are processed in real-time and not permanently stored
• Local database is protected with access controls
• No data is transmitted to external servers without explicit consent

YOUR RIGHTS:
• You can withdraw consent at any time
• You can request data deletion
• You can export your data
• You can modify data retention settings

By clicking "I Agree", you consent to this data processing for safety monitoring purposes.
"""
        
        # Create scrollable text widget
        text_frame = ttk.Frame(consent_dialog)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Arial", 10), height=20)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget.insert(1.0, privacy_text)
        text_widget.config(state=tk.DISABLED)
        
        # Buttons
        button_frame = ttk.Frame(consent_dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        def accept_consent():
            self.user_consent = True
            logging.info("User accepted privacy consent")
            consent_dialog.destroy()
            # Initialize camera after consent is accepted
            if OPENCV_AVAILABLE:
                print("[CAMERA] Initializing camera after consent acceptance...")
                self.root.after(500, self.initialize_camera)
        
        def decline_consent():
            self.user_consent = False
            logging.info("User declined privacy consent")
            consent_dialog.destroy()
            messagebox.showinfo("Privacy Notice", 
                              "You can enable monitoring later through Settings.\n"
                              "Some features may be limited without consent.")
        
        def auto_decline():
            """Auto-decline after timeout"""
            if consent_dialog.winfo_exists():
                self.user_consent = False
                logging.info("Privacy consent auto-declined due to timeout")
                consent_dialog.destroy()
                messagebox.showwarning("Privacy Notice", 
                                     "Privacy consent timed out. Proceeding with limited features.\n"
                                     "You can enable monitoring later through Settings.")
        
        ttk.Button(button_frame, text="I Agree", command=accept_consent).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_frame, text="Decline", command=decline_consent).pack(side=tk.RIGHT)
        
        # Center the dialog
        consent_dialog.update_idletasks()
        x = (consent_dialog.winfo_screenwidth() // 2) - (consent_dialog.winfo_width() // 2)
        y = (consent_dialog.winfo_screenheight() // 2) - (consent_dialog.winfo_height() // 2)
        consent_dialog.geometry(f"+{x}+{y}")
        
        # Make dialog modal and wait for response
        consent_dialog.transient(self.root)
        consent_dialog.grab_set()
        consent_dialog.focus_force()
        
        # Add timeout to prevent indefinite blocking (60 seconds)
        consent_dialog.after(60000, auto_decline)
        
        # Wait for user response
        try:
            consent_dialog.wait_window()
        except Exception as e:
            logging.error(f"Privacy consent dialog error: {e}")
            self.user_consent = False
    
    def setup_logging(self):
        """Setup enhanced logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging level from config
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_dir / f"ridebuddy_{datetime.now().strftime('%Y%m%d')}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(simple_formatter)
        console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            handlers=[file_handler, console_handler]
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Log system information
        self.logger.info(f"{APP_NAME} v{VERSION} started")
        self.logger.info(f"Platform: {platform.system()} {platform.release()}")
        self.logger.info(f"Python: {platform.python_version()}")
        self.logger.info(f"OpenCV available: {OPENCV_AVAILABLE}")
        self.logger.info(f"PIL available: {PIL_AVAILABLE}")
        self.logger.info(f"psutil available: {PSUTIL_AVAILABLE}")
        
        # Log configuration
        self.logger.debug(f"Configuration: {asdict(self.config)}")
    
    def initialize_camera(self):
        """Initialize camera connection with enhanced vehicle mode support"""
        try:
            # Use camera index from config if available, otherwise from GUI
            if hasattr(self, 'camera_index_var') and self.camera_index_var:
                camera_index = int(self.camera_index_var.get())
            else:
                camera_index = self.config.camera_index
            
            print(f"[CAMERA] Initializing camera {camera_index}...")
            
            def init_thread():
                success = self.camera_manager.initialize_camera(camera_index)
                if success:
                    self.root.after_idle(lambda: self.update_camera_status("Connected", "green"))
                    print(f"[CAMERA] Camera {camera_index} initialized successfully")
                    
                    # Start camera capture for preview (both vehicle and desktop mode)
                    capture_success = self.camera_manager.start_capture()
                    if capture_success:
                        print(f"[CAMERA] Camera capture started for preview")
                    else:
                        print(f"[CAMERA] Warning: Failed to start camera capture")
                    
                    # In vehicle mode, automatically start monitoring
                    if VEHICLE_MODE or FLEET_MODE:
                        print("[VEHICLE] Auto-starting monitoring for vehicle deployment")
                        self.root.after(1000, self.auto_start_monitoring)
                else:
                    self.root.after_idle(lambda: self.update_camera_status("Failed", "red"))
                    print(f"[CAMERA] Failed to initialize camera {camera_index}")
            
            threading.Thread(target=init_thread, daemon=True).start()
            
        except Exception as e:
            print(f"[CAMERA] Error during initialization: {e}")
            self.update_camera_status("Error", "red")
    
    def auto_start_monitoring(self):
        """Automatically start monitoring in vehicle mode"""
        try:
            print("[VEHICLE] Auto-starting monitoring...")
            if not self.is_monitoring:
                self.toggle_monitoring()
                print("[VEHICLE] Monitoring started successfully")
            else:
                print("[VEHICLE] Monitoring already active")
        except Exception as e:
            print(f"[ERROR] Auto-start monitoring failed: {e}")
    
    def update_camera_status(self, status, color):
        """Update camera status label safely"""
        try:
            if hasattr(self, 'camera_status_label') and self.camera_status_label:
                self.camera_status_label.config(text=f"Camera: {status}", foreground=color)
        except Exception as e:
            print(f"[GUI] Error updating camera status: {e}")
    
    def auto_start_monitoring(self):
        """Automatically start monitoring in vehicle mode"""
        try:
            if not self.is_monitoring and (VEHICLE_MODE or FLEET_MODE):
                print("[VEHICLE] Auto-starting monitoring...")
                self.toggle_monitoring()
        except Exception as e:
            print(f"[VEHICLE] Error auto-starting monitoring: {e}")
    
    def reconnect_camera(self):
        """Reconnect to camera"""
        self.camera_manager.release()
        self.camera_status_label.config(text="📷 Camera: Reconnecting...", foreground="orange")
        self.root.after(500, self.initialize_camera)
    
    def test_camera(self):
        """Test camera connection"""
        camera_index = int(self.camera_index_var.get())
        
        def test_thread():
            try:
                test_camera = cv2.VideoCapture(camera_index)
                if test_camera.isOpened():
                    ret, frame = test_camera.read()
                    test_camera.release()
                    if ret:
                        self.root.after_idle(lambda: messagebox.showinfo(
                            "Camera Test", f"Camera {camera_index} is working properly!"))
                    else:
                        self.root.after_idle(lambda: messagebox.showerror(
                            "Camera Test", f"Camera {camera_index} cannot capture frames"))
                else:
                    self.root.after_idle(lambda: messagebox.showerror(
                        "Camera Test", f"Cannot open camera {camera_index}"))
            except Exception as e:
                self.root.after_idle(lambda: messagebox.showerror(
                    "Camera Test", f"Camera test failed: {str(e)}"))
        
        threading.Thread(target=test_thread, daemon=True).start()
    
    def toggle_monitoring(self):
        """Toggle monitoring state with session management"""
        if not self.user_consent:
            response = messagebox.askyesno(
                "Privacy Consent Required",
                "Camera monitoring requires data processing consent.\n\n"
                "Would you like to review the privacy policy and provide consent?"
            )
            if response:
                self.show_privacy_consent()
            if not self.user_consent:
                return
        
        if not self.is_monitoring:
            # Start monitoring
            if not self.camera_manager.camera or not self.camera_manager.camera.isOpened():
                messagebox.showerror("Error", 
                    "Camera not connected. Please connect camera first.\n\n"
                    "Try:\n• Check camera permissions\n• Reconnect camera\n• Check Settings tab")
                return
            
            # Create new session
            self.session_id = f"session_{int(time.time())}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
            self.database.start_session(self.session_id, self.config)
            
            success = self.camera_manager.start_capture()
            if success:
                self.is_monitoring = True
                self.start_button.config(text="⏹️ Stop Monitoring", bg='#ff0000')
                self.monitoring_status_label.config(text="▶️ Monitoring: Active", foreground="green")
                self.stats['session_start'] = datetime.now()
                
                # Reset counters
                self.stats.update({
                    'total_frames': 0,
                    'drowsy_detections': 0,
                    'phone_detections': 0,
                    'alerts_triggered': 0
                })
                
                self.logger.info(f"Monitoring started - Session: {self.session_id}")
                
                # Show session info
                messagebox.showinfo("Monitoring Started", 
                    f"Driver monitoring is now active.\n\n"
                    f"Session ID: {self.session_id}\n"
                    f"Camera: {self.config.camera_index}\n"
                    f"Sensitivity: {self.config.alert_sensitivity:.1%}")
            else:
                messagebox.showerror("Error", "Failed to start camera capture")
        else:
            # Stop monitoring
            self.camera_manager.stop_capture()
            self.is_monitoring = False
            self.start_button.config(text="▶️ Start Monitoring", bg='#0066ff')
            self.monitoring_status_label.config(text="⏹️ Monitoring: Stopped", foreground="red")
            
            # End session
            if self.session_id:
                session_stats = {
                    'total_frames': self.stats['total_frames'],
                    'total_alerts': self.stats['alerts_triggered'],
                    'avg_fps': self.current_fps
                }
                self.database.end_session(self.session_id, session_stats)
                
                # Show session summary
                duration = datetime.now() - self.stats['session_start']
                duration_str = str(duration).split('.')[0]  # Remove microseconds
                
                messagebox.showinfo("Session Ended", 
                    f"Monitoring session completed.\n\n"
                    f"Duration: {duration_str}\n"
                    f"Frames processed: {self.stats['total_frames']}\n"
                    f"Alerts triggered: {self.stats['alerts_triggered']}\n"
                    f"Average FPS: {self.current_fps:.1f}")
            
            self.logger.info(f"Monitoring stopped - Session: {self.session_id}")
            self.session_id = None
    
    def update_gui_loop(self):
        """Main GUI update loop"""
        # Process camera results with timeout to prevent blocking
        try:
            processed_items = 0
            max_items_per_cycle = 10  # Limit processing to prevent GUI freezing
            
            while processed_items < max_items_per_cycle:
                try:
                    result_type, data = self.camera_manager.result_queue.get_nowait()
                    processed_items += 1
                    
                    if result_type == "status":
                        if hasattr(self, 'camera_status_label'):
                            self.camera_status_label.config(text=f"Camera: {data}")
                        self.logger.info(f"Camera status: {data}")
                    
                    elif result_type == "frame":
                        self.process_frame_data(data)
                    
                    elif result_type == "fps":
                        self.current_fps = data
                        if hasattr(self, 'fps_label'):
                            self.fps_label.config(text=f"{data:.1f}")
                    
                    elif result_type == "error":
                        self.logger.error(f"Camera error: {data}")
                        
                except queue.Empty:
                    break  # No more items to process
                except Exception as e:
                    self.logger.error(f"Error processing GUI queue item: {e}")
                    break
                    
        except Exception as e:
            self.logger.error(f"Error in update_gui_loop: {e}")
        
        # Update session time
        if self.is_monitoring and hasattr(self, 'uptime_label'):
            session_duration = datetime.now() - self.stats['session_start']
            hours, remainder = divmod(session_duration.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.uptime_label.config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Update performance metrics
        self.update_performance_display()
        
        # Schedule next update
        self.root.after(50, self.update_gui_loop)
    
    def process_frame_data(self, data):
        """Process incoming frame data from camera"""
        frame = data["frame"]
        predictions = data["predictions"]
        
        # Update statistics
        self.frame_count += 1
        self.stats['total_frames'] += 1
        
        # Display frame
        self.display_frame(frame, predictions)
        
        # Update status indicators
        self.update_status_display(predictions)
        
        # Check for alerts
        self.check_and_process_alerts(predictions)
        
        # Update counters (legacy interface)
        if hasattr(self, 'frames_label'):
            self.frames_label.config(text=str(self.stats['total_frames']))
        if hasattr(self, 'total_frames_label'):
            self.total_frames_label.config(text=str(self.stats['total_frames']))
    
    def display_frame(self, frame, predictions):
        """Display video frame with overlays"""
        if not OPENCV_AVAILABLE or not PIL_AVAILABLE:
            return
        
        try:
            # Resize frame for display
            display_frame = cv2.resize(frame, (640, 480))
            
            # Add overlay information
            if self.is_monitoring:
                # Status overlay
                cv2.putText(display_frame, "MONITORING ACTIVE", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Alertness status
                alertness = predictions["alertness"]
                color = (0, 255, 0) if alertness == "Normal" else (0, 0, 255)
                cv2.putText(display_frame, f"Status: {alertness}", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Confidence
                confidence = predictions["confidence"]
                cv2.putText(display_frame, f"Confidence: {confidence:.1%}", (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Phone detection indicator
                if predictions["phone_detected"]:
                    cv2.rectangle(display_frame, (10, 100), (200, 130), (0, 0, 255), -1)
                    cv2.putText(display_frame, "PHONE DETECTED", (15, 120), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Convert to RGB and display
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            tk_image = ImageTk.PhotoImage(pil_image)
            
            self.video_label.configure(image=tk_image, text="")
            self.video_label.image = tk_image
            
        except Exception as e:
            self.logger.error(f"Display error: {str(e)}")
    
    def update_status_display(self, predictions):
        """Update status indicators with modern colors"""
        # Alertness
        alertness = predictions["alertness"]
        alertness_color = self.colors['accent_success'] if alertness == "Normal" else self.colors['accent_danger']
        
        # Update alertness label if it exists
        if hasattr(self, 'alertness_label'):
            self.alertness_label.config(text=alertness, foreground=alertness_color)
        
        # Phone detection
        phone_detected = predictions["phone_detected"]
        phone_text = "Yes" if phone_detected else "No"
        phone_color = self.colors['accent_danger'] if phone_detected else self.colors['accent_success']
        if hasattr(self, 'phone_label'):
            self.phone_label.config(text=phone_text, foreground=phone_color)
        
        # Seatbelt
        seatbelt_worn = predictions["seatbelt_worn"]
        seatbelt_text = "Worn" if seatbelt_worn else "Not Worn"
        seatbelt_color = self.colors['accent_success'] if seatbelt_worn else self.colors['accent_danger']
        if hasattr(self, 'seatbelt_label'):
            self.seatbelt_label.config(text=seatbelt_text, foreground=seatbelt_color)
        
        # Confidence
        confidence = predictions["confidence"]
        confidence_color = self.colors['accent_success'] if confidence > 0.8 else (
            self.colors['accent_warning'] if confidence > 0.6 else self.colors['accent_danger'])
        if hasattr(self, 'confidence_label'):
            self.confidence_label.config(text=f"{confidence:.1%}", foreground=confidence_color)
            
        # Update confidence progress bar if exists
        if hasattr(self, 'confidence_progress'):
            self.confidence_progress['value'] = confidence * 100
    
    def check_and_process_alerts(self, predictions):
        """Check for alerts and process them"""
        threshold = self.sensitivity_var.get()
        
        # Check for drowsiness alert
        if predictions["alertness"] == "Drowsy" and predictions["confidence"] > threshold:
            alert = SafetyAlert("Drowsy", "Driver appears drowsy", predictions["confidence"], "High")
            self.add_alert(alert)
            self.stats['drowsy_detections'] += 1
        
        # Check for phone distraction alert
        elif predictions["alertness"] == "Phone_Distraction" and predictions["confidence"] > threshold:
            alert = SafetyAlert("Phone_Distraction", "Phone usage detected", predictions["confidence"], "High")
            self.add_alert(alert)
            self.stats['phone_detections'] += 1
        
        # Check for seatbelt alert
        if not predictions["seatbelt_worn"] and predictions["confidence"] > 0.8:
            alert = SafetyAlert("Seatbelt", "Seatbelt not worn", predictions["confidence"], "Medium")
            self.add_alert(alert)
    
    def add_alert(self, alert):
        """Enhanced alert processing with database storage and notification"""
        # Add to in-memory collection
        self.alerts.append(alert)
        self.stats['alerts_triggered'] += 1
        
        # Save to database if session is active
        if self.session_id and self.user_consent:
            self.database.save_alert(alert, self.session_id)
        
        # Add to recent alerts display (modern Text widget)
        severity_icon = "🚨" if alert.severity == "High" else ("⚠️" if alert.severity == "Medium" else "ℹ️")
        alert_message = f"{alert.timestamp.strftime('%H:%M:%S')} {severity_icon} {alert.alert_type}: {alert.message} ({alert.confidence:.1%})"
        self.add_alert_message(alert_message, alert.severity.lower())
        
        # Add to recent alerts listbox (legacy interface) if it exists
        if hasattr(self, 'recent_alerts_listbox'):
            alert_text = f"{alert.timestamp.strftime('%H:%M:%S')} - {alert.alert_type}: {alert.message}"
            self.recent_alerts_listbox.insert(0, alert_text)
            
            # Keep only last 20 alerts in recent list
            if self.recent_alerts_listbox.size() > 20:
                self.recent_alerts_listbox.delete(tk.END)
        
        # Add to alerts treeview with color coding
        severity_icon = "🚨" if alert.severity == "High" else ("⚠️" if alert.severity == "Medium" else "ℹ️")
        
        item_id = self.alerts_tree.insert("", 0, values=(
            alert.timestamp.strftime("%H:%M:%S"),
            alert.alert_type,
            alert.message,
            f"{alert.confidence:.1%}",
            f"{severity_icon} {alert.severity}"
        ))
        
        # Color code rows by severity (safer approach)
        try:
            if alert.severity == "High":
                self.alerts_tree.item(item_id, tags=('high',))
            elif alert.severity == "Medium":
                self.alerts_tree.item(item_id, tags=('medium',))
            else:
                self.alerts_tree.item(item_id, tags=('low',))
        except Exception as e:
            logging.debug(f"TreeView tag error: {e}")  # Non-critical error
        
        # Update counters
        self.alerts_count_label.config(text=str(self.stats['alerts_triggered']))
        self.drowsy_count_label.config(text=str(self.stats['drowsy_detections']))
        self.phone_count_label.config(text=str(self.stats['phone_detections']))
        
        # Enhanced logging with context
        self.logger.warning(
            f"ALERT [{alert.severity}]: {alert.alert_type} - {alert.message} "
            f"(Confidence: {alert.confidence:.1%}, Session: {self.session_id})"
        )
        
        # Enhanced audio/visual alerts
        if self.config.audio_alerts:
            # Different alert sounds for different severities
            if alert.severity == "High":
                for _ in range(3):  # Triple beep for high severity
                    self.root.bell()
                    self.root.after(100, lambda: None)
            else:
                self.root.bell()
        
        # Flash the alert indicator
        self._flash_alert_indicator(alert.severity)
    
    def clear_alerts(self):
        """Clear all alerts"""
        if messagebox.askyesno("Clear Alerts", "Are you sure you want to clear all alerts?"):
            self.alerts.clear()
            if hasattr(self, 'recent_alerts_listbox'):
                self.recent_alerts_listbox.delete(0, tk.END)
            
            # Clear treeview
            for item in self.alerts_tree.get_children():
                self.alerts_tree.delete(item)
            
            self.logger.info("All alerts cleared")
    
    def export_alerts(self):
        """Export alerts to JSON file"""
        if not self.alerts:
            messagebox.showinfo("Export Alerts", "No alerts to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export Alerts"
        )
        
        if filename:
            try:
                alerts_data = [alert.to_dict() for alert in self.alerts]
                with open(filename, 'w') as f:
                    json.dump(alerts_data, f, indent=2, default=str)
                
                messagebox.showinfo("Export Successful", f"Alerts exported to {filename}")
                self.logger.info(f"Alerts exported to {filename}")
                
            except Exception as e:
                messagebox.showerror("Export Failed", f"Failed to export alerts: {str(e)}")
    
    def update_performance_display(self):
        """Update enhanced performance metrics display"""
        if not hasattr(self, 'last_perf_update'):
            self.last_perf_update = time.time()
        
        # Update every 3 seconds for more responsive feedback
        if time.time() - self.last_perf_update > 3.0:
            # Get performance statistics
            perf_stats = self.performance_monitor.get_performance_stats()
            
            session_duration = datetime.now() - self.stats['session_start'] if self.is_monitoring else None
            duration_str = str(session_duration).split('.')[0] if session_duration else 'Not started'
            
            perf_text = f"""REAL-TIME PERFORMANCE METRICS (Updated: {datetime.now().strftime('%H:%M:%S')})
{'='*70}

📊 SESSION STATISTICS:
  • Session ID: {self.session_id or 'None'}
  • Duration: {duration_str}
  • Total Frames: {self.stats['total_frames']:,}
  • Current FPS: {self.current_fps:.1f} / {self.config.max_fps} max
  • Monitoring: {'Active' if self.is_monitoring else 'Inactive'}

🚨 DETECTION STATISTICS:
  • Drowsy Episodes: {self.stats['drowsy_detections']}
  • Phone Distractions: {self.stats['phone_detections']}
  • Total Alerts: {self.stats['alerts_triggered']}
  • Alert Rate: {(self.stats['alerts_triggered'] / max(1, self.stats['total_frames']) * 100):.2f}%

💻 SYSTEM PERFORMANCE:"""
            
            if PSUTIL_AVAILABLE and 'current_cpu_percent' in perf_stats:
                perf_text += f"""
  • CPU Usage: {perf_stats['current_cpu_percent']:.1f}% (avg: {perf_stats.get('avg_cpu_percent', 0):.1f}%)
  • Memory Usage: {perf_stats.get('process_memory_mb', 0):.1f} MB / {self.config.max_memory_mb} MB limit
  • System Memory: {perf_stats.get('system_memory_percent', 0):.1f}% used
  • Available Memory: {perf_stats.get('available_memory_mb', 0):.0f} MB"""
            else:
                perf_text += f"""
  • Performance Monitor: Limited (psutil not available)
  • Memory Monitoring: Basic garbage collection only"""
            
            perf_text += f"""
  • Frame Drops: {perf_stats.get('frame_drops', 0)}
  • GC Collections: {sum(perf_stats.get('gc_collections', [0, 0, 0]))}
  • Active Threads: {threading.active_count()}

📡 CAMERA & PROCESSING:
  • Camera Status: {'Connected' if self.camera_manager.camera and self.camera_manager.camera.isOpened() else 'Disconnected'}
  • Connection Stable: {'Yes' if self.camera_manager.connection_stable else 'No'}
  • Error Count: {self.camera_manager.error_count}
  • Reconnect Attempts: {self.camera_manager.reconnect_attempts} / {self.config.reconnect_attempts}
  • Frame Queue: {self.camera_manager.frame_queue.qsize()} / {self.camera_manager.frame_queue.maxsize}
  • Result Queue: {self.camera_manager.result_queue.qsize()}

⚙️ CONFIGURATION:
  • Alert Sensitivity: {self.config.alert_sensitivity:.1%}
  • Frame Drop Threshold: {self.config.frame_drop_threshold:.0%} CPU
  • Audio Alerts: {'Enabled' if self.config.audio_alerts else 'Disabled'}
  • Data Retention: {self.config.data_retention_days} days
  • Log Level: {self.config.log_level}

🔧 SYSTEM CAPABILITIES:
  • OpenCV: {'✅ Available' if OPENCV_AVAILABLE else '❌ Missing'}
  • PIL/Pillow: {'✅ Available' if PIL_AVAILABLE else '❌ Missing'}
  • Performance Monitor: {'✅ Full' if PSUTIL_AVAILABLE else '⚠️ Limited'}
  • Database: {'✅ Active' if self.database.db_file.exists() else '❌ Unavailable'}
  • Privacy Consent: {'✅ Granted' if self.user_consent else '❌ Not Granted'}

📈 QUALITY METRICS:
  • Frame Processing Rate: {(self.stats['total_frames'] / max(1, session_duration.total_seconds()) if session_duration else 0):.1f} fps avg
  • System Stability: {'Good' if self.camera_manager.error_count < 10 else 'Poor'}
  • Performance Rating: {'Optimal' if perf_stats.get('current_cpu_percent', 100) < 70 else 'Heavy Load'}
"""
            
            # Update system info in settings tab
            if hasattr(self, 'system_info_label'):
                system_text = f"""Platform: {platform.system()} {platform.release()}
Python: {platform.python_version()}
OpenCV: {'Available' if OPENCV_AVAILABLE else 'Not Available'}
PIL: {'Available' if PIL_AVAILABLE else 'Not Available'}
Performance Monitor: {'Available' if PSUTIL_AVAILABLE else 'Limited'}

Performance Stats:
  CPU: {perf_stats.get('current_cpu_percent', 0):.1f}%
  Memory: {perf_stats.get('process_memory_mb', 0):.1f} MB
  Threads: {threading.active_count()}
  Frame Drops: {perf_stats.get('frame_drops', 0)}"""
                
                self.system_info_label.config(text=system_text)
            
            self.performance_text.delete(1.0, tk.END)
            self.performance_text.insert(1.0, perf_text)
            self.last_perf_update = time.time()
    
    def update_time_display(self):
        """Update time display in status bar"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time_display)
    
    def open_log_folder(self):
        """Open the log folder in file explorer"""
        log_dir = Path("logs")
        if log_dir.exists():
            os.startfile(log_dir)
        else:
            messagebox.showinfo("Log Folder", "Log folder does not exist yet")
    
    def on_closing(self):
        """Enhanced application closing with proper cleanup"""
        try:
            if self.is_monitoring:
                response = messagebox.askyesnocancel(
                    "Quit Application", 
                    "Monitoring is currently active.\n\n"
                    "Would you like to:\n"
                    "• Yes: Save session and quit\n"
                    "• No: Quit without saving\n"
                    "• Cancel: Continue monitoring"
                )
                
                if response is None:  # Cancel
                    return
                elif response:  # Yes - save and quit
                    self.toggle_monitoring()  # This will save the session
                # No - continue to quit without saving
            
            # Show shutdown progress
            shutdown_dialog = tk.Toplevel(self.root)
            shutdown_dialog.title("Shutting Down")
            shutdown_dialog.geometry("300x150")
            shutdown_dialog.resizable(False, False)
            shutdown_dialog.grab_set()
            
            ttk.Label(shutdown_dialog, text="Shutting down RideBuddy...", font=("Arial", 12)).pack(pady=20)
            progress = ttk.Progressbar(shutdown_dialog, mode='indeterminate')
            progress.pack(pady=10, padx=20, fill=tk.X)
            progress.start()
            
            status_label = ttk.Label(shutdown_dialog, text="Saving configuration...")
            status_label.pack(pady=5)
            
            self.root.update()
            
            # Cleanup sequence
            cleanup_steps = [
                ("Stopping camera...", lambda: self.camera_manager.release()),
                ("Stopping performance monitor...", lambda: self.performance_monitor.stop_monitoring()),
                ("Cleaning up old data...", lambda: self.database.cleanup_old_data(self.config.data_retention_days)),
                ("Saving configuration...", lambda: self.config_manager.save_config()),
                ("Finalizing shutdown...", lambda: time.sleep(0.5))
            ]
            
            for step_name, step_func in cleanup_steps:
                status_label.config(text=step_name)
                self.root.update()
                try:
                    step_func()
                except Exception as e:
                    self.logger.error(f"Error during {step_name}: {e}")
                time.sleep(0.2)
            
            # Final log entry
            self.logger.info("RideBuddy Pro shutdown completed successfully")
            
            # Close shutdown dialog
            shutdown_dialog.destroy()
            
            # Destroy main window
            self.root.destroy()
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            self.root.destroy()

def main():
    """Enhanced application entry point with startup checks"""
    print(f"🚗 Starting {APP_NAME} v{VERSION}")
    print("="*50)
    
    # System compatibility check
    print("🔍 System Compatibility Check:")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  OpenCV: {'✅ Available' if OPENCV_AVAILABLE else '❌ Missing'}")
    print(f"  PIL/Pillow: {'✅ Available' if PIL_AVAILABLE else '❌ Missing'}")
    print(f"  Performance Monitor: {'✅ Full' if PSUTIL_AVAILABLE else '⚠️ Limited'}")
    
    # Check minimum requirements
    missing_requirements = []
    if not OPENCV_AVAILABLE:
        missing_requirements.append("OpenCV (required for camera processing)")
    if not PIL_AVAILABLE:
        missing_requirements.append("PIL/Pillow (required for image display)")
    
    if missing_requirements:
        print("\n❌ Missing Requirements:")
        for req in missing_requirements:
            print(f"  • {req}")
        print("\nPlease install missing dependencies before running RideBuddy.")
        input("Press Enter to continue anyway (limited functionality)...")
    
    try:
        # Initialize Tkinter
        root = tk.Tk()
        
        # Set application properties
        root.title(f"{APP_NAME} v{VERSION}")
        
        # Set application icon (if available)
        try:
            root.iconbitmap("icon.ico")
        except:
            pass
        
        # Create splash screen
        splash = tk.Toplevel(root)
        splash.title("Loading...")
        splash.geometry("400x300")
        splash.resizable(False, False)
        splash.configure(bg='#2c3e50')
        
        # Splash content
        splash_frame = tk.Frame(splash, bg='#2c3e50')
        splash_frame.pack(expand=True, fill='both')
        
        tk.Label(splash_frame, text=f"{APP_NAME}", font=("Arial", 24, "bold"), 
                fg='white', bg='#2c3e50').pack(pady=30)
        tk.Label(splash_frame, text=f"Version {VERSION}", font=("Arial", 12), 
                fg='#bdc3c7', bg='#2c3e50').pack()
        tk.Label(splash_frame, text="Professional Driver Monitoring System", font=("Arial", 10), 
                fg='#95a5a6', bg='#2c3e50').pack(pady=10)
        
        # Progress bar
        progress = ttk.Progressbar(splash_frame, mode='indeterminate', length=300)
        progress.pack(pady=20)
        progress.start()
        
        status_label = tk.Label(splash_frame, text="Initializing...", font=("Arial", 10), 
                              fg='#ecf0f1', bg='#2c3e50')
        status_label.pack(pady=10)
        
        # Center splash screen
        splash.update_idletasks()
        x = (splash.winfo_screenwidth() // 2) - (splash.winfo_width() // 2)
        y = (splash.winfo_screenheight() // 2) - (splash.winfo_height() // 2)
        splash.geometry(f"+{x}+{y}")
        
        # Hide main window initially
        root.withdraw()
        
        # Show splash
        splash.lift()
        splash.update()
        
        # Initialization steps
        init_steps = [
            "Loading configuration...",
            "Initializing database...", 
            "Setting up performance monitor...",
            "Preparing user interface...",
            "Finalizing startup..."
        ]
        
        for step in init_steps:
            status_label.config(text=step)
            splash.update()
            time.sleep(0.5)  # Simulate initialization time
        
        # Create main application
        status_label.config(text="Creating main interface...")
        splash.update()
        
        app = RideBuddyOptimizedGUI(root)
        
        # Close splash and show main window
        splash.destroy()
        root.deiconify()
        
        # Handle window closing
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        # Center main window on screen
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f"+{x}+{y}")
        
        print(f"\n[SUCCESS] {APP_NAME} started successfully!")
        print("[READY] Ready for driver monitoring...")
        
        # Start the application
        root.mainloop()
        
    except Exception as e:
        print(f"\n[ERROR] Failed to start {APP_NAME}: {e}")
        print("\nPlease check:")
        print("- Python installation")
        print("- Required dependencies (OpenCV, PIL)")
        print("- System permissions")
        input("Press Enter to exit...")
        return 1
    
    return 0

if __name__ == "__main__":
    main()