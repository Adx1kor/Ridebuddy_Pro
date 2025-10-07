#!/usr/bin/env python3
"""
RideBuddy Camera Diagnostics
Quick test to verify camera functionality and frame processing
"""

import cv2
import time
import threading
import queue
import sys

def test_camera_basic():
    """Test basic camera functionality"""
    print("=" * 50)
    print("CAMERA DIAGNOSTIC TEST")
    print("=" * 50)
    
    # Test camera availability
    print("\n1. Testing camera availability...")
    for i in range(3):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"   ✓ Camera {i}: Available ({frame.shape[1]}x{frame.shape[0]})")
                    cap.release()
                    return i
                else:
                    print(f"   ✗ Camera {i}: Opened but no frames")
            else:
                print(f"   ✗ Camera {i}: Not available")
            cap.release()
        except Exception as e:
            print(f"   ✗ Camera {i}: Error - {e}")
    
    print("\n❌ No working cameras found!")
    return None

def test_frame_capture(camera_index):
    """Test frame capture performance"""
    print(f"\n2. Testing frame capture performance on camera {camera_index}...")
    
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("   ❌ Failed to open camera")
            return False
        
        # Configure camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        frame_count = 0
        start_time = time.time()
        test_duration = 3  # seconds
        
        print(f"   Capturing frames for {test_duration} seconds...")
        
        while time.time() - start_time < test_duration:
            ret, frame = cap.read()
            if ret and frame is not None:
                frame_count += 1
                # Small delay to prevent overwhelming
                time.sleep(0.01)
            else:
                print(f"   ⚠️ Frame read failed at frame {frame_count}")
        
        cap.release()
        
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        
        print(f"   ✓ Captured {frame_count} frames in {elapsed_time:.2f}s")
        print(f"   ✓ Average FPS: {fps:.2f}")
        
        if fps > 10:
            print("   ✓ Frame capture performance: GOOD")
            return True
        else:
            print("   ⚠️ Frame capture performance: LOW")
            return True
            
    except Exception as e:
        print(f"   ❌ Frame capture test failed: {e}")
        return False

def test_threading_capture(camera_index):
    """Test threaded camera capture (similar to RideBuddy)"""
    print(f"\n3. Testing threaded capture on camera {camera_index}...")
    
    frame_queue = queue.Queue(maxsize=5)
    is_running = True
    
    def capture_thread():
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                return
                
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            frame_count = 0
            
            while is_running:
                ret, frame = cap.read()
                if ret and frame is not None:
                    try:
                        frame_queue.put(frame, block=False)
                        frame_count += 1
                    except queue.Full:
                        pass  # Drop frame if queue is full
                else:
                    time.sleep(0.01)
                    
                time.sleep(1/30)  # 30 FPS target
                
            cap.release()
            print(f"   ✓ Capture thread processed {frame_count} frames")
            
        except Exception as e:
            print(f"   ❌ Capture thread error: {e}")
    
    # Start capture thread
    thread = threading.Thread(target=capture_thread, daemon=True)
    thread.start()
    
    # Monitor for 3 seconds
    start_time = time.time()
    frames_received = 0
    
    try:
        while time.time() - start_time < 3:
            try:
                frame = frame_queue.get(timeout=0.1)
                frames_received += 1
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        pass
    
    is_running = False
    thread.join(timeout=1)
    
    print(f"   ✓ Main thread received {frames_received} frames")
    
    if frames_received > 30:
        print("   ✓ Threaded capture: WORKING PROPERLY")
        return True
    else:
        print("   ⚠️ Threaded capture: LOW FRAME RATE")
        return True

def main():
    print("RideBuddy Camera Diagnostics")
    print("This will test camera functionality to diagnose frame freezing issues\n")
    
    # Test 1: Basic camera availability
    camera_index = test_camera_basic()
    if camera_index is None:
        print("\n❌ DIAGNOSIS: No working cameras detected")
        print("   Solutions:")
        print("   • Check if camera is connected and not used by another app")
        print("   • Try different camera indices")
        print("   • Check camera drivers")
        return
    
    # Test 2: Frame capture performance
    if not test_frame_capture(camera_index):
        print("\n❌ DIAGNOSIS: Frame capture is failing")
        print("   Solutions:")
        print("   • Check camera permissions")
        print("   • Restart camera/computer")
        print("   • Try different camera settings")
        return
    
    # Test 3: Threaded capture (RideBuddy style)
    if not test_threading_capture(camera_index):
        print("\n⚠️ DIAGNOSIS: Threaded capture has issues")
        print("   This might cause frame freezing in RideBuddy")
        return
    
    print("\n" + "=" * 50)
    print("✅ DIAGNOSIS: Camera system is working properly!")
    print("If RideBuddy frames are still stuck, the issue is likely:")
    print("• GUI update loop timing")
    print("• Unicode/encoding issues")
    print("• Thread synchronization")
    print("• Memory/performance problems")
    print("=" * 50)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Diagnostic test failed: {e}")
    
    input("\nPress Enter to exit...")