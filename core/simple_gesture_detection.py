import cv2
import numpy as np
import time
from typing import Dict, List, Tuple

class SimpleGestureDetector:
    """
    Simplified gesture detector that uses basic computer vision techniques
    instead of MediaPipe to avoid dependency conflicts.
    """
    
    def __init__(self):
        self.gesture_history = []
        self.last_detection_time = 0
        self.detection_threshold = 0.5  # Lowered threshold for better sensitivity
        
        # Initialize background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50
        )
        
        # Use face cascade for hand detection approximation (more reliable)
        try:
            self.hand_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except:
            self.hand_cascade = None
            
    def detect_sos_gestures(self, frame: np.ndarray) -> Dict:
        """
        Detect SOS gestures using simplified computer vision techniques
        """
        height, width = frame.shape[:2]
        current_time = time.time()
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Enhanced motion and activity detection
        motion_detected = self._detect_motion(frame)
        rapid_movement = self._detect_rapid_movement(gray)
        hand_regions = self._detect_hand_regions(gray)
        upper_body_activity = self._detect_upper_body_activity(frame)
        waving_motion = self._detect_waving_motion(frame)
        hands_up_pose = self._detect_hands_up_pose(frame)
        
        # Gesture analysis results
        results = {
            'sos_detected': False,
            'gesture_type': 'none',
            'confidence': 0.0,
            'details': {
                'motion_detected': motion_detected,
                'rapid_movement': rapid_movement,
                'hand_regions_count': len(hand_regions),
                'upper_body_activity': upper_body_activity,
                'waving_motion': waving_motion,
                'hands_up_pose': hands_up_pose
            }
        }
        
        # Enhanced heuristic-based gesture detection
        confidence = 0.0
        gesture_type = 'none'
        
        # Help wave detection (improved with actual waving motion)
        if waving_motion > 0.3 and upper_body_activity > 0.4:
            confidence += 0.6
            gesture_type = 'help_wave'
            
        # Hands up detection (vertical hand positions)
        if hands_up_pose > 0.5 and len(hand_regions) >= 1:
            confidence += 0.7
            gesture_type = 'hands_up'
            
        # Distress signal (rapid movements with hand detection)
        if rapid_movement and motion_detected and len(hand_regions) >= 1:
            confidence += 0.5
            gesture_type = 'distress_signal'
            
        # Panic movement (high overall activity with motion)
        if motion_detected and upper_body_activity > 0.6 and rapid_movement:
            confidence += 0.6
            gesture_type = 'panic_movement'
        
        # Emergency waving (combination of multiple indicators)
        if (waving_motion > 0.2 and rapid_movement and 
            upper_body_activity > 0.3 and len(hand_regions) >= 1):
            confidence += 0.8
            gesture_type = 'emergency_wave'
        
        # Update results
        if confidence > self.detection_threshold:
            results['sos_detected'] = True
            results['gesture_type'] = gesture_type
            results['confidence'] = min(confidence, 1.0)
            
            # Store in history
            self.gesture_history.append({
                'timestamp': current_time,
                'gesture': gesture_type,
                'confidence': confidence
            })
            
            # Keep history manageable
            if len(self.gesture_history) > 50:
                self.gesture_history = self.gesture_history[-50:]
                
        return results
    
    def _detect_motion(self, frame: np.ndarray) -> bool:
        """Detect motion using background subtraction"""
        try:
            fg_mask = self.bg_subtractor.apply(frame)
            motion_pixels = cv2.countNonZero(fg_mask)
            frame_pixels = frame.shape[0] * frame.shape[1]
            motion_ratio = motion_pixels / frame_pixels
            return motion_ratio > 0.02  # 2% threshold
        except:
            return False
    
    def _detect_rapid_movement(self, gray: np.ndarray) -> bool:
        """Detect rapid movement using optical flow"""
        try:
            if hasattr(self, 'prev_gray'):
                flow = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, None, None
                )
                if flow[0] is not None:
                    magnitude = np.sqrt(flow[0][:, :, 0]**2 + flow[0][:, :, 1]**2)
                    return np.mean(magnitude) > 5.0
            
            self.prev_gray = gray.copy()
            return False
        except:
            return False
    
    def _detect_hand_regions(self, gray: np.ndarray) -> List:
        """Detect potential hand regions"""
        hand_regions = []
        
        try:
            if self.hand_cascade is not None:
                hands = self.hand_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5
                )
                hand_regions = hands.tolist()
        except:
            pass
            
        # Alternative: Use contour detection for hand-like shapes
        if len(hand_regions) == 0:
            try:
                # Edge detection
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(
                    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Filter contours by size and shape
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 1000 < area < 10000:  # Hand-sized area
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h
                        if 0.5 < aspect_ratio < 2.0:  # Hand-like aspect ratio
                            hand_regions.append([x, y, w, h])
            except:
                pass
                
        return hand_regions
    
    def _detect_waving_motion(self, frame: np.ndarray) -> float:
        """Detect waving motion patterns"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Focus on upper body region for waving
            upper_region = gray[:height//2, :]
            
            # Track motion using frame difference method (more stable)
            if hasattr(self, 'prev_upper_frame'):
                # Calculate frame difference
                diff = cv2.absdiff(self.prev_upper_frame, upper_region)
                
                # Threshold the difference
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                
                # Find contours of moving areas
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Analyze motion patterns
                horizontal_motion_count = 0
                total_motion_area = 0
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:  # Significant motion
                        # Get bounding rectangle
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Check if motion is more horizontal than vertical (waving pattern)
                        if w > h * 1.2:  # Horizontal motion
                            horizontal_motion_count += 1
                        total_motion_area += area
                
                # Calculate waving score based on horizontal motion
                if total_motion_area > 500:  # Sufficient motion detected
                    waving_score = min(horizontal_motion_count / max(len(contours), 1), 1.0)
                    self.prev_upper_frame = upper_region.copy()
                    return waving_score * 0.8  # Scale down for stability
            
            self.prev_upper_frame = upper_region.copy()
            return 0.0
            
        except Exception as e:
            print(f"Waving detection error: {e}")
            return 0.0
    
    def _detect_hands_up_pose(self, frame: np.ndarray) -> float:
        """Detect hands-up pose using contour analysis"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Focus on upper 60% of frame
            upper_region = gray[:int(height * 0.6), :]
            
            # Edge detection for hand/arm contours
            edges = cv2.Canny(upper_region, 30, 100)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze contours for vertical arm-like shapes
            vertical_shapes = 0
            total_shapes = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 5000:  # Reasonable arm/hand size
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = h / w if w > 0 else 0
                    
                    # Look for vertical shapes (arms raised)
                    if aspect_ratio > 1.5 and y < height * 0.4:  # Tall shapes in upper area
                        vertical_shapes += 1
                    total_shapes += 1
            
            # Calculate hands-up score
            if total_shapes > 0:
                hands_up_score = vertical_shapes / max(total_shapes, 1)
                return min(hands_up_score, 1.0)
            
            return 0.0
            
        except Exception as e:
            print(f"Hands-up detection error: {e}")
            return 0.0
    
    def _detect_upper_body_activity(self, frame: np.ndarray) -> float:
        """Detect activity in upper body region"""
        try:
            height, width = frame.shape[:2]
            upper_region = frame[:height//2, :]  # Upper half of frame
            
            # Convert to grayscale and detect edges
            gray_upper = cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_upper, 50, 150)
            
            # Calculate activity level
            edge_pixels = cv2.countNonZero(edges)
            total_pixels = gray_upper.shape[0] * gray_upper.shape[1]
            activity_ratio = edge_pixels / total_pixels
            
            return min(activity_ratio * 10, 1.0)  # Normalize to 0-1
        except:
            return 0.0
    
    def get_gesture_history(self) -> List[Dict]:
        """Get recent gesture detection history"""
        return self.gesture_history[-10:]  # Last 10 detections
    
    def reset_history(self):
        """Reset gesture detection history"""
        self.gesture_history.clear()