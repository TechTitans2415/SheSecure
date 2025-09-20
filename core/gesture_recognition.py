import cv2
import mediapipe as mp
import numpy as np
import time
import math
from typing import Tuple, List, Dict, Optional

class SOSGestureDetector:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Hand detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Pose detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Gesture tracking
        self.gesture_history = []
        self.distress_patterns = []
        self.last_gesture_time = 0
        
        # SOS gesture definitions
        self.sos_gestures = {
            'help_wave': {'confidence': 0, 'duration': 0, 'detected': False},
            'hands_up_surrender': {'confidence': 0, 'duration': 0, 'detected': False},
            'distress_signal': {'confidence': 0, 'duration': 0, 'detected': False},
            'panic_movement': {'confidence': 0, 'duration': 0, 'detected': False},
            'restraint_struggle': {'confidence': 0, 'duration': 0, 'detected': False}
        }
        
    def detect_sos_gestures(self, frame: np.ndarray, person_bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Detect SOS gestures within a person's bounding box
        """
        x1, y1, x2, y2 = person_bbox
        person_region = frame[y1:y2, x1:x2]
        
        if person_region.size == 0:
            return {'gestures': [], 'emergency_level': 'NONE', 'confidence': 0}
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(person_region, cv2.COLOR_BGR2RGB)
        
        # Detect hands and pose
        hand_results = self.hands.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)
        
        current_time = time.time()
        detected_gestures = []
        emergency_level = 'NONE'
        max_confidence = 0
        
        # Analyze hand gestures
        if hand_results.multi_hand_landmarks:
            hand_gestures = self._analyze_hand_gestures(hand_results, person_region.shape)
            detected_gestures.extend(hand_gestures)
        
        # Analyze body pose
        if pose_results.pose_landmarks:
            pose_gestures = self._analyze_pose_gestures(pose_results, person_region.shape)
            detected_gestures.extend(pose_gestures)
        
        # Analyze movement patterns
        movement_analysis = self._analyze_movement_patterns(person_bbox, current_time)
        if movement_analysis:
            detected_gestures.extend(movement_analysis)
        
        # Determine emergency level
        if detected_gestures:
            confidences = [g['confidence'] for g in detected_gestures]
            max_confidence = max(confidences)
            
            # Emergency level determination
            critical_gestures = ['hands_up_surrender', 'restraint_struggle', 'panic_movement']
            high_gestures = ['help_wave', 'distress_signal']
            
            for gesture in detected_gestures:
                if gesture['type'] in critical_gestures and gesture['confidence'] > 0.7:
                    emergency_level = 'CRITICAL'
                    break
                elif gesture['type'] in high_gestures and gesture['confidence'] > 0.6:
                    emergency_level = 'HIGH'
        
        return {
            'gestures': detected_gestures,
            'emergency_level': emergency_level,
            'confidence': max_confidence,
            'timestamp': current_time
        }
    
    def _analyze_hand_gestures(self, hand_results, frame_shape) -> List[Dict]:
        """Analyze hand landmarks for SOS gestures"""
        gestures = []
        
        for hand_landmarks in hand_results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y])
            
            # Convert to numpy array
            landmarks = np.array(landmarks)
            
            # 1. Help Wave Detection (rapid hand movement up and down)
            wave_confidence = self._detect_help_wave(landmarks)
            if wave_confidence > 0.5:
                gestures.append({
                    'type': 'help_wave',
                    'confidence': wave_confidence,
                    'description': 'Rapid waving motion detected - possible distress signal',
                    'coordinates': landmarks
                })
            
            # 2. Hands Up Surrender (both hands raised above head)
            surrender_confidence = self._detect_hands_up(landmarks, frame_shape)
            if surrender_confidence > 0.6:
                gestures.append({
                    'type': 'hands_up_surrender',
                    'confidence': surrender_confidence,
                    'description': 'Hands raised in surrender position - potential threat',
                    'coordinates': landmarks
                })
            
            # 3. Distress Signal (specific finger positions)
            distress_confidence = self._detect_distress_signal(landmarks)
            if distress_confidence > 0.5:
                gestures.append({
                    'type': 'distress_signal',
                    'confidence': distress_confidence,
                    'description': 'Distress hand signal detected',
                    'coordinates': landmarks
                })
        
        return gestures
    
    def _analyze_pose_gestures(self, pose_results, frame_shape) -> List[Dict]:
        """Analyze body pose for distress indicators"""
        gestures = []
        landmarks = []
        
        for lm in pose_results.pose_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.visibility])
        
        landmarks = np.array(landmarks)
        
        # 1. Panic Movement (erratic body movement)
        panic_confidence = self._detect_panic_movement(landmarks)
        if panic_confidence > 0.6:
            gestures.append({
                'type': 'panic_movement',
                'confidence': panic_confidence,
                'description': 'Erratic body movement - possible panic or struggle',
                'coordinates': landmarks
            })
        
        # 2. Restraint Struggle (specific body positions indicating restraint)
        restraint_confidence = self._detect_restraint_struggle(landmarks)
        if restraint_confidence > 0.7:
            gestures.append({
                'type': 'restraint_struggle',
                'confidence': restraint_confidence,
                'description': 'Body position suggests physical restraint or struggle',
                'coordinates': landmarks
            })
        
        return gestures
    
    def _detect_help_wave(self, landmarks) -> float:
        """Detect rapid waving motion"""
        if len(landmarks) < 21:  # Standard hand has 21 landmarks
            return 0.0
        
        # Get wrist and middle finger tip
        wrist = landmarks[0]
        middle_tip = landmarks[12]
        
        # Calculate hand orientation and movement
        hand_vector = middle_tip - wrist
        hand_angle = math.atan2(hand_vector[1], hand_vector[0])
        
        # Store in history for movement analysis
        current_time = time.time()
        self.gesture_history.append({
            'time': current_time,
            'hand_angle': hand_angle,
            'hand_height': middle_tip[1]
        })
        
        # Keep only last 2 seconds
        self.gesture_history = [h for h in self.gesture_history if current_time - h['time'] < 2.0]
        
        if len(self.gesture_history) < 10:
            return 0.0
        
        # Analyze for rapid up-down movement
        heights = [h['hand_height'] for h in self.gesture_history]
        height_variation = np.std(heights)
        
        # Calculate movement frequency
        peaks = 0
        for i in range(1, len(heights) - 1):
            if heights[i] > heights[i-1] and heights[i] > heights[i+1]:
                peaks += 1
        
        frequency = peaks / 2.0  # 2 seconds of data
        
        # Wave confidence based on height variation and frequency
        confidence = min(1.0, (height_variation * 10) * (frequency / 3.0))
        return confidence
    
    def _detect_hands_up(self, landmarks, frame_shape) -> float:
        """Detect hands raised above head in surrender position"""
        if len(landmarks) < 21:
            return 0.0
        
        wrist = landmarks[0]
        middle_tip = landmarks[12]
        
        # Check if hand is raised high (above shoulder level approximately)
        hand_raised = wrist[1] < 0.3  # Top 30% of the frame
        
        # Check if hand is clearly visible and upward
        hand_up_vector = middle_tip[1] - wrist[1]
        hand_pointing_up = hand_up_vector < -0.1
        
        confidence = 0.0
        if hand_raised and hand_pointing_up:
            confidence = 0.8
        elif hand_raised:
            confidence = 0.5
        
        return confidence
    
    def _detect_distress_signal(self, landmarks) -> float:
        """Detect specific distress hand signals"""
        if len(landmarks) < 21:
            return 0.0
        
        # Check for closed fist (fingers curled inward)
        finger_tips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        finger_bases = [landmarks[3], landmarks[6], landmarks[10], landmarks[14], landmarks[18]]
        
        fingers_curled = 0
        for tip, base in zip(finger_tips, finger_bases):
            # If tip is below base, finger is likely curled
            if tip[1] > base[1]:
                fingers_curled += 1
        
        # Closed fist confidence
        fist_confidence = fingers_curled / 5.0
        
        # Additional checks for specific distress signals can be added here
        return fist_confidence * 0.7  # Moderate confidence for basic fist detection
    
    def _detect_panic_movement(self, landmarks) -> float:
        """Detect erratic body movement indicating panic"""
        if len(landmarks) < 33:  # Standard pose has 33 landmarks
            return 0.0
        
        # Get key body points
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Calculate body center
        body_center = [(left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4,
                      (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) / 4]
        
        current_time = time.time()
        
        # Store movement history
        self.distress_patterns.append({
            'time': current_time,
            'body_center': body_center,
            'shoulder_width': abs(right_shoulder[0] - left_shoulder[0])
        })
        
        # Keep only last 3 seconds
        self.distress_patterns = [d for d in self.distress_patterns if current_time - d['time'] < 3.0]
        
        if len(self.distress_patterns) < 15:
            return 0.0
        
        # Analyze movement patterns
        centers = [d['body_center'] for d in self.distress_patterns]
        center_variance = np.var([c[0] for c in centers]) + np.var([c[1] for c in centers])
        
        # High variance indicates erratic movement
        panic_confidence = min(1.0, center_variance * 50)
        return panic_confidence
    
    def _detect_restraint_struggle(self, landmarks) -> float:
        """Detect body positions suggesting physical restraint"""
        if len(landmarks) < 33:
            return 0.0
        
        # Get arm positions
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_elbow = landmarks[13]
        right_elbow = landmarks[14]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        
        # Check for arms behind back (indicating restraint)
        left_arm_behind = left_wrist[0] > left_shoulder[0] and left_elbow[0] > left_shoulder[0]
        right_arm_behind = right_wrist[0] < right_shoulder[0] and right_elbow[0] < right_shoulder[0]
        
        # Check for unnatural arm positions
        left_arm_unnatural = abs(left_wrist[1] - left_shoulder[1]) > 0.3
        right_arm_unnatural = abs(right_wrist[1] - right_shoulder[1]) > 0.3
        
        restraint_indicators = 0
        if left_arm_behind: restraint_indicators += 1
        if right_arm_behind: restraint_indicators += 1
        if left_arm_unnatural: restraint_indicators += 1
        if right_arm_unnatural: restraint_indicators += 1
        
        confidence = restraint_indicators / 4.0
        return confidence
    
    def _analyze_movement_patterns(self, person_bbox, current_time) -> List[Dict]:
        """Analyze overall movement patterns for distress"""
        gestures = []
        
        # This would analyze movement patterns over time
        # For now, return empty list - can be enhanced with tracking
        
        return gestures
    
    def draw_gesture_annotations(self, frame: np.ndarray, person_bbox: Tuple[int, int, int, int], 
                               gesture_results: Dict) -> np.ndarray:
        """Draw gesture detection annotations on frame"""
        x1, y1, x2, y2 = person_bbox
        
        if gesture_results['emergency_level'] != 'NONE':
            # Draw emergency border around person
            if gesture_results['emergency_level'] == 'CRITICAL':
                color = (0, 0, 255)  # Red
                thickness = 6
            else:
                color = (0, 165, 255)  # Orange
                thickness = 4
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw gesture information
            for i, gesture in enumerate(gesture_results['gestures']):
                text = f"{gesture['type']}: {gesture['confidence']:.2f}"
                cv2.putText(frame, text, (x1, y1 - 30 - (i * 20)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw emergency level
            emergency_text = f"EMERGENCY: {gesture_results['emergency_level']}"
            cv2.putText(frame, emergency_text, (x1, y2 + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return frame

# Global gesture detector instance
gesture_detector = SOSGestureDetector()

def detect_sos_gestures(frame: np.ndarray, person_bboxes: List[Tuple[int, int, int, int]]) -> List[Dict]:
    """
    Main function to detect SOS gestures for all persons in frame
    """
    results = []
    
    for bbox in person_bboxes:
        gesture_result = gesture_detector.detect_sos_gestures(frame, bbox)
        if gesture_result['emergency_level'] != 'NONE':
            gesture_result['bbox'] = bbox
            results.append(gesture_result)
            
            # Draw annotations
            frame = gesture_detector.draw_gesture_annotations(frame, bbox, gesture_result)
    
    return results
