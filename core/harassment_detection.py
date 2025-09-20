import cv2
import numpy as np
import time
import math
from typing import List, Dict, Tuple, Optional
from collections import deque

class HarassmentKidnapDetector:
    def __init__(self):
        self.tracking_history = deque(maxlen=100)  # Keep last 100 frames
        self.interaction_patterns = {}
        self.threat_indicators = {
            'aggressive_approach': 0,
            'cornering_behavior': 0,
            'forced_contact': 0,
            'escape_blocking': 0,
            'isolation_attempt': 0
        }
        
        # Scenario detection thresholds
        self.harassment_threshold = 0.7
        self.kidnap_threshold = 0.8
        
    def analyze_interactions(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        """
        Analyze interactions between people to detect harassment/kidnap scenarios
        """
        current_time = time.time()
        
        # Filter detections to get men and women
        men = [d for d in detections if d.get('gender') == 'Man']
        women = [d for d in detections if d.get('gender') == 'Woman']
        
        if not women:
            return {'scenario': 'NONE', 'confidence': 0, 'details': []}
        
        scenario_results = []
        
        # Analyze each woman's situation
        for woman in women:
            woman_analysis = self._analyze_woman_situation(woman, men, current_time)
            if woman_analysis['threat_level'] != 'NONE':
                scenario_results.append(woman_analysis)
        
        # Determine overall scenario
        if scenario_results:
            max_threat = max(scenario_results, key=lambda x: x['confidence'])
            return {
                'scenario': max_threat['threat_level'],
                'confidence': max_threat['confidence'],
                'details': scenario_results,
                'timestamp': current_time
            }
        
        return {'scenario': 'NONE', 'confidence': 0, 'details': []}
    
    def _analyze_woman_situation(self, woman: Dict, men: List[Dict], current_time: float) -> Dict:
        """Analyze a specific woman's situation with surrounding men"""
        woman_bbox = woman['bbox']
        woman_center = self._get_center(woman_bbox)
        
        threat_indicators = {}
        threat_level = 'NONE'
        confidence = 0
        
        if not men:
            return {'threat_level': 'NONE', 'confidence': 0, 'indicators': {}}
        
        # 1. HARASSMENT DETECTION
        harassment_score = self._detect_harassment_patterns(woman, men, current_time)
        if harassment_score > self.harassment_threshold:
            threat_level = 'HARASSMENT'
            confidence = harassment_score
            threat_indicators['harassment'] = harassment_score
        
        # 2. KIDNAP ATTEMPT DETECTION
        kidnap_score = self._detect_kidnap_patterns(woman, men, current_time)
        if kidnap_score > self.kidnap_threshold:
            threat_level = 'KIDNAP_ATTEMPT'
            confidence = max(confidence, kidnap_score)
            threat_indicators['kidnap'] = kidnap_score
        
        # 3. AGGRESSIVE APPROACH
        aggressive_score = self._detect_aggressive_approach(woman, men)
        if aggressive_score > 0.6:
            threat_indicators['aggressive_approach'] = aggressive_score
            if confidence < aggressive_score:
                threat_level = 'AGGRESSIVE_BEHAVIOR'
                confidence = aggressive_score
        
        # 4. CORNERING BEHAVIOR
        cornering_score = self._detect_cornering_behavior(woman, men)
        if cornering_score > 0.7:
            threat_indicators['cornering'] = cornering_score
            if confidence < cornering_score:
                threat_level = 'CORNERING'
                confidence = cornering_score
        
        # 5. ESCAPE ROUTE BLOCKING
        blocking_score = self._detect_escape_blocking(woman, men)
        if blocking_score > 0.6:
            threat_indicators['escape_blocking'] = blocking_score
            confidence = max(confidence, blocking_score)
        
        return {
            'threat_level': threat_level,
            'confidence': confidence,
            'indicators': threat_indicators,
            'woman_location': woman_center,
            'men_count': len(men),
            'timestamp': current_time
        }
    
    def _detect_harassment_patterns(self, woman: Dict, men: List[Dict], current_time: float) -> float:
        """Detect harassment behavior patterns"""
        woman_bbox = woman['bbox']
        woman_center = self._get_center(woman_bbox)
        
        harassment_indicators = []
        
        # 1. Multiple men approaching single woman
        close_men = 0
        for man in men:
            man_center = self._get_center(man['bbox'])
            distance = self._calculate_distance(woman_center, man_center)
            
            if distance < 150:  # Close proximity threshold
                close_men += 1
        
        if close_men >= 2:
            harassment_indicators.append(0.6 + (close_men - 2) * 0.1)
        
        # 2. Persistent following (requires tracking history)
        following_score = self._detect_following_behavior(woman, men, current_time)
        if following_score > 0:
            harassment_indicators.append(following_score)
        
        # 3. Intimidating positioning (surrounding)
        surrounding_score = self._detect_surrounding_pattern(woman, men)
        if surrounding_score > 0:
            harassment_indicators.append(surrounding_score)
        
        # 4. Rapid approach movements
        rapid_approach_score = self._detect_rapid_approach(woman, men)
        if rapid_approach_score > 0:
            harassment_indicators.append(rapid_approach_score)
        
        return max(harassment_indicators) if harassment_indicators else 0.0
    
    def _detect_kidnap_patterns(self, woman: Dict, men: List[Dict], current_time: float) -> float:
        """Detect kidnap attempt patterns"""
        kidnap_indicators = []
        
        # 1. Physical contact/grabbing (proximity analysis)
        contact_score = self._detect_physical_contact(woman, men)
        if contact_score > 0:
            kidnap_indicators.append(contact_score)
        
        # 2. Forced movement (direction analysis)
        forced_movement_score = self._detect_forced_movement(woman, men, current_time)
        if forced_movement_score > 0:
            kidnap_indicators.append(forced_movement_score)
        
        # 3. Multiple attackers coordination
        coordination_score = self._detect_coordinated_attack(woman, men)
        if coordination_score > 0:
            kidnap_indicators.append(coordination_score)
        
        # 4. Isolation attempt (moving away from public areas)
        isolation_score = self._detect_isolation_attempt(woman, men, current_time)
        if isolation_score > 0:
            kidnap_indicators.append(isolation_score)
        
        return max(kidnap_indicators) if kidnap_indicators else 0.0
    
    def _detect_aggressive_approach(self, woman: Dict, men: List[Dict]) -> float:
        """Detect aggressive approach patterns"""
        woman_center = self._get_center(woman['bbox'])
        aggressive_score = 0
        
        for man in men:
            man_center = self._get_center(man['bbox'])
            distance = self._calculate_distance(woman_center, man_center)
            
            # Very close proximity indicates aggressive behavior
            if distance < 80:  # Very close
                aggressive_score = max(aggressive_score, 0.9)
            elif distance < 120:  # Close
                aggressive_score = max(aggressive_score, 0.7)
            elif distance < 160:  # Approaching
                aggressive_score = max(aggressive_score, 0.5)
        
        return aggressive_score
    
    def _detect_cornering_behavior(self, woman: Dict, men: List[Dict]) -> float:
        """Detect if woman is being cornered"""
        woman_center = self._get_center(woman['bbox'])
        
        if len(men) < 2:
            return 0.0
        
        # Calculate angles of men relative to woman
        angles = []
        for man in men:
            man_center = self._get_center(man['bbox'])
            distance = self._calculate_distance(woman_center, man_center)
            
            if distance < 200:  # Only consider nearby men
                angle = math.atan2(man_center[1] - woman_center[1], 
                                 man_center[0] - woman_center[0])
                angles.append(angle)
        
        if len(angles) < 2:
            return 0.0
        
        # Check if men are positioned to block escape routes
        angles.sort()
        max_gap = 0
        
        for i in range(len(angles)):
            gap = angles[(i + 1) % len(angles)] - angles[i]
            if gap < 0:
                gap += 2 * math.pi
            max_gap = max(max_gap, gap)
        
        # If largest gap is small, woman is being cornered
        cornering_score = 1.0 - (max_gap / (2 * math.pi))
        return max(0.0, cornering_score)
    
    def _detect_escape_blocking(self, woman: Dict, men: List[Dict]) -> float:
        """Detect if escape routes are being blocked"""
        woman_bbox = woman['bbox']
        frame_edges = {
            'left': 0,
            'right': 1920,  # Assume 1920x1080 frame
            'top': 0,
            'bottom': 1080
        }
        
        # Calculate distances to frame edges
        edge_distances = {
            'left': woman_bbox[0],
            'right': frame_edges['right'] - woman_bbox[2],
            'top': woman_bbox[1],
            'bottom': frame_edges['bottom'] - woman_bbox[3]
        }
        
        # Check if men are blocking paths to edges
        blocked_exits = 0
        total_exits = 4
        
        woman_center = self._get_center(woman_bbox)
        
        for direction, distance in edge_distances.items():
            if distance < 200:  # Near an edge
                # Check if any man is blocking this direction
                for man in men:
                    man_center = self._get_center(man['bbox'])
                    
                    if direction == 'left' and man_center[0] < woman_center[0]:
                        if abs(man_center[1] - woman_center[1]) < 100:
                            blocked_exits += 1
                            break
                    elif direction == 'right' and man_center[0] > woman_center[0]:
                        if abs(man_center[1] - woman_center[1]) < 100:
                            blocked_exits += 1
                            break
                    elif direction == 'top' and man_center[1] < woman_center[1]:
                        if abs(man_center[0] - woman_center[0]) < 100:
                            blocked_exits += 1
                            break
                    elif direction == 'bottom' and man_center[1] > woman_center[1]:
                        if abs(man_center[0] - woman_center[0]) < 100:
                            blocked_exits += 1
                            break
        
        return blocked_exits / total_exits
    
    def _detect_following_behavior(self, woman: Dict, men: List[Dict], current_time: float) -> float:
        """Detect persistent following behavior"""
        # This would require frame-to-frame tracking
        # For now, return basic proximity-based score
        return 0.0
    
    def _detect_surrounding_pattern(self, woman: Dict, men: List[Dict]) -> float:
        """Detect if woman is surrounded"""
        if len(men) < 3:
            return 0.0
        
        woman_center = self._get_center(woman['bbox'])
        close_men = []
        
        for man in men:
            man_center = self._get_center(man['bbox'])
            distance = self._calculate_distance(woman_center, man_center)
            
            if distance < 200:
                close_men.append(man_center)
        
        if len(close_men) < 3:
            return 0.0
        
        # Calculate if men form a rough circle around woman
        angles = []
        for man_center in close_men:
            angle = math.atan2(man_center[1] - woman_center[1], 
                             man_center[0] - woman_center[0])
            angles.append(angle)
        
        angles.sort()
        
        # Check angle distribution
        ideal_gap = 2 * math.pi / len(angles)
        actual_gaps = []
        
        for i in range(len(angles)):
            gap = angles[(i + 1) % len(angles)] - angles[i]
            if gap < 0:
                gap += 2 * math.pi
            actual_gaps.append(gap)
        
        # Calculate how evenly distributed the men are
        gap_variance = np.var(actual_gaps)
        surrounding_score = max(0.0, 1.0 - gap_variance)
        
        return surrounding_score * (len(close_men) / 5.0)  # Scale by number of men
    
    def _detect_rapid_approach(self, woman: Dict, men: List[Dict]) -> float:
        """Detect rapid approach movements"""
        # This would require movement tracking between frames
        # For now, return 0
        return 0.0
    
    def _detect_physical_contact(self, woman: Dict, men: List[Dict]) -> float:
        """Detect potential physical contact/grabbing"""
        woman_center = self._get_center(woman['bbox'])
        contact_score = 0
        
        for man in men:
            man_center = self._get_center(man['bbox'])
            distance = self._calculate_distance(woman_center, man_center)
            
            # Very close proximity suggests contact
            if distance < 50:  # Overlapping bounding boxes
                contact_score = max(contact_score, 0.9)
            elif distance < 80:  # Very close
                contact_score = max(contact_score, 0.7)
        
        return contact_score
    
    def _detect_forced_movement(self, woman: Dict, men: List[Dict], current_time: float) -> float:
        """Detect forced movement patterns"""
        # This would require tracking movement vectors
        # For now, return 0
        return 0.0
    
    def _detect_coordinated_attack(self, woman: Dict, men: List[Dict]) -> float:
        """Detect coordinated attack patterns"""
        if len(men) < 2:
            return 0.0
        
        woman_center = self._get_center(woman['bbox'])
        
        # Check if multiple men are positioned strategically
        strategic_positions = 0
        
        for man in men:
            man_center = self._get_center(man['bbox'])
            distance = self._calculate_distance(woman_center, man_center)
            
            if 80 < distance < 150:  # Optimal attack distance
                strategic_positions += 1
        
        coordination_score = strategic_positions / len(men)
        return coordination_score if strategic_positions >= 2 else 0.0
    
    def _detect_isolation_attempt(self, woman: Dict, men: List[Dict], current_time: float) -> float:
        """Detect attempts to isolate the woman"""
        # This would analyze movement toward isolated areas
        # For now, return 0
        return 0.0
    
    def _get_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def draw_threat_annotations(self, frame: np.ndarray, analysis_result: Dict) -> np.ndarray:
        """Draw threat detection annotations on frame"""
        if analysis_result['scenario'] == 'NONE':
            return frame
        
        # Choose color based on threat level
        if analysis_result['scenario'] == 'KIDNAP_ATTEMPT':
            color = (0, 0, 255)  # Red
            thickness = 8
        elif analysis_result['scenario'] == 'HARASSMENT':
            color = (0, 100, 255)  # Orange-Red
            thickness = 6
        else:
            color = (0, 165, 255)  # Orange
            thickness = 4
        
        # Draw frame border
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), color, thickness)
        
        # Draw threat level text
        threat_text = f"THREAT DETECTED: {analysis_result['scenario']}"
        confidence_text = f"Confidence: {analysis_result['confidence']:.2f}"
        
        cv2.putText(frame, threat_text, (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, confidence_text, (50, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw detailed indicators for each woman
        y_offset = 160
        for detail in analysis_result['details']:
            if detail['indicators']:
                for indicator, score in detail['indicators'].items():
                    indicator_text = f"{indicator}: {score:.2f}"
                    cv2.putText(frame, indicator_text, (50, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += 25
        
        return frame

# Global harassment detector instance
harassment_detector = HarassmentKidnapDetector()

def detect_harassment_kidnap(frame: np.ndarray, detections: List[Dict]) -> Dict:
    """
    Main function to detect harassment and kidnap scenarios
    """
    return harassment_detector.analyze_interactions(frame, detections)