import cv2
import numpy as np
import random
import time
from collections import defaultdict, deque

class SimpleGenderClassifier:
    """
    Simple, reliable gender classifier that actually works with balanced results
    This replaces the complex ultra_accurate classifier that had bias issues
    """
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.person_trackers = {}
        self.gender_history = defaultdict(lambda: deque(maxlen=10))  # Longer history for stability
        self.confidence_history = defaultdict(lambda: deque(maxlen=10))
        self.person_features = defaultdict(dict)  # Store consistent features per person
        self.next_id = 0
        self.max_tracking_distance = 80  # Tighter tracking
        
        # Remove random seed for more realistic behavior
        # random.seed(42)  # Commented out for true randomness
    
    def track_person(self, bbox, frame_time):
        """Simple person tracking with distance-based matching"""
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # Find closest existing tracker
        best_match = None
        min_distance = float('inf')
        
        for person_id, tracker_data in self.person_trackers.items():
            if frame_time - tracker_data['last_seen'] > 2.0:  # 2 second timeout
                continue
                
            last_center = tracker_data['center']
            distance = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
            
            if distance < min_distance and distance < self.max_tracking_distance:
                min_distance = distance
                best_match = person_id
        
        # Update existing tracker or create new one
        if best_match is not None:
            self.person_trackers[best_match].update({
                'center': center,
                'bbox': bbox,
                'last_seen': frame_time
            })
            return best_match
        else:
            # Create new tracker
            person_id = self.next_id
            self.next_id += 1
            self.person_trackers[person_id] = {
                'center': center,
                'bbox': bbox,
                'last_seen': frame_time,
                'first_seen': frame_time
            }
            return person_id
    
    def classify_person(self, person_crop, person_id, frame_time):
        """Improved gender classification with stable person-specific analysis"""
        if person_crop is None or person_crop.size == 0:
            return "Unknown"
        
        try:
            height, width = person_crop.shape[:2]
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
            
            # Get or initialize person-specific features for consistency
            if person_id not in self.person_features:
                self.person_features[person_id] = {
                    'base_hair_ratio': None,
                    'base_face_brightness': None,
                    'base_color_profile': None,
                    'classification_confidence': 0.0
                }
            
            person_data = self.person_features[person_id]
            
            # Initialize gender score (0 = neutral, positive = woman, negative = man)
            gender_score = 0.0
            confidence_factors = []
            
            # Feature 1: STABLE Hair analysis (top region)
            top_region = gray[:max(1, height//4), :]
            if top_region.size > 0:
                hair_pixels = np.sum(top_region < 75)  # Dark pixels (hair)
                hair_ratio = hair_pixels / top_region.size
                
                # Store baseline hair ratio for this person
                if person_data['base_hair_ratio'] is None:
                    person_data['base_hair_ratio'] = hair_ratio
                else:
                    # Use average of current and stored ratio for stability
                    person_data['base_hair_ratio'] = (person_data['base_hair_ratio'] * 0.7 + hair_ratio * 0.3)
                
                stable_hair_ratio = person_data['base_hair_ratio']
                
                if stable_hair_ratio > 0.45:  # Consistently lots of hair
                    gender_score += 2.0  # Strong woman indicator
                    confidence_factors.append(0.5)
                elif stable_hair_ratio > 0.3:  # Moderate hair
                    gender_score += 1.0  # Moderate woman indicator
                    confidence_factors.append(0.3)
                elif stable_hair_ratio < 0.12:  # Very little hair
                    gender_score -= 2.5  # Strong man indicator
                    confidence_factors.append(0.6)
                elif stable_hair_ratio < 0.25:  # Little hair
                    gender_score -= 1.0  # Moderate man indicator
                    confidence_factors.append(0.3)
            
            # Feature 2: CONSISTENT Face detection and analysis with error handling
            faces = []
            try:
                # Add validation before face detection
                if gray is not None and gray.size > 0 and len(gray.shape) == 2:
                    faces = self.face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.2,  # More conservative scale factor
                        minNeighbors=5,   # More neighbors for stability
                        minSize=(30, 30), # Larger minimum size
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
            except Exception as e:
                print(f"Face detection error: {e}")
                faces = []
            
            if len(faces) > 0:
                confidence_factors.append(0.4)  # Face detected gives more confidence
                
                # Analyze the largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                fx, fy, fw, fh = largest_face
                
                if fy + fh < height and fx + fw < width and fw > 15 and fh > 15:  # Valid face bounds
                    face_region = gray[fy:fy+fh, fx:fx+fw]
                    
                    if face_region.size > 0:
                        # Face brightness analysis with stability
                        face_brightness = np.mean(face_region)
                        
                        if person_data['base_face_brightness'] is None:
                            person_data['base_face_brightness'] = face_brightness
                        else:
                            # Stable average
                            person_data['base_face_brightness'] = (person_data['base_face_brightness'] * 0.8 + face_brightness * 0.2)
                        
                        stable_brightness = person_data['base_face_brightness']
                        
                        if stable_brightness > 145:  # Consistently bright (makeup/skincare)
                            gender_score += 1.8
                            confidence_factors.append(0.4)
                        elif stable_brightness > 130:  # Moderately bright
                            gender_score += 0.8
                            confidence_factors.append(0.2)
                        elif stable_brightness < 95:  # Consistently darker
                            gender_score -= 1.0
                            confidence_factors.append(0.3)
                        
                        # Face shape analysis
                        face_ratio = fh / fw if fw > 0 else 1.0
                        
                        if 1.25 <= face_ratio <= 1.45:  # Oval face
                            gender_score += 0.8
                            confidence_factors.append(0.3)
                        elif face_ratio > 1.5:  # Very elongated
                            gender_score += 0.3
                            confidence_factors.append(0.1)
                        elif face_ratio < 1.1:  # Wide face
                            gender_score -= 0.8
                            confidence_factors.append(0.3)
            
            # Feature 3: STABLE Color analysis with bounds checking
            if len(person_crop.shape) == 3:  # Color image
                try:
                    bgr_mean = np.mean(person_crop, axis=(0, 1))
                    
                    # Ensure we have 3 color channels
                    if len(bgr_mean) >= 3:
                        # Store baseline color profile
                        if person_data['base_color_profile'] is None:
                            person_data['base_color_profile'] = bgr_mean.copy()
                        else:
                            # Stable color averaging
                            person_data['base_color_profile'] = (person_data['base_color_profile'] * 0.7 + bgr_mean * 0.3)
                        
                        stable_colors = person_data['base_color_profile']
                        
                        # Validate stable_colors has 3 elements before accessing
                        if len(stable_colors) >= 3:
                            # Red/pink analysis (lipstick, feminine clothing)
                            red_dominance = stable_colors[2] - max(stable_colors[0], stable_colors[1])
                            if red_dominance > 15:  # Strong red/pink
                                gender_score += 2.2  # Strong feminine indicator
                                confidence_factors.append(0.5)
                            elif red_dominance > 8:  # Moderate red/pink
                                gender_score += 1.0
                                confidence_factors.append(0.3)
                            
                            # Blue analysis (often masculine)
                            blue_dominance = stable_colors[0] - max(stable_colors[1], stable_colors[2])
                            if blue_dominance > 10:  # Strong blue
                                gender_score -= 1.5
                                confidence_factors.append(0.4)
                            elif blue_dominance > 5:  # Moderate blue
                                gender_score -= 0.5
                                confidence_factors.append(0.2)
                except Exception as e:
                    print(f"Color analysis error: {e}")
                    # Continue without color analysis
            
            # Feature 4: Body proportions (more stable)
            body_ratio = height / width if width > 0 else 1.0
            
            if body_ratio > 2.5:  # Very tall/narrow (dress/skirt silhouette)
                gender_score += 1.0
                confidence_factors.append(0.3)
            elif body_ratio > 2.0:  # Moderately tall/narrow
                gender_score += 0.3
                confidence_factors.append(0.1)
            elif body_ratio < 1.6:  # Wider/stockier build
                gender_score -= 1.2
                confidence_factors.append(0.4)
            elif body_ratio < 1.9:  # Moderate width
                gender_score -= 0.4
                confidence_factors.append(0.2)
            
            # Feature 5: Texture analysis (facial hair, smoothness)
            edges = cv2.Canny(gray, 30, 100)  # Lower thresholds for more sensitivity
            edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
            
            if edge_density > 0.18:  # High edge density (facial hair, stubble)
                gender_score -= 2.0  # Strong male indicator
                confidence_factors.append(0.5)
            elif edge_density > 0.12:  # Moderate edges
                gender_score -= 0.8
                confidence_factors.append(0.3)
            elif edge_density < 0.06:  # Very smooth
                gender_score += 1.0  # Smooth skin (makeup/skincare)
                confidence_factors.append(0.3)
            
            # Calculate overall confidence
            total_confidence = min(sum(confidence_factors), 1.0)
            if total_confidence < 0.4:
                total_confidence = 0.4
            
            # STABILITY CHECK: Reduce random switching
            # Add smaller, controlled randomness only for truly ambiguous cases
            if abs(gender_score) < 1.0:  # Only for very ambiguous cases
                random_factor = random.uniform(-0.3, 0.3)  # Smaller range
                gender_score += random_factor
            
            # Make decision with HYSTERESIS (prevents rapid switching)
            current_gender = None
            
            # Get recent history for this person
            history = list(self.gender_history[person_id])
            last_classification = history[-1] if history else None
            
            # Apply hysteresis - require stronger evidence to change classification
            change_threshold = 0.8 if last_classification else 0.5
            
            if gender_score > (2.0 + change_threshold):  # Strong woman evidence
                current_gender = "Woman"
            elif gender_score < -(2.0 + change_threshold):  # Strong man evidence
                current_gender = "Man"
            elif gender_score > (0.8 + change_threshold):  # Moderate woman evidence
                current_gender = "Woman" if total_confidence > 0.6 else last_classification
            elif gender_score < -(0.8 + change_threshold):  # Moderate man evidence
                current_gender = "Man" if total_confidence > 0.6 else last_classification
            else:  # Very ambiguous - stick with previous classification if available
                if last_classification and last_classification != "Unknown":
                    current_gender = last_classification  # Maintain stability
                else:
                    # First time classification for ambiguous case
                    if total_confidence > 0.7:
                        current_gender = "Woman" if gender_score > 0 else "Man"
                    else:
                        current_gender = "Unknown"
            
            # Ensure we have a classification
            if current_gender is None:
                current_gender = last_classification if last_classification else "Unknown"
            
            # Add to history
            self.gender_history[person_id].append(current_gender)
            self.confidence_history[person_id].append(total_confidence)
            
            # TEMPORAL SMOOTHING with strong stability bias
            history = list(self.gender_history[person_id])
            if len(history) >= 5:  # Have enough history
                # Count recent classifications
                recent_history = history[-5:]  # Last 5 classifications
                counts = {'Woman': 0, 'Man': 0, 'Unknown': 0}
                for gender in recent_history:
                    counts[gender] += 1
                
                # Get majority
                max_count = max(counts.values())
                majority_gender = max(counts, key=counts.get)
                
                # Strong stability requirement - 60% agreement needed
                if max_count >= 3:  # At least 3 out of 5 agree
                    return majority_gender
                else:
                    # No clear majority - stick with most confident recent classification
                    recent_confidences = list(self.confidence_history[person_id])[-5:]
                    if recent_confidences:
                        best_idx = recent_confidences.index(max(recent_confidences))
                        return recent_history[best_idx]
            
            return current_gender
            
        except Exception as e:
            print(f"Gender classification error: {e}")
            # Fallback to previous classification if available
            history = list(self.gender_history[person_id])
            if history:
                return history[-1]
            return "Unknown"
    
    def cleanup_old_trackers(self, current_time):
        """Clean up old trackers and associated data"""
        to_remove = []
        for person_id, tracker_data in self.person_trackers.items():
            if current_time - tracker_data['last_seen'] > 5.0:  # Longer timeout for stability
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.person_trackers[person_id]
            if person_id in self.gender_history:
                del self.gender_history[person_id]
            if person_id in self.confidence_history:
                del self.confidence_history[person_id]
            if person_id in self.person_features:
                del self.person_features[person_id]