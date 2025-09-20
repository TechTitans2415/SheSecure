import cv2
import numpy as np
from collections import defaultdict, deque
import time

class StableGenderClassifier:
    """
    Improved gender classification with temporal smoothing and multiple detection methods
    """
    
    def __init__(self):
        # Tracking for stability
        self.person_trackers = {}
        self.gender_history = defaultdict(lambda: deque(maxlen=10))  # Keep last 10 predictions per person
        self.next_id = 0
        self.max_tracking_distance = 100  # Maximum distance to consider same person
        
        # Enhanced detection parameters
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Gender classification confidence thresholds
        self.min_confidence = 0.6
        self.stability_threshold = 3  # Minimum consistent predictions needed
        
    def track_person(self, bbox, frame_time):
        """Track person across frames for stable gender classification"""
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # Find closest existing tracker
        best_match = None
        min_distance = float('inf')
        
        for person_id, tracker_data in self.person_trackers.items():
            if frame_time - tracker_data['last_seen'] > 3.0:  # Remove stale trackers
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
    
    def extract_multiple_faces(self, person_crop):
        """Extract faces using multiple methods for better detection"""
        faces = []
        
        try:
            gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Frontal face detection
            frontal_faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(24, 24)
            )
            
            for (x, y, w, h) in frontal_faces:
                face = person_crop[y:y+h, x:x+w]
                if face.size > 0:
                    faces.append(('frontal', face, (x, y, w, h)))
            
            # Method 2: Profile face detection
            profile_faces = self.profile_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(24, 24)
            )
            
            for (x, y, w, h) in profile_faces:
                face = person_crop[y:y+h, x:x+w]
                if face.size > 0:
                    faces.append(('profile', face, (x, y, w, h)))
            
            # Method 3: Head region estimation (if no faces detected)
            if not faces:
                h, w = person_crop.shape[:2]
                # Estimate head region (top 25% of person detection)
                head_region = person_crop[:h//4, :]
                if head_region.size > 0:
                    faces.append(('head_estimate', head_region, (0, 0, w, h//4)))
                    
        except Exception as e:
            print(f"Face extraction error: {e}")
        
        return faces
    
    def enhanced_gender_analysis(self, face_region, detection_type='frontal'):
        """Enhanced gender classification with multiple features"""
        try:
            if face_region is None or face_region.size == 0:
                return None, 0.0
            
            # Resize to standard size for consistency
            face = cv2.resize(face_region, (64, 64))
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # Feature extraction
            features = self._extract_facial_features(gray_face, face)
            
            # Gender scoring with different weights based on detection type
            woman_score, man_score, confidence = self._calculate_gender_scores(features, detection_type)
            
            # Decision making
            if confidence < self.min_confidence:
                return None, confidence
            elif woman_score > man_score:
                return "Woman", confidence
            else:
                return "Man", confidence
                
        except Exception as e:
            print(f"Gender analysis error: {e}")
            return None, 0.0
    
    def _extract_facial_features(self, gray_face, color_face):
        """Extract comprehensive facial features for gender classification"""
        h, w = gray_face.shape
        features = {}
        
        try:
            # 1. Hair analysis (improved region selection)
            hair_region = gray_face[:h//4, :]  # Top quarter
            hair_pixels = np.sum(hair_region < 80)  # Dark hair threshold
            features['hair_coverage'] = hair_pixels / hair_region.size if hair_region.size > 0 else 0
            
            # 2. Face proportions
            features['face_ratio'] = h / w if w > 0 else 1.0
            
            # 3. Skin texture analysis (multiple regions)
            cheek_regions = [
                gray_face[h//3:2*h//3, :w//3],      # Left cheek
                gray_face[h//3:2*h//3, 2*w//3:],     # Right cheek
                gray_face[h//4:h//2, w//4:3*w//4]    # Forehead region
            ]
            
            skin_variance = []
            for region in cheek_regions:
                if region.size > 0:
                    skin_variance.append(np.var(region))
            features['skin_smoothness'] = np.mean(skin_variance) if skin_variance else 20
            
            # 4. Eye region analysis
            eye_region = gray_face[h//4:h//2, w//6:5*w//6]
            features['eye_brightness'] = np.mean(eye_region) if eye_region.size > 0 else 128
            features['eye_contrast'] = np.std(eye_region) if eye_region.size > 0 else 15
            
            # 5. Color analysis (using color image)
            hsv_face = cv2.cvtColor(color_face, cv2.COLOR_BGR2HSV)
            
            # Lip region analysis (bottom third, center)
            lip_region = hsv_face[2*h//3:, w//4:3*w//4]
            if lip_region.size > 0:
                # Check for red/pink lip color (makeup indicator)
                red_mask = cv2.inRange(lip_region, (0, 50, 50), (10, 255, 255))
                pink_mask = cv2.inRange(lip_region, (160, 50, 50), (180, 255, 255))
                lip_color_ratio = (np.sum(red_mask) + np.sum(pink_mask)) / (lip_region.shape[0] * lip_region.shape[1] * 255)
                features['lip_color'] = lip_color_ratio
            else:
                features['lip_color'] = 0
            
            # 6. Edge analysis (facial hair detection)
            edges = cv2.Canny(gray_face, 50, 150)
            jaw_region = edges[2*h//3:, :]  # Lower face for facial hair
            features['edge_density'] = np.sum(jaw_region > 0) / jaw_region.size if jaw_region.size > 0 else 0
            
            # 7. Symmetry analysis
            left_half = gray_face[:, :w//2]
            right_half = cv2.flip(gray_face[:, w//2:], 1)
            if left_half.shape == right_half.shape:
                features['face_symmetry'] = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
            else:
                features['face_symmetry'] = 0.5
                
        except Exception as e:
            print(f"Feature extraction error: {e}")
        
        return features
    
    def _calculate_gender_scores(self, features, detection_type):
        """Calculate gender scores with confidence estimation"""
        woman_score = 0
        man_score = 0
        confidence_factors = []
        
        try:
            # Hair coverage analysis
            hair_cov = features.get('hair_coverage', 0)
            if hair_cov > 0.4:  # High hair coverage
                woman_score += 3
                confidence_factors.append(0.4)
            elif hair_cov > 0.25:
                woman_score += 1
                confidence_factors.append(0.2)
            elif hair_cov < 0.1:  # Very low hair (possibly bald/very short)
                man_score += 2
                confidence_factors.append(0.3)
            
            # Face ratio analysis
            face_ratio = features.get('face_ratio', 1.0)
            if 1.1 <= face_ratio <= 1.3:  # Oval face shape
                woman_score += 1
                confidence_factors.append(0.2)
            elif face_ratio > 1.4:  # Elongated
                man_score += 1
                confidence_factors.append(0.2)
            
            # Skin smoothness
            skin_smooth = features.get('skin_smoothness', 20)
            if skin_smooth < 15:  # Very smooth
                woman_score += 2
                confidence_factors.append(0.3)
            elif skin_smooth > 30:  # Rough texture
                man_score += 2
                confidence_factors.append(0.3)
            
            # Eye analysis
            eye_brightness = features.get('eye_brightness', 128)
            eye_contrast = features.get('eye_contrast', 15)
            if eye_brightness > 140 and eye_contrast > 20:  # Bright with high contrast (makeup)
                woman_score += 2
                confidence_factors.append(0.3)
            elif eye_brightness < 100:  # Dark eyes
                man_score += 1
                confidence_factors.append(0.1)
            
            # Lip color analysis
            lip_color = features.get('lip_color', 0)
            if lip_color > 0.1:  # Red/pink lips detected
                woman_score += 3
                confidence_factors.append(0.4)
            
            # Edge density (facial hair indicator)
            edge_density = features.get('edge_density', 0)
            if edge_density > 0.15:  # High edge density in jaw area
                man_score += 2
                confidence_factors.append(0.3)
            
            # Face symmetry
            symmetry = features.get('face_symmetry', 0.5)
            if symmetry > 0.8:  # High symmetry
                woman_score += 1
                confidence_factors.append(0.1)
            
            # Adjust scores based on detection type
            if detection_type == 'profile':
                # Profile faces are less reliable, reduce confidence
                confidence_factors = [f * 0.8 for f in confidence_factors]
            elif detection_type == 'head_estimate':
                # Head estimates are least reliable
                confidence_factors = [f * 0.6 for f in confidence_factors]
            
            # Calculate final confidence
            total_confidence = min(sum(confidence_factors), 1.0)
            
            return woman_score, man_score, total_confidence
            
        except Exception as e:
            print(f"Score calculation error: {e}")
            return 0, 0, 0.0
    
    def classify_person(self, person_crop, person_id, frame_time):
        """Classify gender for a tracked person with temporal smoothing"""
        # Extract faces using multiple methods
        faces = self.extract_multiple_faces(person_crop)
        
        if not faces:
            return "Unknown"
        
        # Analyze each detected face
        gender_predictions = []
        
        for detection_type, face, coords in faces:
            gender, confidence = self.enhanced_gender_analysis(face, detection_type)
            if gender and confidence > self.min_confidence:
                gender_predictions.append((gender, confidence))
        
        # If no good predictions, return Unknown
        if not gender_predictions:
            return "Unknown"
        
        # Take the prediction with highest confidence
        best_gender, best_confidence = max(gender_predictions, key=lambda x: x[1])
        
        # Add to history for temporal smoothing
        self.gender_history[person_id].append(best_gender)
        
        # Apply temporal smoothing
        if len(self.gender_history[person_id]) >= self.stability_threshold:
            # Count occurrences
            gender_counts = {'Woman': 0, 'Man': 0, 'Unknown': 0}
            for g in self.gender_history[person_id]:
                gender_counts[g] += 1
            
            # Return most frequent gender
            stable_gender = max(gender_counts, key=gender_counts.get)
            
            # Only return if we have sufficient confidence
            if gender_counts[stable_gender] >= len(self.gender_history[person_id]) * 0.6:
                return stable_gender
        
        return best_gender
    
    def cleanup_old_trackers(self, current_time):
        """Remove trackers that haven't been seen recently"""
        to_remove = []
        for person_id, tracker_data in self.person_trackers.items():
            if current_time - tracker_data['last_seen'] > 5.0:  # 5 seconds timeout
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.person_trackers[person_id]
            if person_id in self.gender_history:
                del self.gender_history[person_id]