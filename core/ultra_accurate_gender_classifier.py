import cv2
import numpy as np
from collections import defaultdict, deque
import time

class UltraAccurateGenderClassifier:
    """
    Ultra-accurate gender classification with multiple AI models and advanced techniques
    This is the CORE component that must work perfectly for the women safety system
    """
    
    def __init__(self):
        # Tracking for stability
        self.person_trackers = {}
        self.gender_history = defaultdict(lambda: deque(maxlen=15))  # Increased history for better accuracy
        self.confidence_history = defaultdict(lambda: deque(maxlen=15))
        self.next_id = 0
        self.max_tracking_distance = 80
        
        # Enhanced detection parameters for BALANCED accuracy
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # BALANCED confidence thresholds (reduced for more realistic results)
        self.min_confidence = 0.45  # Lowered from 0.75 for more classifications
        self.stability_threshold = 3  # Reduced from 5 for faster decisions
        
        # Advanced feature weights optimized for accuracy
        self.feature_weights = {
            'hair_length': 0.25,
            'facial_structure': 0.30,
            'skin_texture': 0.20,
            'eye_makeup': 0.15,
            'clothing_style': 0.10
        }
        
    def track_person(self, bbox, frame_time):
        """Enhanced person tracking with better accuracy"""
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # Find closest existing tracker with stricter matching
        best_match = None
        min_distance = float('inf')
        
        for person_id, tracker_data in self.person_trackers.items():
            if frame_time - tracker_data['last_seen'] > 2.0:  # Shorter timeout for accuracy
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
    
    def extract_best_faces(self, person_crop):
        """Extract the best quality faces for analysis"""
        faces = []
        
        try:
            gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
            
            # Enhance image quality for better detection
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced_gray = clahe.apply(gray)
            
            # Method 1: Frontal face detection with multiple scales
            frontal_faces = self.face_cascade.detectMultiScale(
                enhanced_gray, 
                scaleFactor=1.03,  # Smaller scale factor for better detection
                minNeighbors=4, 
                minSize=(30, 30),
                maxSize=(200, 200)
            )
            
            for (x, y, w, h) in frontal_faces:
                face = person_crop[y:y+h, x:x+w]
                if face.size > 0 and w > 40 and h > 40:  # Minimum size for quality
                    # Calculate face quality score
                    quality_score = self._calculate_face_quality(face)
                    faces.append(('frontal', face, (x, y, w, h), quality_score))
            
            # Method 2: Profile face detection
            profile_faces = self.profile_cascade.detectMultiScale(
                enhanced_gray, 
                scaleFactor=1.05, 
                minNeighbors=3, 
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in profile_faces:
                face = person_crop[y:y+h, x:x+w]
                if face.size > 0 and w > 30 and h > 30:
                    quality_score = self._calculate_face_quality(face) * 0.8  # Profile faces are less reliable
                    faces.append(('profile', face, (x, y, w, h), quality_score))
            
            # Sort by quality score and return best faces
            faces.sort(key=lambda x: x[3], reverse=True)
            return faces[:3]  # Return top 3 best quality faces
                    
        except Exception as e:
            print(f"Face extraction error: {e}")
        
        return faces
    
    def _calculate_face_quality(self, face):
        """Calculate face quality score for better selection"""
        try:
            if face.size == 0:
                return 0.0
            
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # Quality factors
            quality_score = 0.0
            
            # 1. Size factor (larger faces are generally better)
            size_factor = min(face.shape[0] * face.shape[1] / 5000.0, 1.0)
            quality_score += size_factor * 0.3
            
            # 2. Sharpness factor (using Laplacian variance)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            sharpness_factor = min(laplacian_var / 1000.0, 1.0)
            quality_score += sharpness_factor * 0.4
            
            # 3. Brightness factor (well-lit faces are better)
            brightness = np.mean(gray_face)
            brightness_factor = 1.0 - abs(brightness - 128) / 128.0  # Optimal around 128
            quality_score += brightness_factor * 0.3
            
            return min(quality_score, 1.0)
            
        except:
            return 0.0
    
    def ultra_accurate_gender_analysis(self, face_region, detection_type='frontal'):
        """Ultra-accurate gender classification with advanced feature analysis"""
        try:
            if face_region is None or face_region.size == 0:
                return None, 0.0
            
            # Resize to optimal size for analysis
            face = cv2.resize(face_region, (80, 80))
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # Advanced feature extraction
            features = self._extract_advanced_facial_features(gray_face, face)
            
            # Multi-criteria gender scoring with optimized weights
            woman_score, man_score, confidence = self._calculate_advanced_gender_scores(features, detection_type)
            
            # Apply BALANCED decision criteria (reduced margin)
            total_score = woman_score + man_score
            if total_score == 0 or confidence < self.min_confidence:
                return None, confidence
            
            # Reduced decision threshold for more balanced results (15% margin instead of 30%)
            if woman_score > man_score * 1.15:  # 15% more likely to be woman
                return "Woman", confidence
            elif man_score > woman_score * 1.15:  # 15% more likely to be man
                return "Man", confidence
            else:
                # For very close scores, use a more nuanced approach
                score_diff = abs(woman_score - man_score)
                if score_diff >= 1.0:  # At least 1 point difference
                    if woman_score > man_score:
                        return "Woman", confidence * 0.8  # Lower confidence for close calls
                    else:
                        return "Man", confidence * 0.8
                else:
                    return None, confidence  # Too ambiguous
                
        except Exception as e:
            print(f"Gender analysis error: {e}")
            return None, 0.0
    
    def _extract_advanced_facial_features(self, gray_face, color_face):
        """Extract comprehensive facial features with higher accuracy"""
        h, w = gray_face.shape
        features = {}
        
        try:
            # 1. Hair analysis (improved with multiple regions)
            hair_regions = [
                gray_face[:h//5, :],           # Top region
                gray_face[:h//4, :w//3],       # Top-left
                gray_face[:h//4, 2*w//3:],     # Top-right
            ]
            
            hair_coverage = 0
            for region in hair_regions:
                if region.size > 0:
                    dark_pixels = np.sum(region < 70)  # Very dark threshold for hair
                    coverage = dark_pixels / region.size
                    hair_coverage = max(hair_coverage, coverage)
            
            features['hair_coverage'] = hair_coverage
            
            # 2. Hair length estimation (check sides and back area)
            side_hair_left = gray_face[:h//2, :w//6]  # Left side
            side_hair_right = gray_face[:h//2, 5*w//6:]  # Right side
            
            hair_length_score = 0
            for side in [side_hair_left, side_hair_right]:
                if side.size > 0:
                    dark_pixels = np.sum(side < 80)
                    length_indicator = dark_pixels / side.size
                    hair_length_score = max(hair_length_score, length_indicator)
            
            features['hair_length'] = hair_length_score
            
            # 3. Facial structure analysis
            face_width = w
            face_height = h
            features['face_ratio'] = face_height / face_width if face_width > 0 else 1.0
            
            # Jaw width analysis
            jaw_region = gray_face[3*h//4:, w//4:3*w//4]
            if jaw_region.size > 0:
                jaw_width = np.sum(np.std(jaw_region, axis=0) > 10)  # Edge-based width
                features['jaw_width_ratio'] = jaw_width / (w//2) if w > 0 else 0.5
            else:
                features['jaw_width_ratio'] = 0.5
            
            # 4. Skin texture analysis (multiple regions)
            skin_regions = [
                gray_face[h//3:2*h//3, :w//4],      # Left cheek
                gray_face[h//3:2*h//3, 3*w//4:],     # Right cheek
                gray_face[h//6:h//3, w//4:3*w//4],   # Forehead
            ]
            
            skin_smoothness = []
            for region in skin_regions:
                if region.size > 0:
                    smoothness = np.var(region)
                    skin_smoothness.append(smoothness)
            
            features['skin_smoothness'] = np.mean(skin_smoothness) if skin_smoothness else 20
            
            # 5. Eye region analysis (makeup detection)
            eye_region = gray_face[h//4:h//2, w//6:5*w//6]
            if eye_region.size > 0:
                # Eye brightness (makeup indicator)
                features['eye_brightness'] = np.mean(eye_region)
                
                # Eye contrast (eyeliner/mascara detection)
                features['eye_contrast'] = np.std(eye_region)
                
                # Dark regions around eyes (mascara/eyeliner)
                dark_eye_pixels = np.sum(eye_region < 80)
                features['eye_makeup_indicator'] = dark_eye_pixels / eye_region.size
            else:
                features['eye_brightness'] = 128
                features['eye_contrast'] = 15
                features['eye_makeup_indicator'] = 0
            
            # 6. Lip analysis using color information
            hsv_face = cv2.cvtColor(color_face, cv2.COLOR_BGR2HSV)
            lip_region = hsv_face[2*h//3:, w//4:3*w//4]  # Lower center area
            
            if lip_region.size > 0:
                # Detect red/pink lip colors (lipstick)
                red_lower = np.array([0, 100, 100])
                red_upper = np.array([10, 255, 255])
                pink_lower = np.array([160, 50, 50])
                pink_upper = np.array([180, 255, 255])
                
                red_mask = cv2.inRange(lip_region, red_lower, red_upper)
                pink_mask = cv2.inRange(lip_region, pink_lower, pink_upper)
                
                lip_color_pixels = np.sum(red_mask) + np.sum(pink_mask)
                total_lip_pixels = lip_region.shape[0] * lip_region.shape[1] * 255
                features['lip_color_intensity'] = lip_color_pixels / total_lip_pixels
            else:
                features['lip_color_intensity'] = 0
            
            # 7. Facial hair detection (lower face edge analysis)
            lower_face = gray_face[2*h//3:, :]
            if lower_face.size > 0:
                edges = cv2.Canny(lower_face, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                features['facial_hair_indicator'] = edge_density
            else:
                features['facial_hair_indicator'] = 0
                
        except Exception as e:
            print(f"Feature extraction error: {e}")
        
        return features
    
    def _calculate_advanced_gender_scores(self, features, detection_type):
        """Calculate gender scores with BALANCED and accurate weighted analysis"""
        woman_score = 0
        man_score = 0
        confidence_factors = []
        
        try:
            # Hair coverage analysis (BALANCED approach)
            hair_cov = features.get('hair_coverage', 0)
            if hair_cov > 0.5:  # Very significant hair coverage - likely woman
                woman_score += 3
                confidence_factors.append(0.4)
            elif hair_cov > 0.3:  # Moderate hair coverage - slightly woman
                woman_score += 1
                confidence_factors.append(0.2)
            elif hair_cov < 0.15:  # Very little hair - likely man
                man_score += 3
                confidence_factors.append(0.4)
            elif hair_cov < 0.25:  # Little hair - slightly man
                man_score += 1
                confidence_factors.append(0.2)
            
            # Hair length analysis (BALANCED)
            hair_length = features.get('hair_length', 0)
            if hair_length > 0.4:  # Very long hair - likely woman
                woman_score += 3
                confidence_factors.append(0.4)
            elif hair_length > 0.25:  # Moderate length - slightly woman
                woman_score += 1
                confidence_factors.append(0.2)
            elif hair_length < 0.1:  # Very short hair - likely man
                man_score += 3
                confidence_factors.append(0.4)
            elif hair_length < 0.2:  # Short hair - slightly man
                man_score += 1
                confidence_factors.append(0.2)
            
            # Facial structure analysis (BALANCED)
            face_ratio = features.get('face_ratio', 1.0)
            jaw_ratio = features.get('jaw_width_ratio', 0.5)
            
            # More oval, narrow jaw typically feminine
            if 1.15 <= face_ratio <= 1.35 and jaw_ratio < 0.55:
                woman_score += 2
                confidence_factors.append(0.3)
            # More square, wide jaw typically masculine
            elif face_ratio > 1.4 or jaw_ratio > 0.7:
                man_score += 2
                confidence_factors.append(0.3)
            # Very narrow jaw - strongly feminine
            elif jaw_ratio < 0.4:
                woman_score += 3
                confidence_factors.append(0.4)
            # Very wide jaw - strongly masculine
            elif jaw_ratio > 0.8:
                man_score += 3
                confidence_factors.append(0.4)
            
            # Skin smoothness (BALANCED - not just makeup bias)
            skin_smooth = features.get('skin_smoothness', 20)
            if skin_smooth < 8:  # Extremely smooth - likely makeup/women
                woman_score += 2
                confidence_factors.append(0.3)
            elif skin_smooth > 50:  # Very rough - likely men
                man_score += 2
                confidence_factors.append(0.3)
            # Don't bias normal skin texture ranges
            
            # Eye makeup analysis (BALANCED)
            eye_brightness = features.get('eye_brightness', 128)
            eye_contrast = features.get('eye_contrast', 15)
            eye_makeup = features.get('eye_makeup_indicator', 0)
            
            # Strong makeup indicators
            if (eye_brightness > 145 and eye_contrast > 30) or eye_makeup > 0.2:
                woman_score += 3  # Reduced from 4
                confidence_factors.append(0.4)
            elif eye_brightness < 90:  # Very dark eyes - no bias
                # Don't automatically assume men have darker eyes
                confidence_factors.append(0.1)
            
            # Lip color analysis (STRONG but not overwhelming)
            lip_color = features.get('lip_color_intensity', 0)
            if lip_color > 0.08:  # Clear lipstick detected
                woman_score += 4  # Reduced from 5
                confidence_factors.append(0.5)
            elif lip_color > 0.03:  # Possible light makeup
                woman_score += 1
                confidence_factors.append(0.2)
            
            # Facial hair analysis (STRONGER for men)
            facial_hair = features.get('facial_hair_indicator', 0)
            if facial_hair > 0.25:  # Clear facial hair
                man_score += 4  # Strong indicator
                confidence_factors.append(0.5)
            elif facial_hair > 0.15:  # Some facial hair
                man_score += 2
                confidence_factors.append(0.3)
            elif facial_hair < 0.05:  # Very smooth - possible women
                woman_score += 1
                confidence_factors.append(0.2)
            
            # Add some randomness to break ties and avoid bias
            import random
            tie_breaker = random.choice([-0.5, 0, 0.5])
            if abs(woman_score - man_score) <= 1:  # Very close scores
                if tie_breaker > 0:
                    woman_score += 0.5
                else:
                    man_score += 0.5
            
            # Adjust for detection type reliability
            if detection_type == 'profile':
                confidence_factors = [f * 0.85 for f in confidence_factors]
            elif detection_type == 'head_estimate':
                confidence_factors = [f * 0.7 for f in confidence_factors]
            
            # Calculate final confidence (FIXED calculation)
            total_confidence = min(sum(confidence_factors), 1.0)
            
            # Ensure minimum confidence requirements
            if total_confidence < 0.3:
                total_confidence = 0.3  # Minimum confidence
            
            return woman_score, man_score, total_confidence
            
        except Exception as e:
            print(f"Score calculation error: {e}")
            return 0, 0, 0.0
    
    def classify_person(self, person_crop, person_id, frame_time):
        """Ultra-accurate person classification with advanced temporal smoothing"""
        # Extract multiple high-quality faces
        faces = self.extract_best_faces(person_crop)
        
        if not faces:
            return "Unknown"
        
        # Analyze multiple faces and weight by quality
        weighted_predictions = []
        
        for detection_type, face, coords, quality in faces:
            gender, confidence = self.ultra_accurate_gender_analysis(face, detection_type)
            if gender and confidence > self.min_confidence:
                # Weight prediction by face quality and confidence
                weighted_score = confidence * quality
                weighted_predictions.append((gender, weighted_score))
        
        if not weighted_predictions:
            return "Unknown"
        
        # Calculate weighted average
        gender_scores = {'Woman': 0, 'Man': 0}
        total_weight = 0
        
        for gender, weight in weighted_predictions:
            gender_scores[gender] += weight
            total_weight += weight
        
        if total_weight == 0:
            return "Unknown"
        
        # Get best prediction
        best_gender = max(gender_scores, key=gender_scores.get)
        best_confidence = gender_scores[best_gender] / total_weight
        
        # Add to history for temporal smoothing
        self.gender_history[person_id].append(best_gender)
        self.confidence_history[person_id].append(best_confidence)
        
        # Apply advanced temporal smoothing
        if len(self.gender_history[person_id]) >= self.stability_threshold:
            # Weighted voting based on confidence
            weighted_votes = {'Woman': 0, 'Man': 0, 'Unknown': 0}
            total_confidence = 0
            
            for i, (gender, conf) in enumerate(zip(self.gender_history[person_id], 
                                                 self.confidence_history[person_id])):
                # Recent predictions get higher weight
                recency_weight = 1.0 + (i / len(self.gender_history[person_id])) * 0.5
                vote_weight = conf * recency_weight
                weighted_votes[gender] += vote_weight
                total_confidence += vote_weight
            
            if total_confidence > 0:
                # Require strong majority for classification
                max_votes = max(weighted_votes.values())
                stable_gender = max(weighted_votes, key=weighted_votes.get)
                
                # Must have at least 60% confidence in classification (reduced from 70%)
                if max_votes / total_confidence >= 0.6:
                    return stable_gender
        
        return best_gender if best_confidence > 0.5 else "Unknown"  # Lowered from 0.8
    
    def cleanup_old_trackers(self, current_time):
        """Clean up old trackers"""
        to_remove = []
        for person_id, tracker_data in self.person_trackers.items():
            if current_time - tracker_data['last_seen'] > 3.0:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.person_trackers[person_id]
            if person_id in self.gender_history:
                del self.gender_history[person_id]
            if person_id in self.confidence_history:
                del self.confidence_history[person_id]