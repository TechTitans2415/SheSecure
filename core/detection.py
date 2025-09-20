import cv2
import numpy as np
from ultralytics import YOLO
from .sos_logic import analyze_anomalies
import sys
import os

# Add parent directory to path for importing services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.emergency_response import trigger_emergency_response

# Load YOLOv8 model for person detection
yolo = YOLO("models/yolov8n.pt")

# Initialize DeepFace availability flag
deepface_available = False
DeepFace = None

def try_load_deepface():
    """Try to load DeepFace, return True if successful"""
    global deepface_available, DeepFace
    if not deepface_available and DeepFace is None:
        try:
            from deepface import DeepFace as DF
            DeepFace = DF
            deepface_available = True
            print("DeepFace loaded successfully")
        except Exception as e:
            print(f"DeepFace not available: {e}")
            deepface_available = False
    return deepface_available

def extract_face_from_person(person_crop):
    """Extract face region from person detection using OpenCV face detection"""
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        
        # Use OpenCV's built-in face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            # Return the largest face detected
            largest_face = max(faces, key=lambda x: x[2] * x[3])  # x, y, w, h
            fx, fy, fw, fh = largest_face
            face = person_crop[fy:fy+fh, fx:fx+fw]
            return face, (fx, fy, fw, fh)
    except Exception as e:
        print(f"Face extraction error: {e}")
    
    return None, None

def simple_gender_heuristic(face_region):
    """Enhanced gender classification with multiple detection methods and confidence scoring"""
    try:
        if face_region is None or face_region.size == 0:
            return "Unknown"
        
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        height, width = gray_face.shape
        
        if height < 20 or width < 20:  # Too small to analyze reliably
            return "Unknown"
        
        # Enhanced feature analysis with better thresholds
        features = {}
        
        # 1. Hair analysis (top 20% of face for better hair detection)
        hair_region = gray_face[:height//5, :]
        dark_hair_pixels = np.sum(hair_region < 100)  # Darker threshold for hair
        hair_coverage = dark_hair_pixels / hair_region.size if hair_region.size > 0 else 0
        features['hair_coverage'] = hair_coverage
        
        # 2. Face shape ratio (height/width - women often have more oval faces)
        face_ratio = height / width if width > 0 else 1.0
        features['face_ratio'] = face_ratio
        
        # 3. Skin texture analysis (cheek areas for smoothness)
        cheek_left = gray_face[height//3:2*height//3, :width//4]
        cheek_right = gray_face[height//3:2*height//3, 3*width//4:]
        skin_smoothness = 0
        if cheek_left.size > 0 and cheek_right.size > 0:
            skin_smoothness = (np.std(cheek_left) + np.std(cheek_right)) / 2
        features['skin_smoothness'] = skin_smoothness
        
        # 4. Eye region analysis (brighter often indicates makeup/women)
        eye_region = gray_face[height//4:height//2, width//4:3*width//4]
        eye_brightness = np.mean(eye_region) if eye_region.size > 0 else 128
        features['eye_brightness'] = eye_brightness
        
        # 5. Forehead analysis
        forehead = gray_face[height//8:height//3, width//4:3*width//4]
        forehead_smoothness = np.std(forehead) if forehead.size > 0 else 15
        features['forehead_smoothness'] = forehead_smoothness
        
        # Enhanced scoring system with confidence tracking
        woman_score = 0
        man_score = 0
        confidence_factors = []
        
        # Hair coverage (women typically have more visible hair in frame)
        if hair_coverage > 0.35:
            woman_score += 3
            confidence_factors.append(0.4)
        elif hair_coverage > 0.25:
            woman_score += 1
            confidence_factors.append(0.2)
        elif hair_coverage < 0.15:
            man_score += 2
            confidence_factors.append(0.3)
            
        # Face shape ratio (oval faces more common in women)
        if 1.15 <= face_ratio <= 1.35:  # Optimal oval range
            woman_score += 2
            confidence_factors.append(0.3)
        elif face_ratio > 1.45:  # Very elongated
            man_score += 1
            confidence_factors.append(0.2)
        elif face_ratio < 1.1:  # Very wide
            man_score += 1
            confidence_factors.append(0.2)
            
        # Skin smoothness (makeup and natural skin differences)
        if skin_smoothness < 18:  # Very smooth
            woman_score += 2
            confidence_factors.append(0.3)
        elif skin_smoothness < 25:  # Moderately smooth
            woman_score += 1
            confidence_factors.append(0.2)
        elif skin_smoothness > 35:  # Rough texture
            man_score += 1
            confidence_factors.append(0.2)
            
        # Eye brightness (makeup/eye features)
        if eye_brightness > 130:  # Very bright (likely makeup)
            woman_score += 2
            confidence_factors.append(0.2)
        elif eye_brightness > 115:
            woman_score += 1
            confidence_factors.append(0.1)
        elif eye_brightness < 85:  # Very dark
            man_score += 1
            confidence_factors.append(0.1)
            
        # Forehead smoothness
        if forehead_smoothness < 12:  # Very smooth forehead
            woman_score += 1
            confidence_factors.append(0.1)
        elif forehead_smoothness > 25:  # Rough forehead
            man_score += 1
            confidence_factors.append(0.1)
        
        # Calculate total confidence
        total_confidence = sum(confidence_factors)
        
        # Decision making with confidence threshold
        if total_confidence < 0.4:  # Low confidence - return Unknown
            return "Unknown"
        elif woman_score > man_score + 1:  # Clear woman indication
            return "Woman"
        elif man_score > woman_score + 1:  # Clear man indication
            return "Man"
        elif woman_score >= man_score:  # Tie or slight woman advantage
            # For safety system, slightly bias toward woman detection
            return "Woman"
        else:
            return "Man"
            
    except Exception as e:
        print(f"Gender heuristic error: {e}")
        return "Unknown"

def detect_and_classify(frame):
    """Detect people and classify gender"""
    results = yolo(frame)
    genders = []

    for r in results:
        for box in r.boxes:
            # Only process person class (class 0 in COCO)
            if int(box.cls[0]) != 0:  # 0 = person class
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
            # Skip low confidence detections
            if confidence < 0.5:
                continue
            
            # Extract person region
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                continue

            # Extract face from person detection
            face, face_coords = extract_face_from_person(person_crop)
            
            gender = "Unknown"
            if face is not None and face.size > 0:
                # Try to load DeepFace on first use
                if try_load_deepface():
                    try:
                        # Resize face if too small but keep reasonable size
                        if face.shape[0] < 64 or face.shape[1] < 64:
                            face = cv2.resize(face, (64, 64))
                        
                        # Improve face quality
                        face = cv2.GaussianBlur(face, (3, 3), 0)  # Slight blur to reduce noise
                        
                        analysis = DeepFace.analyze(
                            face,
                            actions=['gender'],
                            detector_backend='opencv',
                            enforce_detection=False,
                            silent=True  # Reduce console output
                        )
                        
                        if isinstance(analysis, list):
                            analysis = analysis[0]
                        
                        # Handle different possible gender formats
                        if 'dominant_gender' in analysis:
                            gender = analysis['dominant_gender']
                            if 'gender' in analysis and gender in analysis['gender']:
                                conf = analysis['gender'][gender]
                            else:
                                conf = 0.8  # Default confidence
                        elif 'gender' in analysis:
                            # Handle alternative format
                            gender_data = analysis['gender']
                            if isinstance(gender_data, dict):
                                gender = max(gender_data, key=gender_data.get)
                                conf = gender_data[gender]
                            else:
                                gender = str(gender_data)
                                conf = 0.7
                        
                        # Normalize gender names
                        if gender.lower() in ['woman', 'female']:
                            gender = "Woman"
                        elif gender.lower() in ['man', 'male']:
                            gender = "Man"
                        
                        # Apply confidence threshold - lowered for better detection
                        if conf < 0.55:  # Lower threshold for more detections
                            gender = simple_gender_heuristic(face)
                            
                    except Exception as e:
                        print(f"DeepFace analysis error: {e}")
                        # Always fallback to heuristic instead of unknown
                        gender = simple_gender_heuristic(face)
                else:
                    # Use heuristic when DeepFace is not available
                    gender = simple_gender_heuristic(face)
            else:
                # No face detected, but still try to classify from person region
                gender = simple_gender_heuristic(person_crop)

            genders.append((gender, (x1, y1, x2, y2)))

            # Draw bounding box
            if gender == "Man":
                color = (255, 0, 0)  # Blue for men
            elif gender == "Woman":
                color = (255, 20, 147)  # Deep pink for women
            else:
                color = (128, 128, 128)  # Gray for unknown
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{gender} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Analyze for anomalies using SOS logic
    try:
        anomaly_result = analyze_anomalies(genders)
        
        # Draw alerts on frame if any
        if anomaly_result['alerts']:
            alert_y = 50
            for alert in anomaly_result['alerts']:
                # Enhanced visual feedback for emergency situations
                if alert['severity'] == 'CRITICAL':
                    alert_color = (0, 0, 255)  # Red
                    bg_color = (0, 0, 139)     # Dark red background
                    # Add flashing border for critical alerts
                    import time
                    if int(time.time() * 4) % 2:  # Flash every 0.25 seconds
                        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 12)
                elif alert['severity'] == 'HIGH':
                    alert_color = (0, 165, 255)  # Orange
                    bg_color = (0, 100, 139)     # Dark orange background
                else:
                    alert_color = (0, 255, 255)  # Yellow
                    bg_color = (0, 139, 139)     # Dark yellow background
                
                # Create prominent alert display
                alert_text = f"🚨 {alert['severity']} ALERT: {alert['type']}"
                message_text = alert['message']
                
                # Get text dimensions for background
                (text_width, text_height), baseline = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                (msg_width, msg_height), _ = cv2.getTextSize(message_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw background rectangle for alert
                cv2.rectangle(frame, (10, alert_y - text_height - 10), 
                             (max(text_width, msg_width) + 20, alert_y + msg_height + 20), bg_color, -1)
                cv2.rectangle(frame, (10, alert_y - text_height - 10), 
                             (max(text_width, msg_width) + 20, alert_y + msg_height + 20), alert_color, 3)
                
                # Draw alert text with better visibility
                cv2.putText(frame, alert_text, (15, alert_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, alert_color, 2)
                cv2.putText(frame, message_text, (15, alert_y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                alert_y += 80  # More space between alerts
                
                # 🚨 IMMEDIATE EMERGENCY RESPONSE FOR CRITICAL/HIGH ALERTS 🚨
                if alert['severity'] in ['CRITICAL', 'HIGH']:
                    try:
                        print(f"\n🚨 TRIGGERING IMMEDIATE EMERGENCY RESPONSE FOR {alert['type']} 🚨")
                        emergency_report = trigger_emergency_response(alert, frame.copy())
                        if emergency_report:
                            # Enhanced emergency visual indicators
                            # Red pulsing border
                            import time
                            border_width = 15 if int(time.time() * 6) % 2 else 8
                            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), border_width)
                            
                            # Large emergency text
                            cv2.putText(frame, "🚨 EMERGENCY RESPONSE ACTIVATED 🚨", (30, frame.shape[0] - 70),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                            cv2.putText(frame, f"Emergency ID: {emergency_report['emergency_id']}", (30, frame.shape[0] - 40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(frame, "AUTHORITIES NOTIFIED", (30, frame.shape[0] - 15),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            
                            # Add timestamp
                            import datetime
                            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                            cv2.putText(frame, f"Time: {timestamp}", (frame.shape[1] - 200, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    except Exception as e:
                        print(f"Emergency response error: {e}")
                
        # Add stats display
        stats = anomaly_result['stats']
        cv2.putText(frame, f"Men: {stats['men']} | Women: {stats['women']} | Total: {stats['total']}", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                   
    except Exception as e:
        print(f"Anomaly detection error: {e}")
        # Continue without anomaly detection if there's an error

    return frame, genders
