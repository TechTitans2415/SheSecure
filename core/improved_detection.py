import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load YOLOv8 model for person detection
yolo = YOLO("models/yolov8n.pt")

def improved_detect_and_classify(frame, gender_classifier):
    """
    ULTRA-ACCURATE detection with advanced gender classification
    This is the CORE function that MUST provide accurate gender detection
    """
    results = yolo(frame)
    genders = []
    current_time = time.time()
    
    # Clean up old trackers
    gender_classifier.cleanup_old_trackers(current_time)
    
    detected_people = []
    
    for r in results:
        for box in r.boxes:
            # Only process person class (class 0 in COCO)
            if int(box.cls[0]) != 0:  # 0 = person class
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
            # Skip low confidence detections
            if confidence < 0.4:  # Lowered threshold for better detection
                continue
            
            # Ensure valid bounding box
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract person region with bounds checking
            height, width = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                continue
            
            # Track this person and get stable gender classification
            person_id = gender_classifier.track_person([x1, y1, x2, y2], current_time)
            gender = gender_classifier.classify_person(person_crop, person_id, current_time)
            
            # Draw bounding box with gender label
            color = (0, 255, 0)  # Green for successful detection
            if gender == "Woman":
                color = (255, 0, 255)  # Magenta for women
            elif gender == "Man":
                color = (0, 0, 255)  # Blue for men
            elif gender == "Unknown":
                color = (128, 128, 128)  # Gray for unknown
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label with confidence
            label = f"{gender} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Store detection results
            genders.append(gender)
            detected_people.append({
                'bbox': [x1, y1, x2, y2],
                'gender': gender,
                'confidence': confidence,
                'person_id': person_id
            })
    
    return frame, genders, detected_people