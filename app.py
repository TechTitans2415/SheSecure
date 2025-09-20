from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import time
import os
import numpy as np
from core.improved_detection import improved_detect_and_classify
from core.simple_gesture_detection import SimpleGestureDetector
from core.simple_gender_classifier import SimpleGenderClassifier
from core.harassment_detection import HarassmentKidnapDetector
from services.alerts import get_active_alerts, get_alert_history, acknowledge_alert, clear_alert, get_alert_stats, create_alert
from services.emergency_service import EmergencyService
from core.sos_logic import get_system_status

app = Flask(__name__)
camera = cv2.VideoCapture(0)  # Webcam feed

# Initialize advanced detection systems
gesture_detector = SimpleGestureDetector()
gender_classifier = SimpleGenderClassifier()
harassment_detector = HarassmentKidnapDetector()
emergency_service = EmergencyService()  # New emergency service

# Global detection state
detection_state = {
    'total_detections': 0,
    'women_detections': 0,
    'men_detections': 0,
    'last_detection_time': time.time(),
    'emergency_active': False
}

def gen_frames():
    global detection_state
    
    while True:
        success, frame = camera.read()
        if not success:
            break
            
        # Core person detection and gender classification with improved stability
        frame, genders, detected_people = improved_detect_and_classify(frame, gender_classifier)
        
        # Update detection statistics
        detection_state['total_detections'] = len(genders)
        detection_state['men_detections'] = genders.count('Man')
        detection_state['women_detections'] = genders.count('Woman')
        detection_state['last_detection_time'] = time.time()
        
        # Advanced threat detection if people are detected
        if len(genders) > 0:
            # SOS Gesture Detection with Emergency Response
            gesture_results = gesture_detector.detect_sos_gestures(frame)
            if gesture_results['sos_detected']:
                # Create standard alert
                alert_data = {
                    'type': 'SOS_GESTURE',
                    'message': f"SOS gesture detected: {gesture_results['gesture_type']} (Confidence: {gesture_results['confidence']:.2f})",
                    'severity': 'CRITICAL',
                    'timestamp': time.time(),
                    'confidence': gesture_results['confidence'],
                    'gesture_type': gesture_results['gesture_type']
                }
                create_alert(alert_data)
                detection_state['emergency_active'] = True
                
                # 🚨 NEW: Trigger emergency response with snapshot and notifications
                emergency_response = emergency_service.handle_emergency(
                    frame=frame,
                    gesture_type=gesture_results['gesture_type'],
                    confidence=gesture_results['confidence'],
                    location_info="Camera 1 - Main Surveillance Area"
                )
                
                if emergency_response['status'] == 'success':
                    print(f"✅ Emergency response activated: {emergency_response['message']}")
                    print(f"📸 Snapshot saved: {emergency_response['snapshot_path']}")
                    print(f"🆔 Incident ID: {emergency_response['incident_id']}")
                else:
                    print(f"❌ Emergency response failed: {emergency_response.get('error', 'Unknown error')}")
                
                # Add enhanced visual indicator on frame
                cv2.putText(frame, f"🚨 EMERGENCY ALERT 🚨", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(frame, f"SOS: {gesture_results['gesture_type'].upper()}", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, f"Confidence: {gesture_results['confidence']:.2f}", 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if emergency_response['status'] == 'success':
                    cv2.putText(frame, f"📸 Snapshot captured | 📧 Authorities notified", 
                              (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # Show gesture detection status
                details = gesture_results['details']
                status_text = f"Gesture Monitoring: Motion={details['motion_detected']}, Activity={details['upper_body_activity']:.2f}"
                cv2.putText(frame, status_text, (10, frame.shape[0] - 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Harassment/Kidnap Detection using actual detection data
            harassment_results = harassment_detector.analyze_interactions(frame, detected_people)
            
            if harassment_results['scenario'] != 'NONE':
                alert_data = {
                    'type': harassment_results['scenario'],
                    'message': f"Harassment scenario detected: {harassment_results['scenario']} (Confidence: {harassment_results['confidence']:.2f})",
                    'severity': 'CRITICAL' if harassment_results['confidence'] > 0.8 else 'HIGH',
                    'timestamp': time.time(),
                    'confidence': harassment_results['confidence']
                }
                create_alert(alert_data)
                detection_state['emergency_active'] = True
                
                # Add visual indicator on frame
                cv2.putText(frame, f"THREAT: {harassment_results['scenario']}", 
                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# API endpoints for alerts
@app.route('/api/alerts/active')
def api_active_alerts():
    """Get active alerts"""
    try:
        alerts = get_active_alerts()
        return jsonify({'success': True, 'alerts': alerts})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/alerts/history')
def api_alert_history():
    """Get alert history"""
    try:
        limit = request.args.get('limit', 50, type=int)
        history = get_alert_history(limit)
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/alerts/stats')
def api_alert_stats():
    """Get alert statistics"""
    try:
        stats = get_alert_stats()
        
        # Add live detection stats
        stats.update({
            'total_detections': detection_state['total_detections'],
            'women_detections': detection_state['women_detections'],
            'men_detections': detection_state['men_detections'],
            'emergency_active': detection_state['emergency_active'],
            'last_detection_time': detection_state['last_detection_time']
        })
        
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/alerts/acknowledge', methods=['POST'])
def api_acknowledge_alert():
    """Acknowledge an alert"""
    try:
        data = request.get_json()
        alert_id = data.get('alert_id')
        if not alert_id:
            return jsonify({'success': False, 'error': 'alert_id required'})
        
        acknowledge_alert(alert_id)
        return jsonify({'success': True, 'message': 'Alert acknowledged'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/alerts/clear', methods=['POST'])
def api_clear_alert():
    """Clear an alert"""
    try:
        data = request.get_json()
        alert_id = data.get('alert_id')
        if not alert_id:
            return jsonify({'success': False, 'error': 'alert_id required'})
        
        clear_alert(alert_id)
        return jsonify({'success': True, 'message': 'Alert cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/emergency/incidents')
def api_emergency_incidents():
    """Get recent emergency incidents with snapshots"""
    try:
        incidents = emergency_service.get_recent_incidents(limit=20)
        return jsonify({'success': True, 'incidents': incidents})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/emergency/snapshot/<incident_id>')
def api_emergency_snapshot(incident_id):
    """Get emergency snapshot by incident ID"""
    try:
        snapshot_path = f"incidents/{incident_id}_snapshot.jpg"
        if os.path.exists(snapshot_path):
            return send_file(snapshot_path, mimetype='image/jpeg')
        else:
            return jsonify({'success': False, 'error': 'Snapshot not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/system/status')
def api_system_status():
    """Get system monitoring status"""
    try:
        status = get_system_status()
        return jsonify({'success': True, 'status': status})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/emergency/manual', methods=['POST'])
def api_manual_emergency():
    """Trigger manual emergency alert"""
    try:
        data = request.get_json()
        alert_data = {
            'type': data.get('type', 'MANUAL_EMERGENCY'),
            'message': data.get('message', 'Manual emergency button activated'),
            'severity': data.get('severity', 'CRITICAL'),
            'timestamp': time.time(),
            'location': 'Manual trigger from UI'
        }
        
        # Create alert in system
        alert_id = create_alert(alert_data)
        
        # Update detection state
        detection_state['emergency_active'] = True
        
        return jsonify({
            'success': True, 
            'message': 'Manual emergency alert activated',
            'alert_id': alert_id
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    import time
    print("🚀 Starting Women Safety CCTV System...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("🎥 Video stream at: http://localhost:5000/video_feed")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    app.run(debug=True, host='127.0.0.1', port=5000)
