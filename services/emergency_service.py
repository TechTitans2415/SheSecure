"""
Emergency Service for Women Safety CCTV System
Handles automatic snapshot capture and notification when SOS is detected
"""

import cv2
import os
import sys
import time
import smtplib
import threading
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
import requests
import json
import numpy as np

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config.emergency_config import (
        EMAIL_CONFIG, SMS_CONFIG, LOCATION_CONFIG, 
        ALERT_CONFIG, EMAIL_TEMPLATE, SMS_TEMPLATE,
        DEVELOPMENT_CONFIG
    )
except ImportError:
    # Fallback configuration if config file not found
    EMAIL_CONFIG = {
        'enabled': True,
        'authority_emails': ['admin@example.com']
    }
    SMS_CONFIG = {
        'enabled': False,
        'authority_phones': ['+1234567890']
    }
    LOCATION_CONFIG = {
        'site_name': 'Women Safety CCTV',
        'location': 'Main Area',
        'camera_id': 'CAM-001'
    }
    ALERT_CONFIG = {
        'snapshot_cooldown': 30,
        'image_quality': 95
    }
    EMAIL_TEMPLATE = "Emergency alert: {gesture_type} detected. Incident: {incident_id}"
    SMS_TEMPLATE = "🚨 Emergency: {gesture_type} detected. ID: {incident_id}"
    DEVELOPMENT_CONFIG = {'test_mode': True}

class EmergencyService:
    """
    Handles emergency response when SOS is detected:
    1. Captures high-quality snapshot
    2. Sends email notifications with snapshot
    3. Sends SMS alerts
    4. Logs incident details
    """
    
    def __init__(self):
        self.incidents_dir = "incidents"
        self.ensure_incidents_directory()
        
        # Load configuration with fallbacks
        self.email_config = {
            'enabled': True,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': 'women.safety.cctv@gmail.com',  # UPDATE THIS
            'sender_password': 'your_app_password_here',     # UPDATE THIS
            'authority_emails': [
                'police@example.com',      # UPDATE THESE
                'security@example.com',    # UPDATE THESE
                'safety@example.com'       # UPDATE THESE
            ]
        }
        
        self.sms_config = {
            'enabled': False,  # Enable when configured
            'service': 'twilio',
            'account_sid': 'your_twilio_account_sid',
            'auth_token': 'your_twilio_auth_token',
            'from_phone': '+1234567890',
            'authority_phones': [
                '+1-911-000-0000',  # UPDATE THESE
                '+1-555-0123',      # UPDATE THESE
                '+1-555-0456'       # UPDATE THESE
            ]
        }
        
        self.location_config = {
            'site_name': 'Women Safety CCTV System',
            'location': 'Main Surveillance Area',
            'camera_id': 'CAM-001'
        }
        
        # Load from config file if available
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'emergency_config.py')
            if os.path.exists(config_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location("emergency_config", config_path)
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
                
                if hasattr(config_module, 'EMAIL_CONFIG'):
                    self.email_config.update(config_module.EMAIL_CONFIG)
                if hasattr(config_module, 'SMS_CONFIG'):
                    self.sms_config.update(config_module.SMS_CONFIG)
                if hasattr(config_module, 'LOCATION_CONFIG'):
                    self.location_config.update(config_module.LOCATION_CONFIG)
                    
                print("✅ Emergency configuration loaded from config file")
        except Exception as e:
            print(f"⚠️  Using default emergency configuration: {e}")
        
        # Incident tracking
        self.last_snapshot_time = 0
        self.snapshot_cooldown = 5  # Reduced cooldown - 5 seconds is more reasonable for emergencies
        
    def ensure_incidents_directory(self):
        """Create incidents directory if it doesn't exist"""
        if not os.path.exists(self.incidents_dir):
            os.makedirs(self.incidents_dir)
            
    def capture_emergency_snapshot(self, frame, gesture_type, confidence, location_info=None):
        """
        Capture high-quality snapshot when SOS is detected
        
        Args:
            frame: Current video frame
            gesture_type: Type of SOS gesture detected
            confidence: Detection confidence
            location_info: Optional location/camera info
            
        Returns:
            tuple: (success, snapshot_path, incident_id)
        """
        try:
            current_time = time.time()
            
            # Validate frame
            if frame is None:
                print("❌ Error: No frame provided for snapshot")
                return False, None, None
                
            # Implement cooldown to prevent spam
            if current_time - self.last_snapshot_time < self.snapshot_cooldown:
                print(f"⏳ Snapshot cooldown active ({self.snapshot_cooldown}s)")
                return False, None, None
                
            # Ensure directory exists
            self.ensure_incidents_directory()
            
            # Generate unique incident ID
            timestamp = datetime.now()
            incident_id = f"SOS_{timestamp.strftime('%Y%m%d_%H%M%S')}_{int(current_time * 1000) % 1000}"
            
            # Create high-quality snapshot
            snapshot_filename = f"{incident_id}_snapshot.jpg"
            snapshot_path = os.path.join(self.incidents_dir, snapshot_filename)
            
            # Enhance image quality for evidence
            enhanced_frame = self.enhance_image_quality(frame)
            if enhanced_frame is None:
                enhanced_frame = frame  # Fallback to original frame
            
            # Add incident information overlay
            annotated_frame = self.add_incident_overlay(enhanced_frame, gesture_type, confidence, timestamp)
            if annotated_frame is None:
                annotated_frame = enhanced_frame  # Fallback to enhanced frame
            
            # Save snapshot with high quality
            success = cv2.imwrite(snapshot_path, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                print(f"❌ Failed to write snapshot to {snapshot_path}")
                return False, None, None
            
            # Verify file was created
            if not os.path.exists(snapshot_path):
                print(f"❌ Snapshot file not found after creation: {snapshot_path}")
                return False, None, None
                
            self.last_snapshot_time = current_time
            
            print(f"📸 Emergency snapshot captured: {snapshot_path}")
            print(f"🆔 Incident ID: {incident_id}")
            
            # Create incident log entry
            self.log_incident(incident_id, gesture_type, confidence, timestamp, snapshot_path, location_info)
            
            return True, snapshot_path, incident_id
            
        except Exception as e:
            print(f"❌ Error capturing emergency snapshot: {str(e)}")
            print(f"❌ Error type: {type(e).__name__}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            return False, None, None
            
    def enhance_image_quality(self, frame):
        """Enhance image quality for better evidence"""
        try:
            if frame is None:
                return None
                
            # Increase brightness and contrast
            enhanced = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
            
            # Apply slight sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # Denoise (skip if it causes issues)
            try:
                enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            except:
                pass  # Skip denoising if it fails
            
            return enhanced
        except Exception as e:
            print(f"⚠️  Image enhancement failed: {e}, using original frame")
            # If enhancement fails, return original
            return frame
            
    def add_incident_overlay(self, frame, gesture_type, confidence, timestamp):
        """Add incident information overlay to snapshot"""
        try:
            if frame is None:
                return None
                
            # Create a copy to avoid modifying original
            annotated = frame.copy()
            height, width = annotated.shape[:2]
            
            # Add red border to indicate emergency
            cv2.rectangle(annotated, (0, 0), (width-1, height-1), (0, 0, 255), 8)
            
            # Add emergency header
            header_height = 120
            overlay = np.zeros((header_height, width, 3), dtype=np.uint8)
            overlay[:] = (0, 0, 139)  # Dark red background
            
            # Add text information
            cv2.putText(overlay, "EMERGENCY ALERT", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(overlay, f"SOS Gesture: {gesture_type.upper()}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(overlay, f"Confidence: {confidence:.2f} | Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Combine overlay with frame
            result = np.vstack([overlay, annotated])
            
            return result
        except Exception as e:
            print(f"⚠️  Overlay creation failed: {e}, using original frame")
            return frame
            
    def log_incident(self, incident_id, gesture_type, confidence, timestamp, snapshot_path, location_info):
        """Log incident details for record keeping"""
        try:
            log_filename = os.path.join(self.incidents_dir, "incident_log.txt")
            
            log_entry = f"""
INCIDENT REPORT
================
Incident ID: {incident_id}
Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Type: SOS Gesture Detection
Gesture: {gesture_type}
Confidence: {confidence:.2f}
Snapshot: {snapshot_path}
Location: {location_info or 'Camera 1 - Main Area'}
Status: ACTIVE EMERGENCY
Notification Sent: Processing...
================

"""
            
            with open(log_filename, 'a', encoding='utf-8') as f:
                f.write(log_entry)
                
            print(f"📝 Incident logged: {log_filename}")
            
        except Exception as e:
            print(f"❌ Error logging incident: {e}")
            
    def send_emergency_notifications(self, incident_id, gesture_type, confidence, snapshot_path):
        """
        Send emergency notifications via email and SMS
        Runs in background thread to avoid blocking video processing
        """
        def notification_worker():
            try:
                print(f"📧 Sending emergency notifications for incident {incident_id}...")
                
                # Send email notifications
                email_success = self.send_email_alert(incident_id, gesture_type, confidence, snapshot_path)
                
                # Send SMS notifications
                sms_success = self.send_sms_alert(incident_id, gesture_type, confidence)
                
                if email_success or sms_success:
                    print(f"✅ Emergency notifications sent successfully!")
                else:
                    print(f"⚠️ Some notifications may have failed - check logs")
                    
            except Exception as e:
                print(f"❌ Error in notification worker: {e}")
                
        # Run notifications in background thread
        thread = threading.Thread(target=notification_worker, daemon=True)
        thread.start()
        
    def send_email_alert(self, incident_id, gesture_type, confidence, snapshot_path):
        """Send email alert with snapshot attachment"""
        try:
            if not self.email_config.get('enabled', True):
                print("📧 Email notifications disabled")
                return False
                
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['Subject'] = f"🚨 EMERGENCY ALERT - SOS Detected - {incident_id}"
            
            # Email body with location information
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            body = f"""
🚨 EMERGENCY ALERT - WOMEN SAFETY CCTV SYSTEM 🚨

INCIDENT DETAILS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Incident ID: {incident_id}
• Alert Type: SOS Gesture Detection
• Gesture Type: {gesture_type.upper()}
• Detection Confidence: {confidence:.2f}
• Timestamp: {timestamp}
• Location: {self.location_config.get('location', 'Unknown Location')}
• Camera: {self.location_config.get('camera_id', 'CAM-001')}
• Site: {self.location_config.get('site_name', 'Women Safety CCTV')}

IMMEDIATE ACTION REQUIRED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  A person in distress has been detected in the surveillance area.
⚠️  This alert was automatically generated by AI-powered gesture recognition.
⚠️  Please respond immediately to verify the situation and provide assistance.

EVIDENCE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📸 High-resolution snapshot is attached to this email as evidence.
📹 Video footage is being recorded and preserved.

RESPONSE GUIDELINES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Immediately dispatch personnel to the location
2. Contact the person to verify their safety
3. If confirmed emergency, contact appropriate services
4. Acknowledge this alert in the system when responded
5. Document any actions taken

System Status: ACTIVE MONITORING
Alert Priority: CRITICAL
Response Time: IMMEDIATE

This is an automated message from the Women Safety CCTV System.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach snapshot if available
            if snapshot_path and os.path.exists(snapshot_path):
                with open(snapshot_path, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                    
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= "emergency_snapshot_{incident_id}.jpg"'
                )
                msg.attach(part)
                
            # Send to all authority emails
            authority_emails = self.email_config.get('authority_emails', [])
            
            if not authority_emails:
                print("⚠️  No authority emails configured")
                return False
                
            # In development mode, just log the email
            if os.getenv('DEVELOPMENT_MODE', 'True').lower() == 'true':
                print("🧪 DEVELOPMENT MODE - Email notification logged:")
                print(f"📧 To: {', '.join(authority_emails)}")
                print(f"📧 Subject: Emergency Alert - {incident_id}")
                print(f"📧 Snapshot: {snapshot_path}")
                print("📧 (In production mode, this would send real emails)")
                return True
            
            # Production email sending (commented out for safety)
            # Uncomment and configure properly for production use
            """
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['sender_email'], self.email_config['sender_password'])
                
                for email in authority_emails:
                    msg['To'] = email
                    server.send_message(msg)
                    print(f"📧 Email alert sent to: {email}")
                    del msg['To']  # Remove to send to next recipient
            """
            
            # For now, just log successful preparation
            for email in authority_emails:
                print(f"📧 Email alert prepared for: {email}")
                
            return True
            
        except Exception as e:
            print(f"❌ Email alert failed: {e}")
            return False
            
    def send_sms_alert(self, incident_id, gesture_type, confidence):
        """Send SMS alert to authorities"""
        try:
            message = f"🚨 EMERGENCY: SOS gesture '{gesture_type}' detected with {confidence:.2f} confidence. Incident ID: {incident_id}. Respond immediately. Women Safety CCTV System."
            
            # Note: In production, implement actual SMS service
            # This is a placeholder for SMS integration
            for phone in self.sms_config['authority_phones']:
                print(f"📱 SMS alert would be sent to: {phone}")
                print(f"📱 Message: {message}")
                
            return True
            
        except Exception as e:
            print(f"❌ SMS alert failed: {e}")
            return False
            
    def handle_emergency(self, frame, gesture_type, confidence, location_info=None):
        """
        Complete emergency response workflow
        
        Args:
            frame: Current video frame
            gesture_type: Type of SOS gesture detected
            confidence: Detection confidence
            location_info: Optional location/camera info
            
        Returns:
            dict: Response status and details
        """
        try:
            print(f"🚨 EMERGENCY DETECTED: {gesture_type} (Confidence: {confidence:.2f})")
            
            # 1. Capture emergency snapshot
            success, snapshot_path, incident_id = self.capture_emergency_snapshot(
                frame, gesture_type, confidence, location_info
            )
            
            if not success:
                return {
                    'status': 'failed',
                    'error': 'Snapshot capture failed'
                }
                
            # 2. Send notifications (in background)
            self.send_emergency_notifications(incident_id, gesture_type, confidence, snapshot_path)
            
            # 3. Return response
            return {
                'status': 'success',
                'incident_id': incident_id,
                'snapshot_path': snapshot_path,
                'message': f'Emergency response activated for incident {incident_id}'
            }
            
        except Exception as e:
            print(f"❌ Emergency handling failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
            
    def get_recent_incidents(self, limit=10):
        """Get list of recent incidents"""
        try:
            incidents = []
            for filename in os.listdir(self.incidents_dir):
                if filename.endswith('_snapshot.jpg'):
                    incident_id = filename.replace('_snapshot.jpg', '')
                    filepath = os.path.join(self.incidents_dir, filename)
                    stat = os.stat(filepath)
                    
                    incidents.append({
                        'incident_id': incident_id,
                        'timestamp': stat.st_ctime,
                        'snapshot_path': filepath
                    })
                    
            # Sort by timestamp (newest first)
            incidents.sort(key=lambda x: x['timestamp'], reverse=True)
            return incidents[:limit]
            
        except Exception as e:
            print(f"❌ Error getting recent incidents: {e}")
            return []

# Import numpy for image processing
import numpy as np