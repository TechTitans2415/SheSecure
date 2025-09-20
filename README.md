# 🚨 Women Safety CCTV System with SOS Detection

A real-time AI-powered surveillance system that automatically detects SOS gestures and triggers emergency response protocols with snapshot capture and authority notifications.

![System Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-2.0+-lightgrey)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-yellow)

## 🎯 Key Features

### 🔍 **Real-time SOS Detection**
- AI-powered gesture recognition for emergency signals
- Supports multiple SOS gestures (hands_up, distress signals)
- Confidence scoring and threshold-based detection
- Real-time processing with minimal latency

### 📸 **Automatic Emergency Response**
- **Instant snapshot capture** when SOS detected
- High-quality image enhancement for evidence
- Automatic incident ID generation
- 5-second cooldown to prevent spam

### 🚨 **Multi-Channel Notifications**
- **Email alerts** with incident details and snapshots
- **SMS notifications** for immediate response
- **Real-time dashboard alerts**
- Configurable emergency contact lists

### 📊 **Comprehensive Logging**
- Detailed incident reports in JSON format
- Emergency logs with timestamps
- Snapshot archival with metadata
- Location tracking and mapping integration

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam or IP camera
- Windows/Linux/macOS

🚨 How It Works
Detection Workflow
Video Processing: Real-time analysis of camera feed
Person Detection: YOLO-based person identification
Gesture Recognition: AI analysis for SOS signals
Emergency Trigger: Automatic response when SOS detected
Snapshot Capture: High-quality evidence capture
Notification Dispatch: Multi-channel alert system
Incident Logging: Comprehensive record keeping

SOS Gestures Supported  
🙋‍♀️ Hands Up: Classic distress signal
🆘 Help Gestures: Various emergency indicators
🔴 Configurable: Easy to add new gesture types

🚨 CRITICAL ALERT 🚨
Type: SOS_GESTURE
Message: SOS gesture detected: hands_up (Confidence: 0.85)
Time: 2025-09-20T15:30:45.123456
--------------------------------------------------
📸 Emergency snapshot captured: incidents/SOS_20250920_153045_123_snapshot.jpg
🆔 Incident ID: SOS_20250920_153045_123
📧 Emergency notifications sent to 3 contacts
✅ Emergency response activated successfully

🛡️ Security Features
Secure configuration management
Development mode safety (prevents accidental alerts)
Input validation and error handling
Incident data protection
Configurable access controls

🔧 Technical Specifications
Language: Python 3.8+
Framework: Flask 2.0+
AI Models: YOLOv8, Custom gesture recognition
Computer Vision: OpenCV 4.0+
Real-time Processing: < 200ms latency
Supported Formats: MP4, AVI, webcam, IP cameras
Image Quality: Enhanced 95% JPEG, multiple formats
