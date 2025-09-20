import time
import datetime
from typing import List, Tuple, Dict
import logging
import sys
import os

# Add parent directory to path for importing services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.alerts import trigger_alert

# Setup logging for incidents
logging.basicConfig(
    filename='logs/incidents.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AnomalyDetector:
    def __init__(self):
        self.alert_cooldown = 15  # Reduced cooldown for faster response
        self.last_alerts = {}  # Track last alert times by type
        self.incident_threshold = {
            'lone_woman': 2,      # Reduced from 5 to 2 seconds for faster detection
            'woman_surrounded': 1, # Reduced from 3 to 1 second for immediate response
            'crowd_density': 4,    # Reduced from 8 to 4 seconds
            'immediate_danger': 0  # Immediate trigger for critical situations
        }
        self.detection_history = []  # Track detections over time
        self.immediate_alerts = []  # Store immediate alerts for UI display
        
    def analyze_frame(self, genders: List[Tuple[str, Tuple[int, int, int, int]]]) -> Dict:
        """
        Analyze a frame for anomalies
        Returns: Dictionary with alert information
        """
        current_time = time.time()
        
        # Count people by gender
        men_count = sum(1 for gender, _ in genders if gender == "Man")
        women_count = sum(1 for gender, _ in genders if gender == "Woman")
        unknown_count = sum(1 for gender, _ in genders if gender == "Unknown")
        total_people = len(genders)
        
        # Store current detection
        detection_data = {
            'timestamp': current_time,
            'men': men_count,
            'women': women_count,
            'unknown': unknown_count,
            'total': total_people,
            'genders': genders
        }
        
        # Keep only last 30 seconds of history
        self.detection_history.append(detection_data)
        self.detection_history = [d for d in self.detection_history 
                                 if current_time - d['timestamp'] <= 30]
        
        # Analyze for anomalies
        alerts = []
        
        # 1. Lone woman detection
        lone_woman_alert = self._check_lone_woman(detection_data)
        if lone_woman_alert:
            alerts.append(lone_woman_alert)
            
        # 2. Woman surrounded by men
        surrounded_alert = self._check_woman_surrounded(detection_data)
        if surrounded_alert:
            alerts.append(surrounded_alert)
            
        # 3. High crowd density (optional)
        crowd_alert = self._check_crowd_density(detection_data)
        if crowd_alert:
            alerts.append(crowd_alert)
            
        return {
            'alerts': alerts,
            'stats': {
                'men': men_count,
                'women': women_count,
                'unknown': unknown_count,
                'total': total_people
            }
        }
    
    def _check_lone_woman(self, detection_data: Dict) -> Dict:
        """Check for lone woman scenario - Enhanced for safety"""
        men_count = detection_data['men']
        women_count = detection_data['women']
        total_people = detection_data['total']
        
        # Trigger if: exactly 1 woman, no men, especially at night/isolated areas
        if women_count == 1 and men_count == 0 and total_people == 1:
            # Check if this condition persisted for threshold duration
            persistent_duration = self._get_persistent_duration(
                lambda d: d['women'] == 1 and d['men'] == 0 and d['total'] == 1
            )
            
            if persistent_duration >= self.incident_threshold['lone_woman']:
                if self._can_send_alert('lone_woman'):
                    self._log_incident('LONE_WOMAN', detection_data)
                    alert_data = {
                        'type': 'LONE_WOMAN',
                        'severity': 'HIGH',
                        'message': '⚠️ Lone woman detected - monitoring for safety',
                        'duration': persistent_duration,
                        'location': self._get_woman_location(detection_data['genders']),
                        'requires_snapshot': True,
                        'monitor_level': 'HIGH'
                    }
                    # Trigger alert through alert system
                    trigger_alert(alert_data)
                    return alert_data
        
        # Additional check: Lone woman in area where men were recently present
        if women_count == 1 and men_count == 0 and total_people == 1:
            # Check if men were in the area in last 30 seconds
            recent_men = self._check_recent_men_presence(30)
            if recent_men > 0:
                if self._can_send_alert('lone_woman_after_men'):
                    alert_data = {
                        'type': 'LONE_WOMAN_AFTER_MEN',
                        'severity': 'HIGH',
                        'message': f'⚠️ Woman alone after {recent_men} men left area - enhanced monitoring',
                        'duration': persistent_duration,
                        'location': self._get_woman_location(detection_data['genders']),
                        'requires_snapshot': True,
                        'previous_men_count': recent_men
                    }
                    trigger_alert(alert_data)
                    return alert_data
        
        return None
    
    def _check_woman_surrounded(self, detection_data: Dict) -> Dict:
        """Check for woman surrounded by men scenario - IMMEDIATE RESPONSE"""
        men_count = detection_data['men']
        women_count = detection_data['women']
        
        # IMMEDIATE trigger if: 1 woman surrounded by 2+ men (lowered threshold)
        if women_count == 1 and men_count >= 2:
            persistent_duration = self._get_persistent_duration(
                lambda d: d['women'] == 1 and d['men'] >= 2
            )
            
            # More aggressive detection - trigger faster
            if persistent_duration >= self.incident_threshold['woman_surrounded']:
                if self._can_send_alert('woman_surrounded'):
                    self._log_incident('WOMAN_SURROUNDED', detection_data)
                    alert_data = {
                        'type': 'WOMAN_SURROUNDED',
                        'severity': 'CRITICAL',
                        'message': f'🚨 EMERGENCY: Woman surrounded by {men_count} men - IMMEDIATE ATTENTION REQUIRED',
                        'duration': persistent_duration,
                        'men_count': men_count,
                        'location': self._get_woman_location(detection_data['genders']),
                        'emergency_level': 'IMMEDIATE',
                        'requires_snapshot': True,
                        'auto_notify_authorities': True
                    }
                    # Trigger alert through alert system
                    trigger_alert(alert_data)
                    return alert_data
        
        # Even more critical: 1 woman with 4+ men = immediate emergency
        if women_count == 1 and men_count >= 4:
            if self._can_send_alert('critical_surrounded'):
                self._log_incident('CRITICAL_SURROUNDED', detection_data)
                alert_data = {
                    'type': 'CRITICAL_SURROUNDED',
                    'severity': 'CRITICAL',
                    'message': f'🚨🚨 CRITICAL EMERGENCY: Woman outnumbered {men_count} to 1 - AUTHORITIES NOTIFIED',
                    'duration': 0,  # Immediate
                    'men_count': men_count,
                    'location': self._get_woman_location(detection_data['genders']),
                    'emergency_level': 'CRITICAL',
                    'requires_snapshot': True,
                    'auto_notify_authorities': True,
                    'trigger_alarm': True
                }
                trigger_alert(alert_data)
                return alert_data
        
        return None
    
    def _check_crowd_density(self, detection_data: Dict) -> Dict:
        """Check for unusual crowd density"""
        total_people = detection_data['total']
        
        # Trigger if: more than 8 people in frame (adjust based on your camera coverage)
        if total_people > 8:
            persistent_duration = self._get_persistent_duration(
                lambda d: d['total'] > 8
            )
            
            if persistent_duration >= self.incident_threshold['crowd_density']:
                if self._can_send_alert('crowd_density'):
                    self._log_incident('HIGH_CROWD_DENSITY', detection_data)
                    alert_data = {
                        'type': 'HIGH_CROWD_DENSITY',
                        'severity': 'MEDIUM',
                        'message': f'High crowd density detected - {total_people} people in area',
                        'duration': persistent_duration,
                        'crowd_size': total_people
                    }
                    # Trigger alert through alert system
                    trigger_alert(alert_data)
                    return alert_data
        
        return None
    
    def _get_persistent_duration(self, condition_func) -> float:
        """Calculate how long a condition has been true"""
        if not self.detection_history:
            return 0
            
        current_time = self.detection_history[-1]['timestamp']
        duration = 0
        
        # Check from most recent backwards
        for detection in reversed(self.detection_history):
            if condition_func(detection):
                duration = current_time - detection['timestamp']
            else:
                break
                
        return duration
    
    def _can_send_alert(self, alert_type: str) -> bool:
        """Check if enough time has passed since last alert of this type"""
        current_time = time.time()
        last_alert_time = self.last_alerts.get(alert_type, 0)
        
        if current_time - last_alert_time >= self.alert_cooldown:
            self.last_alerts[alert_type] = current_time
            return True
        return False
    
    def _check_recent_men_presence(self, seconds_back: int) -> int:
        """Check if men were present in the area within the last N seconds"""
        current_time = time.time()
        max_men_count = 0
        
        for detection in reversed(self.detection_history):
            time_diff = current_time - detection['timestamp']
            if time_diff <= seconds_back:
                max_men_count = max(max_men_count, detection['men'])
            else:
                break
                
        return max_men_count
    
    def _get_woman_location(self, genders: List[Tuple[str, Tuple[int, int, int, int]]]) -> Tuple[int, int]:
        """Get the center coordinates of the first woman detected"""
        for gender, (x1, y1, x2, y2) in genders:
            if gender == "Woman":
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                return (center_x, center_y)
        return (0, 0)
    
    def _log_incident(self, incident_type: str, detection_data: Dict):
        """Log incident to file"""
        timestamp = datetime.datetime.fromtimestamp(detection_data['timestamp'])
        logging.info(
            f"INCIDENT: {incident_type} | "
            f"Time: {timestamp} | "
            f"Men: {detection_data['men']} | "
            f"Women: {detection_data['women']} | "
            f"Total: {detection_data['total']}"
        )
        
    def get_status_summary(self) -> Dict:
        """Get current system status"""
        if not self.detection_history:
            return {'status': 'No recent detections', 'last_detection': None}
            
        latest = self.detection_history[-1]
        return {
            'status': 'Active',
            'last_detection': datetime.datetime.fromtimestamp(latest['timestamp']),
            'current_count': {
                'men': latest['men'],
                'women': latest['women'],
                'total': latest['total']
            },
            'monitoring_duration': f"{len(self.detection_history)} frames in last 30s"
        }

# Global anomaly detector instance
anomaly_detector = AnomalyDetector()

def analyze_anomalies(genders: List[Tuple[str, Tuple[int, int, int, int]]]) -> Dict:
    """
    Main function to analyze frame for anomalies
    Called from detection.py with the detected genders
    """
    return anomaly_detector.analyze_frame(genders)

def get_system_status() -> Dict:
    """Get current system monitoring status"""
    return anomaly_detector.get_status_summary()
