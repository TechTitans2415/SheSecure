import time
import json
from datetime import datetime
from typing import Dict, List
import os

class AlertManager:
    def __init__(self):
        self.active_alerts = []  # Store active alerts
        self.alert_history = []  # Store alert history
        self.max_history = 100   # Keep last 100 alerts
        
        # Create alerts directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
    def trigger_alert(self, alert_data: Dict):
        """Trigger a new alert"""
        alert = {
            'id': f"alert_{int(time.time() * 1000)}",
            'timestamp': datetime.now().isoformat(),
            'type': alert_data.get('type', 'UNKNOWN'),
            'severity': alert_data.get('severity', 'MEDIUM'),
            'message': alert_data.get('message', 'Alert triggered'),
            'details': alert_data,
            'status': 'ACTIVE',
            'acknowledged': False
        }
        
        # Add to active alerts
        self.active_alerts.append(alert)
        
        # Add to history
        self.alert_history.append(alert.copy())
        
        # Keep history manageable
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
            
        # Save to file
        self._save_alert_to_file(alert)
        
        # Trigger notification actions
        self._send_notifications(alert)
        
        return alert
    
    def acknowledge_alert(self, alert_id: str):
        """Mark an alert as acknowledged"""
        for alert in self.active_alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                alert['ack_timestamp'] = datetime.now().isoformat()
                break
                
    def clear_alert(self, alert_id: str):
        """Remove an alert from active alerts"""
        self.active_alerts = [a for a in self.active_alerts if a['id'] != alert_id]
        
    def get_active_alerts(self) -> List[Dict]:
        """Get all active alerts"""
        return self.active_alerts.copy()
        
    def get_alert_history(self, limit: int = 50) -> List[Dict]:
        """Get recent alert history"""
        return self.alert_history[-limit:] if self.alert_history else []
        
    def get_critical_alerts(self) -> List[Dict]:
        """Get only critical active alerts"""
        return [a for a in self.active_alerts if a['severity'] == 'CRITICAL']
        
    def _save_alert_to_file(self, alert: Dict):
        """Save alert to JSON file"""
        try:
            alerts_file = 'logs/alerts.json'
            
            # Load existing alerts
            alerts_data = []
            if os.path.exists(alerts_file):
                try:
                    with open(alerts_file, 'r') as f:
                        alerts_data = json.load(f)
                except:
                    alerts_data = []
            
            # Add new alert
            alerts_data.append(alert)
            
            # Keep only last 200 alerts in file
            if len(alerts_data) > 200:
                alerts_data = alerts_data[-200:]
            
            # Save back to file
            with open(alerts_file, 'w') as f:
                json.dump(alerts_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving alert to file: {e}")
    
    def _send_notifications(self, alert: Dict):
        """Send notifications for alert (placeholder for future integrations)"""
        # Print to console for now
        severity_icon = {
            'CRITICAL': '🚨',
            'HIGH': '⚠️',
            'MEDIUM': '⚡',
            'LOW': 'ℹ️'
        }
        
        icon = severity_icon.get(alert['severity'], '📢')
        print(f"\n{icon} {alert['severity']} ALERT {icon}")
        print(f"Type: {alert['type']}")
        print(f"Message: {alert['message']}")
        print(f"Time: {alert['timestamp']}")
        print("-" * 50)
        
        # Future integrations can be added here:
        # - Email notifications
        # - SMS alerts  
        # - Webhook calls
        # - Push notifications
        
    def get_alert_stats(self) -> Dict:
        """Get alert statistics"""
        total_alerts = len(self.alert_history)
        active_count = len(self.active_alerts)
        critical_count = len(self.get_critical_alerts())
        
        # Count by type
        type_counts = {}
        for alert in self.alert_history:
            alert_type = alert['type']
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
            
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_count,
            'critical_alerts': critical_count,
            'alert_types': type_counts,
            'last_alert': self.alert_history[-1]['timestamp'] if self.alert_history else None
        }

# Global alert manager instance
alert_manager = AlertManager()

def trigger_alert(alert_data: Dict) -> Dict:
    """Trigger a new alert"""
    return alert_manager.trigger_alert(alert_data)

def create_alert(alert_data: Dict) -> str:
    """Create a new alert and return alert ID"""
    alert = alert_manager.trigger_alert(alert_data)
    return alert['id']

def get_active_alerts() -> List[Dict]:
    """Get all active alerts"""
    return alert_manager.get_active_alerts()

def get_alert_history(limit: int = 50) -> List[Dict]:
    """Get recent alert history"""
    return alert_manager.get_alert_history(limit)

def acknowledge_alert(alert_id: str):
    """Acknowledge an alert"""
    return alert_manager.acknowledge_alert(alert_id)

def clear_alert(alert_id: str):
    """Clear an alert"""
    return alert_manager.clear_alert(alert_id)

def get_alert_stats() -> Dict:
    """Get alert statistics"""
    return alert_manager.get_alert_stats()