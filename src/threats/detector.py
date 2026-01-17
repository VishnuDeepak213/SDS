"""Rule-based threat and emergency detection."""
import numpy as np
from collections import defaultdict

class ThreatDetector:
    def __init__(self, config):
        self.config = config
        self.track_speeds = defaultdict(list)
        self.track_last_seen = {}
    
    def detect(self, tracks, density_alerts, flow_stats, frame_num):
        """Detect threats. Returns list of threat dicts."""
        threats = []
        
        if not tracks:
            return threats
        
        # Check panic
        if self._check_high_speed(tracks) and self._check_high_density(density_alerts):
            threats.append({
                'type': 'panic', 'level': 'CRITICAL',
                'message': 'Potential panic detected: High movement + high density'
            })
        
        # Check collapse
        for track in tracks:
            if not track.is_confirmed():
                continue
            bbox = track.to_ltrb()
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if h > 0 and w / h > 2.5:
                threats.append({
                    'type': 'collapse', 'level': 'CRITICAL',
                    'track_id': track.track_id,
                    'message': f'Person collapse detected (ID: {track.track_id})'
                })
        
        return threats
    
    def _check_high_speed(self, tracks):
        speeds = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_ltrb()
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            if track_id in self.track_last_seen:
                prev = self.track_last_seen[track_id]
                speed = np.sqrt((center[0] - prev[0])**2 + (center[1] - prev[1])**2)
                speeds.append(speed)
            self.track_last_seen[track_id] = center
        return np.mean(speeds) > 5 if speeds else False
    
    def _check_high_density(self, density_alerts):
        return any(a['level'] in ['HIGH', 'CRITICAL'] for a in density_alerts)
    
    def __call__(self, tracks, density_alerts, flow_stats, frame_num):
        return self.detect(tracks, density_alerts, flow_stats, frame_num)
