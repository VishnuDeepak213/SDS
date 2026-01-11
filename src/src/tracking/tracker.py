"""DeepSORT multi-person tracker."""
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class PersonTracker:
    def __init__(self, config):
        self.config = config
        self.tracker = DeepSort(
            max_age=config['max_age'],
            n_init=config['n_init'],
            max_iou_distance=config['max_iou_distance'],
            max_cosine_distance=config['max_cosine_distance'],
            nn_budget=config['nn_budget'],
            embedder="mobilenet" if config['use_appearance'] else None,
            embedder_gpu=True
        )
        self.track_history = {}
    
    def update(self, frame, detections):
        """Update tracker with detections. Returns track list."""
        det_list = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            w, h = x2 - x1, y2 - y1
            det_list.append(([x1, y1, w, h], conf, int(cls)))
        
        tracks = self.tracker.update_tracks(det_list, frame=frame)
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_ltrb()
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(center)
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)
        
        return tracks
    
    def __call__(self, frame, detections):
        return self.update(frame, detections)
