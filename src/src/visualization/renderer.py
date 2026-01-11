"""Visualization and rendering."""
import cv2
import numpy as np

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.font_scale = config['font_scale']
        self.line_thickness = config['line_thickness']
        self.colors = [(int(r), int(g), int(b)) 
                      for r, g, b in np.random.randint(0, 255, (100, 3))]
    
    def render(self, frame, tracks=None, detections=None, density_heatmap=None, flow_viz=None,
              alerts=None, threats=None, fps=None):
        """Render visualizations on frame."""
        vis = frame.copy()
        
        if self.config['show_density_heatmap'] and density_heatmap is not None:
            # Ensure heatmap matches frame dimensions
            if density_heatmap.shape[:2] != (frame.shape[0], frame.shape[1]):
                density_heatmap = cv2.resize(density_heatmap, 
                                             (frame.shape[1], frame.shape[0]))
            alpha = 0.4
            vis = cv2.addWeighted(vis, 1 - alpha, density_heatmap, alpha, 0)
        
        if self.config['show_flow_arrows'] and flow_viz is not None:
            # Ensure flow viz matches frame dimensions
            if flow_viz.shape[:2] != (frame.shape[0], frame.shape[1]):
                flow_viz = cv2.resize(flow_viz, 
                                      (frame.shape[1], frame.shape[0]))
            vis = cv2.addWeighted(vis, 0.7, flow_viz, 0.3, 0)
        
        if self.config['show_bboxes']:
            # Prefer tracks when available (with IDs); otherwise draw raw detections
            if tracks:
                vis = self._draw_tracks(vis, tracks)
            elif detections is not None and len(detections) > 0:
                vis = self._draw_detections(vis, detections)
        
        if self.config['show_threats']:
            vis = self._draw_alerts(vis, alerts, threats)
        
        if fps is not None:
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                       (0, 255, 0), 2)
        
        return vis
    
    def _draw_tracks(self, frame, tracks):
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)
            color = self.colors[int(track_id) % len(self.colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)
            if self.config['show_ids']:
                label = f"ID:{track_id}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                           self.font_scale, 1)
                cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                           (255, 255, 255), 1)
        return frame

    def _draw_detections(self, frame, detections):
        # detections: [x1, y1, x2, y2, conf, cls]
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            conf = float(det[4]) if len(det) > 4 else 1.0
            color = (0, 255, 255)  # Yellow for raw detections
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)
            label = f"person {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       self.font_scale, 1)
            cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                        (0, 0, 0), 1)
        return frame
    
    def _draw_alerts(self, frame, alerts, threats):
        y_offset = 60
        if alerts:
            for alert in alerts:
                if alert['level'] in ['HIGH', 'CRITICAL']:
                    color = (0, 0, 255) if alert['level'] == 'CRITICAL' else (0, 165, 255)
                    text = f"{alert['level']}: {alert['message']}"
                    cv2.putText(frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, 2)
                    y_offset += 25
        if threats:
            for threat in threats:
                color = (0, 0, 255) if threat['level'] == 'CRITICAL' else (0, 165, 255)
                text = f"THREAT: {threat['message']}"
                cv2.putText(frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, 2)
                y_offset += 25
        return frame
    
    def __call__(self, frame, **kwargs):
        return self.render(frame, **kwargs)
