"""Optical flow analysis for crowd movement."""
import cv2
import numpy as np

class FlowAnalyzer:
    def __init__(self, config):
        self.config = config
        self.method = config['method']
        self.flow_interval = config['flow_interval']
        self.vector_scale = config['vector_scale']
        self.min_magnitude = config['min_flow_magnitude']
        self.arrow_density = config['arrow_density']
        self.prev_gray = None
        self.frame_count = 0
    
    def analyze(self, frame):
        """Analyze optical flow. Returns (flow, flow_viz, flow_stats)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return None, None, None
        
        self.frame_count += 1
        if self.frame_count % self.flow_interval != 0:
            return None, None, None
        
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, pyr_scale=0.5, levels=3,
            winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        flow_viz = self._visualize_flow(frame, flow)
        flow_stats = self._calculate_stats(flow)
        self.prev_gray = gray
        return flow, flow_viz, flow_stats
    
    def _visualize_flow(self, frame, flow):
        vis = frame.copy()
        h, w = flow.shape[:2]
        for y in range(0, h, self.arrow_density):
            for x in range(0, w, self.arrow_density):
                fx, fy = flow[y, x]
                magnitude = np.sqrt(fx**2 + fy**2)
                if magnitude < self.min_magnitude:
                    continue
                x2, y2 = int(x + fx * self.vector_scale), int(y + fy * self.vector_scale)
                color = self._magnitude_to_color(magnitude)
                cv2.arrowedLine(vis, (x, y), (x2, y2), color, 1, tipLength=0.3)
        return vis
    
    def _magnitude_to_color(self, magnitude):
        norm = min(magnitude / 20.0, 1.0)
        if norm < 0.5:
            r, g, b = int(255 * norm * 2), 255, 0
        else:
            r, g, b = 255, int(255 * (1 - (norm - 0.5) * 2)), 0
        return (b, g, r)
    
    def _calculate_stats(self, flow):
        magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
        mask = magnitude > self.min_magnitude
        if not mask.any():
            return {'avg_direction': 0, 'avg_magnitude': 0, 'dominant_flow': 'none'}
        avg_mag = np.mean(magnitude[mask])
        return {
            'avg_direction': np.arctan2(np.mean(flow[:,:,1][mask]), np.mean(flow[:,:,0][mask])) * 180 / np.pi,
            'avg_magnitude': avg_mag,
            'dominant_flow': 'moving' if avg_mag > 2 else 'static'
        }
    
    def __call__(self, frame):
        return self.analyze(frame)
