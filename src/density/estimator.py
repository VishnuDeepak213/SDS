"""Crowd density estimation."""
import cv2
import numpy as np

class DensityEstimator:
    def __init__(self, config, frame_shape):
        self.config = config
        self.grid_rows, self.grid_cols = config['grid_size']
        self.thresholds = config['thresholds']
        self.frame_height, self.frame_width = frame_shape
        self.cell_height = self.frame_height // self.grid_rows
        self.cell_width = self.frame_width // self.grid_cols
    
    def estimate(self, detections_or_tracks):
        """Estimate density grid. Returns (density_grid, heatmap, alerts)."""
        density_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
        
        for item in detections_or_tracks:
            # Check if it's a Track object or detection array
            if hasattr(item, 'to_ltrb'):  # Track object
                if not item.is_confirmed():
                    continue
                x1, y1, x2, y2 = item.to_ltrb()
            else:  # Detection array
                x1, y1, x2, y2 = item[:4]
            
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            col = int(center_x // self.cell_width)
            row = int(center_y // self.cell_height)
            row = max(0, min(row, self.grid_rows - 1))
            col = max(0, min(col, self.grid_cols - 1))
            density_grid[row, col] += 1
        
        heatmap = self._generate_heatmap(density_grid)
        alerts = self._check_alerts(density_grid)
        return density_grid, heatmap, alerts
    
    def _generate_heatmap(self, density_grid):
        max_density = max(self.thresholds['critical'], density_grid.max() or 1)
        normalized = (density_grid / max_density * 255).astype(np.uint8)
        heatmap = cv2.resize(normalized, (self.frame_width, self.frame_height),
                            interpolation=cv2.INTER_LINEAR)
        return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    def _check_alerts(self, density_grid):
        alerts = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                count = density_grid[row, col]
                if count >= self.thresholds['critical']:
                    level = 'CRITICAL'
                elif count >= self.thresholds['high']:
                    level = 'HIGH'
                elif count >= self.thresholds['medium']:
                    level = 'MEDIUM'
                else:
                    continue
                alerts.append({
                    'type': 'density', 'level': level, 'location': (row, col),
                    'count': count, 'message': f"{level} density: {count} people"
                })
        return alerts
    
    def __call__(self, detections):
        return self.estimate(detections)
