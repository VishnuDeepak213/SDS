"""Main SDS processing pipeline."""
import cv2
import time
from pathlib import Path

from src.detection.detector import PersonDetector
from src.tracking.tracker import PersonTracker
from src.density.estimator import DensityEstimator
from src.flow.analyzer import FlowAnalyzer
from src.threats.detector import ThreatDetector
from src.visualization.renderer import Visualizer
from src.utils.video import VideoCapture, VideoWriter
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class SDSPipeline:
    def __init__(self, config):
        self.config = config
        
        logger.info(f"Opening video source: {config['video']['source']}")
        self.video = VideoCapture(config['video']['source'])
        logger.info(f"Video: {self.video.width}x{self.video.height} @ {self.video.fps:.1f} FPS")
        
        logger.info("Initializing modules...")
        self.detector = PersonDetector(config['detection'])
        self.tracker = PersonTracker(config['tracking'])
        
        if config['density']['enabled']:
            self.density_estimator = DensityEstimator(
                config['density'], (self.video.height, self.video.width)
            )
        else:
            self.density_estimator = None
        
        if config['flow']['enabled']:
            self.flow_analyzer = FlowAnalyzer(config['flow'])
        else:
            self.flow_analyzer = None
        
        if config['threats']['enabled']:
            self.threat_detector = ThreatDetector(config['threats'])
        else:
            self.threat_detector = None
        
        self.visualizer = Visualizer(config['visualization'])
        
        if config['output']['save_video']:
            output_dir = Path(config['output']['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"output_{int(time.time())}.mp4"
            logger.info(f"Saving output to: {output_path}")
            self.video_writer = VideoWriter(
                str(output_path), self.video.fps, self.video.width, self.video.height
            )
        else:
            self.video_writer = None
        
        self.show_display = config['visualization'].get('show_display', True)
    
    def run(self):
        logger.info("Starting SDS pipeline...")
        frame_num = 0
        start_time = time.time()
        fps = 0
        
        try:
            while True:
                ret, frame = self.video.read()
                if not ret:
                    logger.info("End of video stream")
                    break
                
                frame_num += 1
                
                # Detection
                detections = self.detector(frame)
                
                # Tracking
                tracks = self.tracker(frame, detections)
                
                # Density
                density_heatmap, density_alerts = None, []
                if self.density_estimator:
                    _, density_heatmap, density_alerts = self.density_estimator(detections)
                
                # Flow
                flow_viz, flow_stats = None, None
                if self.flow_analyzer:
                    _, flow_viz, flow_stats = self.flow_analyzer(frame)
                
                # Threats
                threats = []
                if self.threat_detector:
                    threats = self.threat_detector(tracks, density_alerts, flow_stats, frame_num)
                
                # Visualization
                vis_frame = self.visualizer.render(
                    frame, tracks=tracks, density_heatmap=density_heatmap,
                    flow_viz=flow_viz, alerts=density_alerts, threats=threats,
                    fps=fps if self.config['output']['show_fps'] else None
                )
                
                if self.show_display:
                    cv2.imshow('SDS', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Stopped by user")
                        break
                
                if self.video_writer:
                    self.video_writer.write(vis_frame)
                
                if frame_num % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = 30 / elapsed
                    start_time = time.time()
                    logger.info(f"Frame {frame_num}: {len(tracks)} tracks, "
                              f"{len(detections)} detections, {fps:.1f} FPS")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        logger.info("Cleaning up...")
        self.video.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        logger.info("Pipeline stopped")
