"""Video utilities."""
import cv2

class VideoCapture:
    def __init__(self, source):
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open: {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def read(self):
        return self.cap.read()
    
    def release(self):
        self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.release()

class VideoWriter:
    def __init__(self, output_path, fps, width, height):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    def write(self, frame):
        self.writer.write(frame)
    
    def release(self):
        self.writer.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.release()
