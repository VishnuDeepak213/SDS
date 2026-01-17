"""
SDS - Smart Detection & Surveillance Dashboard
Video Analysis with Metrics & Graphs
"""
import streamlit as st
import cv2
import yaml
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

# Ensure src/ is on Python path BEFORE importing from src
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Now import from src modules
from src.detection.detector import PersonDetector
from src.tracking.tracker import PersonTracker
from src.density.estimator import DensityEstimator
from src.threats.detector import ThreatDetector

# Page configuration
st.set_page_config(
    page_title="SDS - Video Analysis",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: white;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Load configuration
@st.cache_resource
def load_config():
    config_path = Path(__file__).parent / "config" / "config.yaml"
    if not config_path.exists():
        st.error(f"❌ Missing config file: {config_path}")
        st.stop()
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Initialize modules
@st.cache_resource
def initialize_modules(config):
    detector = PersonDetector(config['detection'])
    tracker = PersonTracker(config['tracking'])
    threat_detector = ThreatDetector(config['threats'])
    return detector, tracker, threat_detector

def main():
    """Main video analysis application"""
    
    st.markdown("""
    <div class='main-header'>
        🎥 SDS - Video Analysis Dashboard
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("Upload a video to analyze crowd dynamics with metrics and visualizations")
    
    config = load_config()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Select Video",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Choose a video file to analyze"
    )
    
    if uploaded_file:
        st.markdown("### ⚙️ Analysis Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_detection = st.checkbox("👤 Person Detection", value=True, key="v_det")
        with col2:
            show_density = st.checkbox("📊 Crowd Density", value=True, key="v_dens")
        with col3:
            show_tracking = st.checkbox("🎯 Tracking", value=True, key="v_track")
        
        max_frames = st.slider("Max Frames to Process", 50, 500, 200, step=50)
        
        if st.button("▶️ Analyze Video", use_container_width=True):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_video_path = tmp_file.name
            
            cap = None
            out = None
            output_video_path = None
            
            try:
                # Load video
                cap = cv2.VideoCapture(tmp_video_path)
                if not cap.isOpened():
                    raise RuntimeError("Failed to open video file.")

                ret_first, first_frame = cap.read()
                if not ret_first:
                    raise RuntimeError("Could not read frames from video.")

                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps is None or fps <= 0:
                    fps = 25.0

                vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if vid_width <= 0 or vid_height <= 0:
                    vid_height, vid_width = first_frame.shape[:2]

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                st.success("✅ Video loaded successfully!")
                st.info(f"📹 Resolution: {vid_width}x{vid_height} | FPS: {fps:.1f} | Total Frames: {total_frames}")

                # Initialize modules
                detector, tracker, threat_detector = initialize_modules(config)
                density_estimator = DensityEstimator(config['density'], (vid_width, vid_height))

                # Prepare writer with H264 codec
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='_processed.mp4').name
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (vid_width, vid_height))
                if not out.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_video_path, fourcc, fps, (vid_width, vid_height))
                    if not out.isOpened():
                        raise RuntimeError("Failed to initialize video writer.")

                # Progress tracking
                progress_bar = st.progress(0)
                processed_frames = 0
                total_detections = []
                density_over_time = []
                frame_numbers = []

                with st.spinner("🔄 Processing frames..."):
                    while cap.isOpened() and processed_frames < max_frames:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        output_frame = frame.copy()
                        detections = detector.detect(frame)
                        total_detections.append(len(detections))
                        frame_numbers.append(processed_frames)

                        if show_detection and len(detections) > 0:
                            for det in detections:
                                x1, y1, x2, y2 = map(int, det[:4])
                                conf = det[4]
                                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(output_frame, f'{conf:.2f}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            if show_tracking:
                                try:
                                    tracks = tracker.update(frame, detections)
                                    for track in tracks:
                                        if track.is_confirmed():
                                            x1, y1, x2, y2 = map(int, track.to_tlbr())
                                            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                            cv2.putText(output_frame, f'ID:{track.track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                except Exception:
                                    pass

                        if show_density and len(detections) > 0:
                            try:
                                density_grid, _, _ = density_estimator.estimate(detections)
                                density_count = density_grid.sum()
                                density_over_time.append(density_count)
                                thresholds = config['density']['thresholds']
                                
                                if density_count >= thresholds['critical']:
                                    level = "CRITICAL"
                                    color = (0, 0, 255)
                                elif density_count >= thresholds['high']:
                                    level = "HIGH"
                                    color = (0, 165, 255)
                                elif density_count >= thresholds['medium']:
                                    level = "MEDIUM"
                                    color = (0, 255, 255)
                                else:
                                    level = "LOW"
                                    color = (0, 255, 0)
                                
                                cv2.putText(output_frame, f'Density: {level} ({density_count:.0f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            except Exception:
                                density_over_time.append(len(detections))

                        out.write(output_frame)
                        processed_frames += 1
                        if processed_frames % 10 == 0 or processed_frames >= max_frames:
                            progress_bar.progress(min(processed_frames / max_frames, 1.0))

                progress_bar.progress(1.0)
                st.success(f"✅ Processing complete! Analyzed {processed_frames} frames")

                # Display metrics
                st.markdown("---")
                st.markdown("### 📊 Analysis Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Frames", processed_frames)
                with col2:
                    st.metric("Avg Persons/Frame", f"{np.mean(total_detections):.1f}")
                with col3:
                    st.metric("Max Persons", int(np.max(total_detections)))
                with col4:
                    st.metric("Min Persons", int(np.min(total_detections)))

                # Create graphs
                st.markdown("### 📈 Detection Count Over Time")
                
                df_detections = pd.DataFrame({
                    'Frame': frame_numbers,
                    'Person Count': total_detections
                })
                
                fig_detections = go.Figure()
                fig_detections.add_trace(go.Scatter(
                    x=df_detections['Frame'],
                    y=df_detections['Person Count'],
                    mode='lines+markers',
                    name='Person Count',
                    line=dict(color='#667eea', width=2),
                    marker=dict(size=5)
                ))
                fig_detections.update_layout(
                    title='Person Detection Count Over Time',
                    xaxis_title='Frame Number',
                    yaxis_title='Number of Persons Detected',
                    hovermode='x unified',
                    template='plotly_dark'
                )
                st.plotly_chart(fig_detections, use_container_width=True)

                # Density graph
                if show_density and len(density_over_time) > 0:
                    st.markdown("### 📊 Crowd Density Over Time")
                    
                    df_density = pd.DataFrame({
                        'Frame': frame_numbers[:len(density_over_time)],
                        'Density Count': density_over_time
                    })
                    
                    fig_density = go.Figure()
                    fig_density.add_trace(go.Scatter(
                        x=df_density['Frame'],
                        y=df_density['Density Count'],
                        mode='lines+markers',
                        name='Density',
                        line=dict(color='#FF6B6B', width=2),
                        marker=dict(size=5),
                        fill='tozeroy'
                    ))
                    
                    # Add threshold lines
                    thresholds = config['density']['thresholds']
                    fig_density.add_hline(y=thresholds['critical'], line_dash="dash", line_color="red", 
                                        annotation_text="Critical", annotation_position="right")
                    fig_density.add_hline(y=thresholds['high'], line_dash="dash", line_color="orange", 
                                        annotation_text="High", annotation_position="right")
                    fig_density.add_hline(y=thresholds['medium'], line_dash="dash", line_color="yellow", 
                                        annotation_text="Medium", annotation_position="right")
                    
                    fig_density.update_layout(
                        title='Crowd Density Over Time',
                        xaxis_title='Frame Number',
                        yaxis_title='Density Count',
                        hovermode='x unified',
                        template='plotly_dark'
                    )
                    st.plotly_chart(fig_density, use_container_width=True)

                # Download processed video
                if output_video_path and os.path.exists(output_video_path):
                    with open(output_video_path, 'rb') as f:
                        video_bytes = f.read()
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=video_bytes,
                        file_name="video_analysis.mp4",
                        mime="video/mp4"
                    )

            except Exception as e:
                st.error(f"❌ Error processing video: {str(e)}")
            finally:
                # Clean up
                if cap is not None:
                    try:
                        cap.release()
                    except:
                        pass
                if out is not None:
                    try:
                        out.release()
                    except:
                        pass
                if os.path.exists(tmp_video_path):
                    try:
                        os.unlink(tmp_video_path)
                    except:
                        pass
                if output_video_path and os.path.exists(output_video_path):
                    try:
                        os.unlink(output_video_path)
                    except:
                        pass
    else:
        st.info("📤 Upload a video to begin analysis")

if __name__ == "__main__":
    main()
