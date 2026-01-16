"""
SDS - Smart Detection & Surveillance Dashboard
Simple interactive web interface for crowd analysis
"""
import streamlit as st
import cv2
import yaml
import numpy as np
import tempfile
import os
from pathlib import Path
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.detection.detector import PersonDetector
from src.tracking.tracker import PersonTracker
from src.density.estimator import DensityEstimator
from src.flow.analyzer import FlowAnalyzer
from src.threats.detector import ThreatDetector
from src.visualization.renderer import Visualizer

# Page configuration
st.set_page_config(
    page_title="SDS - Crowd Analysis Dashboard",
    page_icon="👥",
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
    .feature-box-image {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .feature-box-video {
        background: linear-gradient(135deg, #FF6B6B 0%, #FFE66D 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load configuration
@st.cache_resource
def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

# Initialize modules
@st.cache_resource
def initialize_modules(config):
    detector = PersonDetector(config['detection'])
    tracker = PersonTracker(config['tracking'])
    threat_detector = ThreatDetector(config['threats'])
    return detector, tracker, threat_detector

def process_image(uploaded_file, features, config):
    """Process uploaded image with selected features"""
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w = frame.shape[:2]
    
    # Initialize modules
    detector, tracker, threat_detector = initialize_modules(config)
    density_estimator = DensityEstimator(config['density'], (w, h))
    visualizer = Visualizer(config['visualization'])
    
    # Detection
    detections = detector(frame)
    results = {
        'frame': frame,
        'detections': detections,
        'num_persons': len(detections),
        'tracks': [],
        'density': None
    }
    
    # Tracking
    if features['tracking']:
        tracks = tracker.update(frame, detections)
        results['tracks'] = [t for t in tracks if t.is_confirmed()]
    
    # Density estimation
    if features['density']:
        density_grid, density_heatmap, density_alerts = density_estimator.estimate(detections)
        total_count = density_grid.sum()
        
        # Determine level
        if total_count >= config['density']['thresholds']['critical']:
            level = 'CRITICAL'
        elif total_count >= config['density']['thresholds']['high']:
            level = 'HIGH'
        elif total_count >= config['density']['thresholds']['medium']:
            level = 'MEDIUM'
        else:
            level = 'LOW'
        
        results['density'] = {
            'grid': density_grid,
            'heatmap': density_heatmap,
            'level': level,
            'count': int(total_count),
            'alerts': density_alerts
        }
    
    # Draw detections on frame
    vis_frame = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        conf = det[4]
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_frame, f'{conf:.2f}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    results['visualized'] = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
    results['original'] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return results

def main():
    """Main dashboard application"""
    
    # Sidebar navigation
    st.sidebar.markdown("# 🎬 SDS Dashboard")
    page = st.sidebar.radio(
        "Select Analysis Type",
        ["🏠 Home", "🖼️ Image Analysis", "🎥 Video Analysis"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🖼️ Image Analysis":
        show_image_analysis()
    elif page == "🎥 Video Analysis":
        show_video_analysis()

def show_home_page():
    """Display home page with menu"""
    st.markdown("""
    <div class='main-header'>
        👥 SDS - Smart Detection & Surveillance
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the **SDS Crowd Analysis Dashboard**. 
    
    This system provides real-time analysis of crowds and individuals in images and videos.
    """)
    
    st.markdown("---")
    
    # Feature cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='feature-box-image'>
            <h2>🖼️ IMAGE ANALYSIS</h2>
            <p>Upload a single image and get instant analysis:</p>
            <ul>
                <li>👤 Person Detection</li>
                <li>🎯 Individual Tracking</li>
                <li>📊 Crowd Density Estimation</li>
                <li>⚠️ Threat Detection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-box-video'>
            <h2>🎥 VIDEO ANALYSIS</h2>
            <p>Upload a video for comprehensive analysis:</p>
            <ul>
                <li>🎬 Real-time Detection</li>
                <li>📈 Crowd Density Over Time</li>
                <li>🔄 Optical Flow Analysis</li>
                <li>🚨 Anomaly Detection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    ### 📋 Key Features
    
    - **YOLOv8 Detection**: Fast and accurate person detection
    - **DeepSORT Tracking**: Multi-object tracking across frames
    - **Crowd Density**: Grid-based density estimation
    - **Optical Flow**: Movement analysis
    - **Threat Detection**: Anomaly and panic detection
    
    ### 🚀 Getting Started
    1. Select **Image Analysis** or **Video Analysis** from the sidebar
    2. Upload your file
    3. Choose analysis features
    4. View results with visualizations
    """)

def show_image_analysis():
    """Image analysis page"""
    st.markdown("# 🖼️ Image Analysis")
    st.markdown("Upload an image to detect and analyze crowds")
    
    config = load_config()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Select Image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Choose an image file to analyze"
    )
    
    if uploaded_file:
        st.markdown("### ⚙️ Analysis Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_detection = st.checkbox("👤 Person Detection", value=True)
        with col2:
            show_density = st.checkbox("📊 Crowd Density", value=True)
        with col3:
            show_tracking = st.checkbox("🎯 Tracking", value=False)
        
        if st.button("🔍 Analyze Image", use_container_width=True):
            with st.spinner("⏳ Processing image..."):
                try:
                    features = {
                        'detection': show_detection,
                        'tracking': show_tracking,
                        'density': show_density,
                        'flow': False,
                        'threats': False
                    }
                    
                    results = process_image(uploaded_file, features, config)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(results['original'], caption="Original Image", use_column_width=True)
                    with col2:
                        st.image(results['visualized'], caption="Detection Result", use_column_width=True)
                    
                    # Statistics
                    st.markdown("---")
                    st.markdown("### 📊 Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("👤 Persons Detected", results['num_persons'])
                    
                    if results['density']:
                        with col2:
                            st.metric("📊 Density Level", results['density']['level'])
                        with col3:
                            st.metric("Count", results['density']['count'])
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    else:
        st.info("📤 Upload an image to begin analysis")

def show_video_analysis():
    """Video analysis page"""
    st.markdown("# 🎥 Video Analysis")
    st.markdown("Upload a video to analyze crowd dynamics over time")
    
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
            show_detection = st.checkbox("👤 Detection", value=True, key="v_det")
        with col2:
            show_density = st.checkbox("📊 Density", value=True, key="v_dens")
        with col3:
            show_flow = st.checkbox("🔄 Flow", value=False, key="v_flow")
        
        max_frames = st.slider("Max Frames to Process", 50, 500, 200, step=50)
        
        if st.button("▶️ Analyze Video"):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_video_path = tmp_file.name
            
            output_video_path = None
            cap = None
            out = None
            
            try:
                # Initialize modules
                detector, tracker, threat_detector = initialize_modules(config)
                density_estimator = None
                flow_analyzer = None
                visualizer = Visualizer(config['visualization'])
                
                # Open video
                cap = cv2.VideoCapture(tmp_video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Setup video writer for output
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='_analyzed.mp4').name
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (vid_width, vid_height))
                
                st.success(f"✅ Video loaded: {total_frames} frames @ {fps} FPS")
                
                # Create progress bar and placeholders
                progress_bar = st.progress(0)
                frame_count = 0
                processed_frames = 0
                total_detections = []
                density_over_time = []
                
                # Process video frames
                with st.spinner("🔄 Processing video..."):
                    while cap.isOpened() and processed_frames < max_frames:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        output_frame = frame.copy()
                        
                        # Detection
                        detections = detector(frame)
                        total_detections.append(len(detections))
                        
                        # Draw detections
                        if show_detection:
                            for det in detections:
                                x1, y1, x2, y2 = map(int, det[:4])
                                conf = det[4]
                                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(output_frame, f'{conf:.2f}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # Tracking
                            tracks = tracker.update(frame, detections)
                            for track in tracks:
                                if track.is_confirmed():
                                    x1, y1, x2, y2 = map(int, track.to_tlbr())
                                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                    cv2.putText(output_frame, f'ID:{track.track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        # Density estimation
                        if show_density:
                            if density_estimator is None:
                                density_estimator = DensityEstimator(config['density'], (vid_width, vid_height))
                            density_grid, density_heatmap, _ = density_estimator.estimate(detections)
                            density_count = density_grid.sum()
                            density_over_time.append(density_count)
                            
                            # Add density text
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
                            
                            cv2.putText(output_frame, f'Density: {level} ({density_count:.0f})', 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        # Write frame to output video
                        out.write(output_frame)
                        
                        processed_frames += 1
                        progress = processed_frames / max_frames
                        progress_bar.progress(progress)
                    
                    # Properly release video resources BEFORE trying to read/delete
                    if cap is not None:
                        cap.release()
                        cap = None
                    if out is not None:
                        out.release()
                        out = None
                
                # Display results
                st.markdown("---")
                st.success(f"✅ Processing complete! Analyzed {processed_frames} frames")
                
                # Read processed video for display and download
                if os.path.exists(output_video_path):
                    with open(output_video_path, 'rb') as f:
                        video_bytes = f.read()
                    
                    # Display processed video
                    st.markdown("### 📹 Processed Video")
                    st.video(video_bytes)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=video_bytes,
                        file_name="crowd_analysis_result.mp4",
                        mime="video/mp4"
                    )
                
                # Statistics
                st.markdown("### 📊 Video Analysis Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📹 Total Frames", processed_frames)
                with col2:
                    st.metric("👤 Avg Persons/Frame", f"{np.mean(total_detections):.1f}")
                with col3:
                    st.metric("👥 Max Persons", int(np.max(total_detections)))
                with col4:
                    st.metric("⏱️ Duration", f"{processed_frames/fps:.1f}s")
                
                # Charts
                if total_detections:
                    st.markdown("### 📈 Detection Over Time")
                    fig_det = go.Figure()
                    fig_det.add_trace(go.Scatter(
                        y=total_detections,
                        mode='lines+markers',
                        name='Persons Detected',
                        line=dict(color='#4ECDC4', width=2),
                        marker=dict(size=4)
                    ))
                    fig_det.update_layout(
                        title="Persons Detected per Frame",
                        xaxis_title="Frame Number",
                        yaxis_title="Number of Detections",
                        hovermode='x unified',
                        template='plotly_dark',
                        height=400
                    )
                    st.plotly_chart(fig_det, use_container_width=True)
                
                if density_over_time:
                    st.markdown("### 📊 Crowd Density Over Time")
                    fig_dens = go.Figure()
                    fig_dens.add_trace(go.Scatter(
                        y=density_over_time,
                        mode='lines',
                        name='Total Density',
                        line=dict(color='#FF6B6B', width=2),
                        fill='tozeroy'
                    ))
                    fig_dens.update_layout(
                        title="Crowd Density Over Time",
                        xaxis_title="Frame Number",
                        yaxis_title="Density Count",
                        hovermode='x unified',
                        template='plotly_dark',
                        height=400
                    )
                    st.plotly_chart(fig_dens, use_container_width=True)
                
                # Summary metrics
                st.markdown("### 📋 Summary")
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    if density_over_time:
                        avg_density = np.mean(density_over_time)
                        st.metric("📊 Average Density", f"{avg_density:.1f}")
                
                with summary_col2:
                    max_density = max(density_over_time) if density_over_time else 0
                    thresholds = config['density']['thresholds']
                    if max_density >= thresholds['critical']:
                        level = "🔴 CRITICAL"
                    elif max_density >= thresholds['high']:
                        level = "🟠 HIGH"
                    elif max_density >= thresholds['medium']:
                        level = "🟡 MEDIUM"
                    else:
                        level = "🟢 LOW"
                    st.metric("Peak Density Level", level)
                
                with summary_col3:
                    st.metric("Processing Status", "✅ Complete")
                
            except Exception as e:
                st.error(f"❌ Error processing video: {str(e)}")
            finally:
                # Ensure video resources are released
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
                
                # Clean up temp input file
                if os.path.exists(tmp_video_path):
                    try:
                        os.unlink(tmp_video_path)
                    except:
                        pass
                
                # Clean up temp output file
                if output_video_path and os.path.exists(output_video_path):
                    try:
                        os.unlink(output_video_path)
                    except:
                        pass
    else:
        st.info("📤 Upload a video to begin analysis")

if __name__ == "__main__":
    main()
