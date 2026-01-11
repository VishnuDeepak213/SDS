# 👥 Crowd Surveillance Dashboard

An interactive Streamlit web application for detecting, tracking, and analyzing crowd behavior in real-time using YOLOv8 and DeepSORT.

## ✨ Features

- **🖼️ Image Analysis**: Upload images and detect crowds with density heatmaps
- **🎥 Video Analysis**: Process videos to detect, track, and analyze crowd dynamics
- **👤 Person Detection**: Real-time person detection using YOLOv8 (nano/small)
- **📍 Tracking**: DeepSORT-based multi-person tracking with ID persistence
- **📊 Density Estimation**: Grid-based crowd density classification (Low/Medium/High/Critical)
- **🌊 Optical Flow**: Motion analysis and anomaly detection
- **⚡ GPU Acceleration**: Optional NVIDIA GPU support for 27x speedup
- **🎨 Interactive Dashboard**: Streamlit-based web interface with real-time results

## 🚀 Quick Start

### Local Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/sds.git
cd sds

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run streamlit_app.py
```

Access the dashboard at: **http://localhost:8501**

### Streamlit Cloud Deployment
1. Push to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io)
3. Click "New App" and select this repository
4. Set main file to: `streamlit_app.py`
5. Deploy!

## 📋 System Requirements

- **Python**: 3.10+
- **RAM**: 4GB minimum (8GB+ recommended)
- **GPU**: Optional (NVIDIA RTX 3050+ recommended for 30fps+ video)
- **Storage**: 500MB (includes models)

## 📦 Project Structure

```
sds/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── config/
│   └── config.yaml          # Configuration settings
├── src/
│   ├── core/                # Pipeline orchestration
│   ├── detection/           # YOLOv8 person detection
│   ├── tracking/            # DeepSORT tracking
│   ├── density/             # Crowd density estimation
│   ├── flow/                # Optical flow analysis
│   ├── threats/             # Anomaly detection
│   ├── utils/               # Logging and video utilities
│   └── visualization/       # Rendering and overlays
├── yolov8n.pt              # YOLOv8 Nano model weights
├── README.md                # This file
└── .gitignore              # Git ignore rules
```

## 🎯 Dashboard Usage

### Home Page
- Project overview
- Feature selection (Detection, Tracking, Density, Flow)
- Configuration tips

### 🖼️ Image Analysis
1. Click "Image Analysis" in sidebar
2. Upload JPG/PNG image
3. Select desired features
4. Click "Analyze"
5. View results with detection boxes and density heatmap

### 🎥 Video Analysis
1. Click "Video Analysis" in sidebar
2. Upload MP4/AVI video or provide URL
3. Select features and detection sensitivity
4. Click "Analyze Video"
5. Download analyzed video with overlays
6. View metrics (detections, tracking, density)

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
detection:
  device: "cuda"              # "cuda" or "cpu"
  confidence: 0.5             # Detection threshold (0.3-0.7)
  model: "yolov8n"           # "yolov8n" or "yolov8s"

tracking:
  max_age: 30                # Frames before losing track
  n_init: 3                  # Frames to confirm track

density:
  grid_rows: 8               # Density grid rows
  grid_cols: 6               # Density grid columns
  thresholds: [5, 15, 30, 50]  # LOW, MEDIUM, HIGH, CRITICAL
```

## 🔧 Advanced Usage

### GPU Acceleration
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Batch Video Processing
```bash
python analyze_crowd_video.py --input video.mp4 --output results/
```

### Generate Synthetic Test Data
```bash
python generate_crowd_video.py --duration 30 --fps 30 --output test_video.mp4
```

## 📊 Performance Benchmarks

| Video | Resolution | Frames | CPU Time | GPU Time | Speedup |
|-------|------------|--------|----------|----------|---------|
| UMN Dataset | 320x240 | 7,739 | 4.5 hours | 10 minutes | 27x |
| Synthetic | 1280x720 | 450 | 5 minutes | 20 seconds | 15x |

## 🎓 Models & Datasets

- **Detection Model**: YOLOv8 Nano (3.2 MB) - pretrained COCO
- **Tracking**: DeepSORT with person Re-ID embeddings
- **Test Datasets**: 
  - UMN Crowd Activity Dataset
  - ShanghaiTech Crowd Counting

## 🐛 Troubleshooting

### Dashboard won't start
```bash
# Kill existing process and clear cache
pkill streamlit
rm -rf ~/.streamlit

# Restart
streamlit run streamlit_app.py
```

### CUDA not available
```bash
# Check GPU detection
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall CUDA PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Video processing too slow
- Use YOLOv8 Nano model (default)
- Enable GPU acceleration
- Reduce video resolution
- Process in smaller chunks

## 📝 Documentation

- [PROJECT_RESULTS.md](PROJECT_RESULTS.md) - Detailed results and benchmarks
- [QUICK_COMMANDS.md](QUICK_COMMANDS.md) - All useful terminal commands

## 📄 License

[Specify your license here]

## 🤝 Contributing

Contributions welcome! Please feel free to submit pull requests.

## 📧 Contact

[Add contact information]

---

**Last Updated**: January 11, 2026  
**Status**: Production Ready ✅
