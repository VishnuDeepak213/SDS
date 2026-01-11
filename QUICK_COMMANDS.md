# SDS Project - Quick Commands Guide

## 📋 Table of Contents
1. [Setup & Installation](#setup--installation)
2. [Running the Dashboard](#running-the-dashboard)
3. [Video Analysis](#video-analysis)
4. [Image Analysis](#image-analysis)
5. [Configuration](#configuration)
6. [Data & Models](#data--models)
7. [Troubleshooting](#troubleshooting)

---

## Setup & Installation

### Initial Setup
```powershell
# Navigate to project directory
cd v:\sds

# Create virtual environment (if needed)
python -m venv venv

# Activate virtual environment
venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download datasets (optional)
python datasets/download_auto.py
```

### Verify Installation
```powershell
# Check Python packages
V:/sds/venv/Scripts/python.exe -c "import torch; print('PyTorch OK')"

# Check GPU support
V:/sds/venv/Scripts/python.exe -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check models exist
Get-ChildItem yolov8*.pt
```

---

## Running the Dashboard

### Start Dashboard (Recommended)
```powershell
# Navigate to project
cd v:\sds

# Run dashboard on default port 8501
V:/sds/venv/Scripts/streamlit.exe run dashboard_simple.py

# Run on custom port
V:/sds/venv/Scripts/streamlit.exe run dashboard_simple.py --server.port=8888

# Access at: http://localhost:8501
```

### Dashboard Features
- **🏠 Home**: Overview and feature selection
- **🖼️ Image Analysis**: Upload and analyze images
- **🎥 Video Analysis**: Upload and analyze videos
- **⚙️ Configuration**: Select analysis features
- **📊 Results**: View metrics and visualizations

### Stop Dashboard
```powershell
# Press Ctrl+C in the terminal window
# Or in PowerShell:
Get-Process streamlit | Stop-Process -Force
```

---

## Video Analysis

### Analyze Existing Video
```powershell
# Analyze UMN crowd video
V:/sds/venv/Scripts/python.exe analyze_crowd_video.py \
  --input "v:/sds/datasets/downloads/umn_crowd_activity.avi" \
  --output "v:/sds/test_results/video"

# Analyze high-density synthetic video
V:/sds/venv/Scripts/python.exe analyze_crowd_video.py \
  --input "v:/sds/test_results/video/high_density_crowd.mp4" \
  --output "v:/sds/test_results/video"

# Analyze custom video
V:/sds/venv/Scripts/python.exe analyze_crowd_video.py \
  --input "path/to/your/video.mp4" \
  --output "v:/sds/test_results/video"
```

### Generate Synthetic Video
```powershell
# Generate 10-second high-density video
V:/sds/venv/Scripts/python.exe generate_crowd_video.py \
  --output "v:/sds/test_results/video/my_video.mp4" \
  --duration 10 \
  --fps 30

# Generate 30-second video (longer)
V:/sds/venv/Scripts/python.exe generate_crowd_video.py \
  --output "v:/sds/test_results/video/long_video.mp4" \
  --duration 30 \
  --fps 30

# Generate high-resolution video (2K)
V:/sds/venv/Scripts/python.exe generate_crowd_video.py \
  --output "v:/sds/test_results/video/hd_video.mp4" \
  --duration 15 \
  --fps 60
```

### Video Processing Pipeline
```powershell
# Complete analysis workflow
$input = "datasets/downloads/umn_crowd_activity.avi"
$output = "test_results/video"

V:/sds/venv/Scripts/python.exe analyze_crowd_video.py --input $input --output $output

# View results
Get-ChildItem "$output/*.analyzed.mp4"
Get-ChildItem "$output/*.txt"
```

---

## Image Analysis

### Analyze Image (via Dashboard)
```
1. Open http://localhost:8501
2. Select "🖼️ Image Analysis" from sidebar
3. Click "Select Image" and upload JPG/PNG
4. Choose features (Detection, Density, Tracking)
5. Click "🔍 Analyze Image"
6. View results side-by-side
```

### Batch Image Analysis (Manual)
```powershell
# Create script to analyze multiple images
$imagePath = "path/to/image.jpg"

V:/sds/venv/Scripts/python.exe << 'EOF'
import cv2
import numpy as np
from src.detection.detector import PersonDetector
import yaml

# Load config
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

# Load detector
detector = PersonDetector(config['detection'])

# Read image
img = cv2.imread('$imagePath')

# Detect persons
detections = detector(img)

print(f"Detected {len(detections)} persons")
for det in detections:
    print(f"  - Confidence: {det[4]:.2f}")
EOF
```

---

## Configuration

### Modify Config File
```powershell
# Edit configuration
notepad config/config.yaml
```

### Key Configuration Options
```yaml
# Device selection
detection:
  device: "cuda"  # or "cpu"

# Detection sensitivity
detection:
  confidence: 0.5  # 0.3-0.7 recommended

# Tracking parameters
tracking:
  max_age: 30       # frames before track dies
  n_init: 3         # frames to confirm track

# Density thresholds
density:
  thresholds:
    low: 5
    medium: 15
    high: 30
    critical: 50
```

### Switch Between CPU/GPU
```powershell
# Edit config to use GPU
(Get-Content config/config.yaml) -replace 'device: "cpu"', 'device: "cuda"' | Set-Content config/config.yaml

# Verify change
Select-String "device:" config/config.yaml
```

---

## Data & Models

### Available Models
```powershell
# Check installed models
Get-ChildItem yolov8*.pt | Select-Object Name, Length

# Model sizes
# yolov8n.pt - 3.2 MB (Nano, fast)
# yolov8s.pt - 22 MB (Small, accurate)
```

### Download Datasets
```powershell
# Automatic dataset download
V:/sds/venv/Scripts/python.exe datasets/download_auto.py

# Download UMN manually
$url = "http://mha.cs.umn.edu/Movies/Crowd-Activity-All.avi"
$out = "datasets/downloads/umn_crowd_activity.avi"
Invoke-WebRequest -Uri $url -OutFile $out

# List downloaded data
Get-ChildItem datasets/downloads/
```

### Data Locations
```
datasets/
├── downloads/              # Downloaded videos
│   └── umn_crowd_activity.avi
├── custom/                 # Custom images/videos
└── DOWNLOAD_GUIDE.md       # Download instructions

test_results/
├── video/                  # Video analysis outputs
│   ├── high_density_crowd.mp4
│   ├── high_density_crowd_analyzed.mp4
│   └── umn_crowd_activity_analyzed.mp4
└── README.md
```

---

## Performance Testing

### Benchmark CPU vs GPU
```powershell
# CPU Performance Test
Write-Host "Testing CPU..."
$sw = [System.Diagnostics.Stopwatch]::StartNew()

V:/sds/venv/Scripts/python.exe analyze_crowd_video.py `
  --input "datasets/downloads/umn_crowd_activity.avi" `
  --output "test_results/video"

$sw.Stop()
Write-Host "CPU Time: $($sw.Elapsed.TotalMinutes) minutes"

# GPU Performance Test (after enabling CUDA)
Write-Host "Testing GPU..."
# Change config to device: "cuda"
# Then run same command above
```

### Memory Usage Monitoring
```powershell
# Monitor while processing
Get-Process python | Select-Object WorkingSet64, @{N='Memory';E={"{0:N0} MB" -f ($_.WorkingSet64/1MB)}}

# GPU Memory (if CUDA available)
nvidia-smi  # Shows GPU memory usage in real-time
```

---

## File Management

### View Project Structure
```powershell
# Show main files (exclude venv)
Get-ChildItem -Exclude venv | Format-List

# Show all files with sizes
Get-ChildItem -Recurse -Exclude venv | 
  Where-Object {!$_.PSIsContainer} | 
  Select-Object Name, @{N='Size';E={"{0:N0} MB" -f ($_.Length/1MB)}}
```

### Cleanup Results
```powershell
# Remove old analysis results
Remove-Item -Recurse "test_results/video/*.analyzed.mp4" -Force

# Clean temporary files
Remove-Item -Recurse "test_results/video/temp*" -ErrorAction SilentlyContinue

# View stored results
Get-ChildItem "test_results/video/" | Where-Object {$_.Extension -eq ".mp4"}
```

---

## Troubleshooting

### CUDA/GPU Issues
```powershell
# Check NVIDIA driver
nvidia-smi

# Check CUDA in Python
V:/sds/venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"

# Reinstall CUDA-enabled PyTorch
V:/sds/venv/Scripts/pip.exe uninstall torch torchvision -y
V:/sds/venv/Scripts/pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Video Format Issues
```powershell
# Check video codec
ffprobe "video.mp4"  # if ffmpeg installed

# Convert video format
ffmpeg -i input.avi -c:v libx264 -c:a aac output.mp4

# Check supported formats
Write-Host "Supported: MP4, AVI, MOV, MKV"
```

### Memory Issues
```powershell
# Reduce max frames processed
V:/sds/venv/Scripts/python.exe analyze_crowd_video.py `
  --input "video.mp4" --output "results" --max_frames 100

# Enable frame skipping
# Edit analyze_crowd_video.py and set: skip_frames=2
```

### Dashboard Won't Start
```powershell
# Kill any existing streamlit process
Get-Process streamlit -ErrorAction SilentlyContinue | Stop-Process -Force

# Clear streamlit cache
Remove-Item -Recurse "$env:USERPROFILE\.streamlit" -ErrorAction SilentlyContinue

# Try again
V:/sds/venv/Scripts/streamlit.exe run dashboard_simple.py
```

### Module Import Errors
```powershell
# Reinstall requirements
V:/sds/venv/Scripts/pip.exe install -r requirements.txt --force-reinstall

# Check specific module
V:/sds/venv/Scripts/python.exe -c "from src.detection.detector import PersonDetector; print('OK')"
```

---

## Advanced Usage

### Custom Pipeline
```powershell
# Create custom analysis script
$scriptPath = "custom_analysis.py"

@'
import cv2
from src.detection.detector import PersonDetector
from src.tracking.tracker import PersonTracker
import yaml

# Load config
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

# Load models
detector = PersonDetector(config['detection'])
tracker = PersonTracker(config['tracking'])

# Process video
cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect
    detections = detector(frame)
    
    # Track
    tracks = tracker(frame, detections)
    
    # Your custom logic here
    print(f"Frame: {cap.get(1):.0f}, Detections: {len(detections)}, Tracks: {len(tracks)}")
'@ | Out-File $scriptPath

V:/sds/venv/Scripts/python.exe $scriptPath
```

### Batch Processing
```powershell
# Process all videos in a folder
$videoFolder = "datasets/custom"
$outputFolder = "test_results/batch"

Get-ChildItem "$videoFolder/*.mp4" | ForEach-Object {
    Write-Host "Processing: $($_.Name)"
    V:/sds/venv/Scripts/python.exe analyze_crowd_video.py `
      --input $_.FullName `
      --output $outputFolder
}
```

---

## Quick Reference

### Essential Commands
| Task | Command |
|------|---------|
| Start Dashboard | `streamlit run dashboard_simple.py` |
| Analyze Video | `python analyze_crowd_video.py --input video.mp4 --output results` |
| Generate Video | `python generate_crowd_video.py --output video.mp4` |
| Check GPU | `nvidia-smi` |
| List Results | `ls test_results/video/*.mp4` |
| Install Deps | `pip install -r requirements.txt` |

### Environment Variables
```powershell
# Set Python path
$env:PYTHONPATH = "v:\sds"

# Set CUDA device
$env:CUDA_VISIBLE_DEVICES = "0"

# Disable GPU
$env:CUDA_VISIBLE_DEVICES = ""
```

---

## Support

**Dashboard**: http://localhost:8501
**Project Root**: v:\sds
**Config File**: v:\sds\config\config.yaml
**Results**: v:\sds\test_results\video

Last Updated: January 11, 2026

