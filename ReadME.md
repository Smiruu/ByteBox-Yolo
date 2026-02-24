# 🍚 YOLOv8 Rice Detection with Webcam

This project uses a YOLOv8 model trained to detect rice using a webcam feed. When rice is detected with high confidence, the script saves a snapshot and throttles detection to once every 5 minutes.

---

## 🔧 Requirements

- Windows OS
- Python 3.10 (installed via Anaconda)
- GPU with CUDA 11.8 support
- Anaconda

---

## 📦 Conda Environment Setup

1. **Install [Anaconda](https://www.anaconda.com/download)** if you don't already have it.

2. **Create the environment**:
   ```bash
   conda create -n yolov8 python=3.10
   conda activate yolov8

3. **Run this command**:
   ```bash
   pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
   pip install ultralytics opencv-python pandas matplotlib numpy scipy requests tqdm py-cpuinfo fsspec jinja2

**To run model**:

Activate environment using "conda activate ENV_NAME", then change the directory of the model on webcam_detect.py then run python webcam_detect.py