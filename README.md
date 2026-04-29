# Malpractice Detection System

A real-time exam malpractice detection system using a deep learning model deployed on a Raspberry Pi with a webcam for live monitoring.

## How It Works

The system captures live video from a USB webcam, analyzes 16-frame clips using a lightweight MobileNetV2-based model, and flags suspicious behavior when malpractice is detected for more than 8 consecutive seconds. When triggered, it saves a video clip (5 seconds before + 5 seconds after the alert) to the `evidence/videos/` folder and plays an audio alert.

## Project Structure

```
├── model.ipynb          # Model training notebook
├── inference.py         # Live detection script (runs on Raspberry Pi)
├── app.py               # inference server
├── best_model_v4.pth    # Trained model weights
└── requirements.txt     # Python dependencies
```

## Hardware Setup

- **Device:** Raspberry Pi (any model with USB support)
- **Camera:** USB Webcam connected to the Pi
- **Audio (optional):** Place an `alert.wav` file in the project root for sound alerts

## Installation

```bash
git clone https://github.com/ThangakumarC/malpractice-detection-system.git
cd malpractice-detection-system
pip install -r requirements.txt
```

## Running Live Detection

```bash
python inference.py
```

Press **Q** to quit the detection window.

## Detection Settings

| Parameter | Value | Description |
|---|---|---|
| Confidence threshold | 0.6 | Minimum score to flag as suspicious |
| Min duration | 8 seconds | Time before raising an alert |
| Cooldown | 60 seconds | Gap between consecutive alerts |
| Pre/Post buffer | 5 seconds each | Context saved around alert |

## Output

Flagged incidents are saved automatically as `.avi` video files under `evidence/videos/` with a timestamp filename (e.g., `20250429_143022.avi`).

## Dependencies

- PyTorch & TorchVision
- OpenCV
- FastAPI + Uvicorn
- NumPy, Pillow
