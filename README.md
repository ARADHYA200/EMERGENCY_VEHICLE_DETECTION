# Multimodal Emergency Vehicle Detection System

End-to-end final-year major project that detects emergency vehicles using:
- **Vision**: YOLOv8 trained on Open Images (Ambulance, Police car, Fire engine) + COCO (car/bus/truck → `normal_vehicle`).
- **Audio**: CNN trained on UrbanSound8K (siren vs non-siren) using MFCC features.
- **Fusion**: An emergency alert is raised only when an emergency vehicle is seen AND a siren is heard.

## Folder structure

```
emergency_vehicle_detection/
├── dataset/
│   ├── images/{train,val}/        # YOLO images
│   ├── labels/{train,val}/        # YOLO .txt labels
│   └── audio/{siren,non_siren}/   # UrbanSound8K wav clips
├── models/                        # trained weights (yolo .pt, audio .h5)
├── scripts/                       # all runnable pipeline scripts
├── configs/                       # data.yaml for YOLO
├── outputs/                       # annotated videos, plots, metrics
└── report/                        # report.md (abstract..future scope)
```

## Class mapping (final, unified)
```
0 → emergency_vehicle
1 → normal_vehicle
```

## Setup

```bash
python -m venv venv && source venv/bin/activate   # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

Requires Python 3.10+ and (recommended) an NVIDIA GPU with CUDA for training.

## Quick Start (Models Already Trained)

```bash
# Install dependencies
pip install -r requirements.txt

# Run multimodal detection on webcam
python scripts/multimodal_system.py --webcam

# Or on a video
python scripts/multimodal_system.py --video dataset/videos/traffic.mp4

# Evaluate models
python scripts/evaluate.py
```

## Full Pipeline (If Retraining Needed)
# 1. Download only required Open Images classes (Ambulance, Police car, Fire engine)
python scripts/download_openimages.py

# 2. Download only required COCO classes (car, bus, truck) and remap to normal_vehicle
python scripts/download_coco.py

# 3. Download UrbanSound8K and split into siren / non_siren
python scripts/download_urbansound.py

# 4. Convert + merge all annotations into one YOLO dataset
python scripts/prepare_yolo_dataset.py

# 5. Train YOLOv8 vehicle detector
python scripts/train_yolo.py

# 6. Train audio siren CNN
python scripts/train_audio.py

# 7. Evaluate both models
python scripts/evaluate.py

# 8. Run the multimodal demo on a video (with audio track)
python scripts/run_multimodal_demo.py --source outputs/sample.mp4
```

See `report/report.md` for the submission write-up.

## Web UI (Streamlit)

After training both models, launch the browser UI:

```bash
streamlit run app/streamlit_app.py
```

Then upload a video, adjust the YOLO confidence and siren-probability thresholds
in the sidebar, watch live frame-by-frame detection with the fusion alert
banner, and download the annotated MP4.
