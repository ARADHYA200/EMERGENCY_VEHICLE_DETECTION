"""
Evaluate both models and dump metrics to outputs/metrics.txt.
"""
import os, json
from ultralytics import YOLO

ROOT = os.path.join(os.path.dirname(__file__), "..")
W = os.path.join(ROOT, "runs", "detect", "train", "weights", "best.pt")
DATA = os.path.join(ROOT, "configs", "data.yaml")
OUT  = os.path.join(ROOT, "outputs", "metrics.txt")

lines = []

# --- YOLO eval ---
if os.path.exists(W):
    m = YOLO(W).val(data=DATA)
    lines.append(f"YOLOv8 mAP50    : {m.box.map50:.4f}")
    lines.append(f"YOLOv8 mAP50-95 : {m.box.map:.4f}")
    lines.append(f"YOLOv8 Precision: {m.box.mp:.4f}")
    lines.append(f"YOLOv8 Recall   : {m.box.mr:.4f}")
else:
    lines.append("YOLO weights not found — train first.")

# --- Audio eval is run inside train_audio.py and saved as plots ---
lines.append("Audio CNN: see outputs/audio_confusion_matrix.png and outputs/audio_training_curves.png")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f: f.write("\n".join(lines))
print("\n".join(lines))
