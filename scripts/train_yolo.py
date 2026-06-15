"""
Train YOLOv8 using Python (same as CLI command)
"""

import os
from ultralytics import YOLO

# ---------------- PATHS ----------------
ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "configs", "data.yaml")
OUT  = os.path.join(ROOT, "runs")   # same as default YOLO

def main():
    # Load model
    model = YOLO("yolov8n.pt")

    # Train (same as your CLI)
    model.train(
        data=DATA,
        epochs=20,
        imgsz=416,
        batch=16,
        device="cpu",   # ⚠️ IMPORTANT: tu CPU use kar raha hai
        name="train",   # same folder name
        project=OUT
    )

    # Validation (optional but useful)
    metrics = model.val(data=DATA)

    print("\n🔥 FINAL RESULTS")
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

if __name__ == "__main__":
    main()