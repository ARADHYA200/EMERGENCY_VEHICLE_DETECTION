#!/usr/bin/env python3
"""
Real-time / video-file emergency vehicle detection using trained YOLOv8.
Usage:
  python video_detection.py --source 0            # webcam
  python video_detection.py --source video.mp4    # video file
  python video_detection.py --source image.jpg    # single image
"""

import cv2
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ─── CONFIG ────────────────────────────────────────────────────────────────
WEIGHTS    = "runs/detect/train/weights/best.pt"
CONF       = 0.40
IOU        = 0.45
IMG_SIZE   = 640
SAVE_DIR   = Path("outputs/detections")

CLASS_NAMES  = {0: "Emergency Vehicle", 1: "Normal Vehicle"}
CLASS_COLORS = {
    0: (0,   0,  255),   # Emergency → red
    1: (180, 180, 180),  # Normal → grey
}
EMERGENCY_IDS = {0}
# ───────────────────────────────────────────────────────────────────────────

def draw_boxes(frame, results, siren_active: bool = False):
    emergency_found = False
    for box in results[0].boxes:
        cls_id = int(box.cls)
        conf   = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = CLASS_COLORS.get(cls_id, (200, 200, 200))
        label = f"{CLASS_NAMES.get(cls_id, 'Unknown')} {conf:.2f}"

        # Draw filled semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

        # Draw border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        if cls_id in EMERGENCY_IDS:
            emergency_found = True

    return frame, emergency_found

def draw_hud(frame, emergency: bool, siren: bool):
    h, w = frame.shape[:2]
    # Header bar
    cv2.rectangle(frame, (0, 0), (w, 48), (15, 15, 15), -1)
    cv2.putText(frame, "MULTIMODAL EMERGENCY VEHICLE DETECTION", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (220, 220, 220), 2)

    # Status bar at bottom
    cv2.rectangle(frame, (0, h - 52), (w, h), (15, 15, 15), -1)

    siren_color  = (0, 255, 120) if siren     else (100, 100, 100)
    detect_color = (0, 60, 255)  if emergency else (100, 100, 100)

    cv2.putText(frame, f"SIREN:  {'DETECTED' if siren else 'NONE'}",
                (10, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, siren_color, 2)

    if emergency and siren:
        alert = "  EMERGENCY VEHICLE DETECTED  "
        (tw, _), _ = cv2.getTextSize(alert, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        ax = (w - tw) // 2
        cv2.rectangle(frame, (ax - 8, h - 45), (ax + tw + 8, h - 4), (0, 0, 200), -1)
        cv2.putText(frame, alert, (ax, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    else:
        cv2.putText(frame, f"VEHICLE: {'EMERGENCY' if emergency else 'NORMAL'}",
                    (w // 2 - 100, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, detect_color, 2)

    return frame

def run_video(source, siren_active: bool = False, save: bool = True):
    model = YOLO(WEIGHTS)
    model.conf = CONF
    model.iou  = IOU

    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if save:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        out_path = SAVE_DIR / "output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (fw, fh))

    print("[INFO] Press 'q' to quit, 's' to toggle siren simulation.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=IMG_SIZE, verbose=False)
        frame, emergency = draw_boxes(frame, results, siren_active)
        frame = draw_hud(frame, emergency, siren_active)

        if writer:
            writer.write(frame)

        cv2.imshow("Emergency Vehicle Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            siren_active = not siren_active
            print(f"[SIREN] {'ON' if siren_active else 'OFF'}")

    cap.release()
    if writer:
        writer.release()
        print(f"[SAVED] Output → {SAVE_DIR / 'output.mp4'}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Video path or webcam index")
    parser.add_argument("--siren",  action="store_true", help="Simulate siren active")
    parser.add_argument("--nosave", action="store_true", help="Do not save output")
    args = parser.parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    run_video(src, siren_active=args.siren, save=not args.nosave)
