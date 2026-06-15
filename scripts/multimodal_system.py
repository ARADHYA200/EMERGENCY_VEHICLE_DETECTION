"""
Multimodal Fusion System (BINARY)
0 = emergency_vehicle
1 = normal_vehicle
"""

import cv2
import argparse
from ultralytics import YOLO
from audio_inference import SirenDetector

# ================= CONFIG =================
YOLO_WEIGHTS = "runs/detect/train/weights/best.pt"

CLASS_NAMES = {
    0: "Emergency Vehicle",
    1: "Normal Vehicle"
}

CLASS_COLORS = {
    0: (0, 0, 255),     # Red
    1: (180, 180, 180)  # Gray
}

EMERGENCY_ID = 0
# ==========================================


class MultimodalSystem:
    def __init__(self):
        print("[INIT] Loading YOLO...")
        self.model = YOLO(YOLO_WEIGHTS)

        print("[INIT] Loading Audio...")
        self.audio = SirenDetector()
        self.audio.start_realtime()

    # ================= FUSION =================
    def fuse(self, emergency_detected, siren):
        if emergency_detected and siren:
            return "EMERGENCY CONFIRMED 🚨", (0, 0, 255)
        elif emergency_detected and not siren:
            return "SUSPECTED EMERGENCY ⚠️", (0, 165, 255)
        elif not emergency_detected and siren:
            return "UNIDENTIFIED SIREN 🔊", (255, 255, 0)
        else:
            return "NORMAL TRAFFIC ✅", (0, 255, 0)

    # ================= DRAW =================
    def draw_boxes(self, frame, results):
        emergency_found = False

        for box in results[0].boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"{CLASS_NAMES.get(cls, '?')} {conf:.2f}"
            color = CLASS_COLORS.get(cls, (255, 255, 255))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if cls == EMERGENCY_ID:
                emergency_found = True

        return frame, emergency_found

    def draw_status(self, frame, text, color):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame

    # ================= RUN =================
    def run(self, source):
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print("❌ Video open nahi hua")
            return

        print("▶ Running... Press Q to exit")

        # Check if video file or webcam
        is_video_file = isinstance(source, str)
        if is_video_file:
            # Video file mode: save output
            import os
            os.makedirs('outputs', exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter('outputs/processed_video.mp4', fourcc, fps, (width, height))
            print(f"📹 Processing video, output will be saved to outputs/processed_video.mp4")
        else:
            # Webcam mode
            out = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO detection
            results = self.model(frame, imgsz=640, conf=0.4, verbose=False)

            frame, emergency = self.draw_boxes(frame, results)

            # Audio detection
            siren = self.audio.siren_active

            # Fusion
            status, color = self.fuse(emergency, siren)

            # Draw
            frame = self.draw_status(frame, status, color)

            if is_video_file:
                # Save frame to video
                out.write(frame)
            else:
                # Webcam: try to display
                try:
                    cv2.imshow("Multimodal Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error as e:
                    print(f"❌ GUI display failed: {e}")
                    print("💡 Switching to headless mode. Frames will be saved to outputs/frames/")
                    import os
                    os.makedirs('outputs/frames', exist_ok=True)
                    frame_count = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # Process frame
                        results = self.model(frame, imgsz=640, conf=0.4, verbose=False)
                        frame, emergency = self.draw_boxes(frame, results)
                        siren = self.audio.siren_active
                        status, color = self.fuse(emergency, siren)
                        frame = self.draw_status(frame, status, color)
                        cv2.imwrite(f'outputs/frames/frame_{frame_count:06d}.png', frame)
                        frame_count += 1
                        print(f"📸 Saved frame {frame_count}")
                    break

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()


# ================= MAIN =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str)
    parser.add_argument("--webcam", action="store_true")

    args = parser.parse_args()

    if args.webcam:
        source = 0
    else:
        source = args.video

    system = MultimodalSystem()
    system.run(source)


# #!/usr/bin/env python3
# """
# Multimodal Fusion System
# Combines YOLOv8 visual detection + CNN siren audio detection.

# FUSION LOGIC:
#   IF (vehicle ∈ {ambulance, police, fire_truck}) AND (siren == True)
#       → EMERGENCY VEHICLE CONFIRMED
#   ELIF (vehicle ∈ {ambulance, police, fire_truck}) AND (siren == False)
#       → SUSPECTED EMERGENCY (visual only)
#   ELIF (vehicle == normal_vehicle) AND (siren == True)
#       → UNIDENTIFIED SIREN
#   ELSE
#       → NORMAL TRAFFIC

# Usage:
#   python multimodal_system.py --video test.mp4
#   python multimodal_system.py --webcam
# """

# import cv2
# import argparse
# import numpy as np
# import threading
# from pathlib import Path
# from ultralytics import YOLO
# from audio_inference import SirenDetector

# # ─── CONFIG ─────────────────────────────────────────────
# YOLO_WEIGHTS = "outputs/runs/emergency_vehicle_v1/weights/best.pt"
# IMG_SIZE     = 640
# CONF         = 0.40
# IOU          = 0.45
# SAVE_DIR     = Path("outputs/multimodal")

# CLASS_NAMES  = {0: "Ambulance", 1: "Police", 2: "Fire Truck", 3: "Normal Vehicle"}
# CLASS_COLORS = {
#     0: (0,   80,  255),
#     1: (255, 180,   0),
#     2: (0,   200,  50),
#     3: (180, 180, 180),
# }
# EMERGENCY_IDS = {0, 1, 2}
# # ────────────────────────────────────────────────────────

# class MultimodalSystem:
#     def __init__(self):
#         print("[INIT] Loading visual model (YOLOv8)...")
#         self.visual_model = YOLO(YOLO_WEIGHTS)

#         print("[INIT] Loading audio model (Siren CNN)...")
#         self.audio_model  = SirenDetector()
#         self.audio_model.start_realtime()   # non-blocking background thread

#         self._frame_count = 0

#     # ──────────────────────────── FUSION LOGIC ────────────────────────────
#     def fuse(self, has_emergency_vehicle: bool, siren_active: bool) -> dict:
#         """
#         Core fusion logic → returns status dict.
#         """
#         if has_emergency_vehicle and siren_active:
#             return {
#                 "status":  "EMERGENCY CONFIRMED",
#                 "level":   "critical",
#                 "color":   (0, 0, 220),
#                 "message": "EMERGENCY VEHICLE DETECTED — CLEAR PATH",
#             }
#         elif has_emergency_vehicle and not siren_active:
#             return {
#                 "status":  "SUSPECTED EMERGENCY",
#                 "level":   "warning",
#                 "color":   (0, 140, 255),
#                 "message": "Emergency vehicle (no siren) — stay alert",
#             }
#         elif not has_emergency_vehicle and siren_active:
#             return {
#                 "status":  "UNIDENTIFIED SIREN",
#                 "level":   "warning",
#                 "color":   (0, 200, 200),
#                 "message": "Siren detected — no visual match",
#             }
#         else:
#             return {
#                 "status":  "NORMAL TRAFFIC",
#                 "level":   "normal",
#                 "color":   (80, 180, 80),
#                 "message": "No emergency detected",
#             }

#     # ──────────────────────────── DRAWING ─────────────────────────────────
#     def draw_boxes(self, frame, results):
#         emergency_found = False
#         for box in results[0].boxes:
#             cls_id = int(box.cls)
#             conf   = float(box.conf)
#             x1, y1, x2, y2 = map(int, box.xyxy[0])

#             color = CLASS_COLORS.get(cls_id, (200, 200, 200))
#             label = f"{CLASS_NAMES.get(cls_id, '?')}  {conf:.0%}"

#             overlay = frame.copy()
#             cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
#             cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#             (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
#             cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
#             cv2.putText(frame, label, (x1 + 3, y1 - 4),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

#             if cls_id in EMERGENCY_IDS:
#                 emergency_found = True

#         return frame, emergency_found

#     def draw_hud(self, frame, fusion_result, siren: bool, siren_conf: float):
#         h, w = frame.shape[:2]
#         status_color = fusion_result["color"]
#         level        = fusion_result["level"]

#         # Top bar
#         cv2.rectangle(frame, (0, 0), (w, 52), (12, 12, 12), -1)
#         cv2.putText(frame, "MULTIMODAL EMERGENCY VEHICLE DETECTION SYSTEM",
#                     (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 2)

#         # Bottom fusion bar
#         bar_h = 58
#         cv2.rectangle(frame, (0, h - bar_h), (w, h),
#                       status_color if level == "critical" else (18, 18, 18), -1)

#         # Siren indicator (left)
#         s_color = (0, 255, 120) if siren else (80, 80, 80)
#         cv2.putText(frame, f"SIREN: {'ACTIVE ({:.0%})'.format(siren_conf) if siren else 'NONE'}",
#                     (12, h - bar_h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, s_color, 2)

#         # Status (center)
#         msg = fusion_result["message"]
#         (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
#         ax = (w - tw) // 2
#         txt_color = (255, 255, 255) if level == "critical" else status_color
#         cv2.putText(frame, msg, (ax, h - bar_h + 36),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, txt_color, 2)

#         # Frame counter (right)
#         cv2.putText(frame, f"FRAME: {self._frame_count:06d}",
#                     (w - 180, h - bar_h + 36),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
#         return frame

#     # ──────────────────────────── MAIN LOOP ───────────────────────────────
#     def run(self, source, save: bool = True):
#         cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
#         if not cap.isOpened():
#             raise RuntimeError(f"Cannot open: {source}")

#         fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = cap.get(cv2.CAP_PROP_FPS) or 30

#         writer = None
#         if save:
#             SAVE_DIR.mkdir(parents=True, exist_ok=True)
#             out_path = SAVE_DIR / "multimodal_output.mp4"
#             writer = cv2.VideoWriter(str(out_path),
#                                      cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
#         print("[INFO] Running multimodal system. Press 'q' to quit.")

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             self._frame_count += 1

#             # Visual detection
#             results = self.visual_model(frame, imgsz=IMG_SIZE,
#                                         conf=CONF, iou=IOU, verbose=False)
#             frame, has_emergency = self.draw_boxes(frame, results)

#             # Audio detection (live from background thread)
#             siren_active = self.audio_model.siren_active
#             _, siren_conf = (True, 0.9) if siren_active else (False, 0.0)

#             # Fusion
#             fusion = self.fuse(has_emergency, siren_active)

#             # HUD overlay
#             frame = self.draw_hud(frame, fusion, siren_active, siren_conf)

#             if writer:
#                 writer.write(frame)

#             cv2.imshow("Multimodal Emergency Detection", frame)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break

#         cap.release()
#         if writer:
#             writer.release()
#             print(f"[SAVED] {SAVE_DIR / 'multimodal_output.mp4'}")
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--video",  type=str, default=None)
#     parser.add_argument("--webcam", action="store_true")
#     parser.add_argument("--nosave", action="store_true")
#     args = parser.parse_args()

#     source = 0 if args.webcam else (args.video or 0)
#     sys_obj = MultimodalSystem()
#     sys_obj.run(source, save=not args.nosave)
