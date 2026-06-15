#!/usr/bin/env python3
"""
Multimodal Emergency Vehicle Detection System (IMPROVED)
========================================================

Combines:
  1. YOLOv8 visual detection (2-class: emergency_vehicle, normal_vehicle)
  2. CNN audio siren detection (binary: siren vs non_siren)
  3. Fusion logic for robust detection

Fusion Rules:
  ✅ EMERGENCY CONFIRMED: Emergency vehicle + Siren detected
  ⚠️  SUSPECTED EMERGENCY: Emergency vehicle only (no siren)
  🔊 ALERT: Siren detected (no emergency vehicle in frame)
  ✅ NORMAL: Regular traffic (no alerts)

Usage:
  python scripts/multimodal_system.py --webcam
  python scripts/multimodal_system.py --video path/to/video.mp4

Author: AI Assistant
Date: 2026-04-25
Version: 2.0 (Improved & Audited)
"""

import cv2
import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
from audio_inference import SirenDetector

# ============================================================================
# CONFIG
# ============================================================================

# Model paths (relative to project root)
YOLO_WEIGHTS = Path("runs/detect/train/weights/best.pt")
AUDIO_MODEL = Path("models/siren_cnn.h5")

# Class configuration (BINARY - 2 classes only)
CLASS_NAMES = {
    0: "Emergency Vehicle",
    1: "Normal Vehicle"
}

CLASS_COLORS = {
    0: (0, 0, 255),     # Red - Emergency
    1: (180, 180, 180)  # Gray - Normal
}

EMERGENCY_ID = 0

# Detection thresholds
YOLO_CONF = 0.4        # YOLO confidence threshold
YOLO_IOU = 0.45        # YOLO NMS IOU threshold
YOLO_SIZE = 640        # Input image size

SIREN_THRESHOLD = 0.5  # Audio probability threshold

# Display settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2
BOX_THICKNESS = 2

# ============================================================================
# MULTIMODAL SYSTEM CLASS
# ============================================================================

class MultimodalSystem:
    """Main system class handling vision + audio fusion."""
    
    def __init__(self):
        """Initialize YOLO and audio models."""
        try:
            print("[INIT] Loading YOLO model...")
            if not YOLO_WEIGHTS.exists():
                raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")
            self.yolo_model = YOLO(str(YOLO_WEIGHTS))
            print(f"✅ YOLO loaded: {YOLO_WEIGHTS}")
            
        except Exception as e:
            print(f"❌ YOLO loading failed: {e}")
            sys.exit(1)
        
        try:
            print("[INIT] Loading audio siren detector...")
            self.audio_detector = SirenDetector()
            self.audio_detector.start_realtime()
            print("✅ Audio detector started")
            
        except Exception as e:
            print(f"⚠️  Audio loading failed: {e}")
            print("   Continuing with vision-only mode...")
            self.audio_detector = None

    def fuse_detections(self, emergency_detected, siren_probability):
        """
        Fusion logic combining vision + audio.
        
        Args:
            emergency_detected (bool): Is emergency vehicle detected?
            siren_probability (float): Audio siren probability (0-1)
            
        Returns:
            tuple: (status_text, color_bgr)
        """
        siren_active = siren_probability >= SIREN_THRESHOLD
        
        if emergency_detected and siren_active:
            return "🚨 EMERGENCY CONFIRMED 🚨", (0, 0, 255)
        elif emergency_detected and not siren_active:
            return "⚠️  SUSPECTED EMERGENCY ⚠️", (0, 165, 255)
        elif not emergency_detected and siren_active:
            return "🔊 ALERT: UNIDENTIFIED SIREN 🔊", (255, 255, 0)
        else:
            return "✅ NORMAL TRAFFIC ✅", (0, 255, 0)

    def draw_detections(self, frame, results):
        """
        Draw YOLO detections on frame.
        
        Args:
            frame: Input frame (BGR)
            results: YOLO detection results
            
        Returns:
            tuple: (annotated_frame, emergency_found)
        """
        emergency_found = False
        
        for box in results[0].boxes:
            cls_id = int(box.cls)
            confidence = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            label = f"{CLASS_NAMES.get(cls_id, 'Unknown')} {confidence:.2f}"
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
            
            # Draw label background
            text_size = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)[0]
            cv2.rectangle(frame, 
                         (x1, y1 - text_size[1] - 5),
                         (x1 + text_size[0] + 5, y1),
                         color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                       FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
            
            if cls_id == EMERGENCY_ID:
                emergency_found = True
        
        return frame, emergency_found

    def draw_status_banner(self, frame, status_text, status_color):
        """
        Draw status banner at top of frame.
        
        Args:
            frame: Input frame
            status_text: Status message
            status_color: Color (BGR tuple)
            
        Returns:
            Annotated frame
        """
        # Draw background rectangle
        height = 60
        cv2.rectangle(frame, (0, 0), (frame.shape[1], height), (0, 0, 0), -1)
        
        # Draw colored border
        cv2.rectangle(frame, (0, 0), (frame.shape[1], height), status_color, 3)
        
        # Draw text
        cv2.putText(frame, status_text, (20, 40),
                   FONT, FONT_SCALE, status_color, FONT_THICKNESS)
        
        # Draw audio indicator (if audio is enabled)
        if self.audio_detector:
            siren_prob = getattr(self.audio_detector, '_current_prob', 0.0)
            audio_text = f"Siren Prob: {siren_prob:.2f}"
            text_size = cv2.getTextSize(audio_text, FONT, 0.6, 1)[0]
            cv2.putText(frame, audio_text,
                       (frame.shape[1] - text_size[0] - 20, 40),
                       FONT, 0.6, (200, 200, 200), 1)
        
        return frame

    def process_frame(self, frame):
        """
        Process single frame through entire pipeline.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Annotated frame
        """
        # YOLO detection
        results = self.yolo_model(frame, imgsz=YOLO_SIZE, conf=YOLO_CONF, 
                                  iou=YOLO_IOU, verbose=False)
        frame, emergency_detected = self.draw_detections(frame, results)
        
        # Audio detection
        siren_probability = 0.0
        if self.audio_detector:
            siren_probability = getattr(self.audio_detector, '_current_prob', 0.0)
        
        # Fusion
        status_text, status_color = self.fuse_detections(emergency_detected, 
                                                         siren_probability)
        
        # Draw status
        frame = self.draw_status_banner(frame, status_text, status_color)
        
        return frame

    def run(self, source):
        """
        Main inference loop.
        
        Args:
            source: Video file path (str) or webcam index (int)
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"❌ Failed to open source: {source}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"▶️  Starting inference...")
        print(f"   Resolution: {width}x{height} @ {fps:.1f} FPS")
        print(f"   Press 'Q' to exit")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            try:
                frame = self.process_frame(frame)
            except Exception as e:
                print(f"⚠️  Frame processing error: {e}")
                continue
            
            # Display
            cv2.imshow("Multimodal Emergency Vehicle Detection", frame)
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 Exiting...")
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"✓ Processed {frame_count} frames")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Done! Processed {frame_count} frames total")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Emergency Vehicle Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python multimodal_system.py --webcam
  python multimodal_system.py --video video.mp4
        """
    )
    
    parser.add_argument("--video", type=str, 
                       help="Path to video file")
    parser.add_argument("--webcam", action="store_true",
                       help="Use webcam (index 0)")
    
    args = parser.parse_args()
    
    # Determine source
    if args.webcam:
        source = 0
        print("📷 Using webcam...")
    elif args.video:
        source = args.video
        if not Path(source).exists():
            print(f"❌ Video file not found: {source}")
            sys.exit(1)
        print(f"🎬 Using video: {source}")
    else:
        parser.print_help()
        sys.exit(1)
    
    # Run system
    try:
        system = MultimodalSystem()
        system.run(source)
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
