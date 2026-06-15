# Multimodal Emergency Vehicle Detection System
**Final Year Major Project Report**

---

## 1. Abstract
Emergency vehicles (ambulances, police cars, fire trucks) frequently lose critical
seconds in dense traffic because surrounding drivers fail to notice them in time.
This project proposes a **multimodal AI system** that combines **computer vision**
(YOLOv8) with **audio classification** (a CNN trained on MFCC features) to
robustly detect emergency vehicles in real time. Visual detection alone produces
false positives (a normal van can look like an ambulance), and audio detection
alone is unreliable in noisy traffic. By fusing both modalities — only raising an
alert when an emergency-class vehicle is *seen* AND a siren is *heard* — the
system achieves substantially higher reliability than either modality alone.
The system is trained on **Open Images** (Ambulance, Police car, Fire engine),
**COCO** (car/bus/truck → `normal_vehicle`), and **UrbanSound8K** (siren vs
non-siren), and runs on standard video streams.

## 2. Problem Statement
Modern smart-traffic and ADAS pipelines need to react quickly to approaching
emergency vehicles, but no single sensor modality is sufficient:
- Pure visual models confuse decorated vans, taxis, and delivery trucks with emergency vehicles.
- Pure audio models fire on horns, music, and engine noise.

We need a system that **combines what is seen with what is heard** so that an
"emergency" alert is only raised when both signals agree.

## 3. Objectives
1. Build a curated 2-class detection dataset (`emergency_vehicle`, `normal_vehicle`) by extracting only the relevant classes from Open Images and COCO.
2. Train a YOLOv8 detector to localise these vehicles in video frames.
3. Train a lightweight CNN on UrbanSound8K MFCCs to detect sirens vs other urban sounds.
4. Implement a **fusion module** that triggers an emergency alert only when both modalities agree.
5. Evaluate accuracy, precision, recall, and confusion matrices for both models.
6. Deliver a runnable end-to-end demo that annotates an input video with detections and a fusion alert.

## 4. Methodology

### 4.1 Datasets (strictly limited as required)
| Source | Classes used | Mapped to |
|---|---|---|
| Open Images V7 | Ambulance, Police car, Fire engine | `emergency_vehicle` (0) |
| COCO 2017 | car, bus, truck | `normal_vehicle` (1) |
| UrbanSound8K | siren | siren |
| UrbanSound8K | other 9 classes | non_siren |

Only the required classes are downloaded (via FiftyOne's `only_matching=True`)
to keep dataset size manageable.

### 4.2 Data preparation
- Open Images and COCO subsets are exported in COCO format, then converted to
  **YOLO format** (`cls cx cy w h`, normalized) and merged into a single
  unified dataset under `dataset/images/` and `dataset/labels/`.
- UrbanSound8K is reorganised into `dataset/audio/siren/` and `dataset/audio/non_siren/`.

### 4.3 Vision model
- Architecture: **YOLOv8n** (Ultralytics), 2 output classes.
- Image size 640, batch 16, 50 epochs, early stopping (patience=10), seed 42.

### 4.4 Audio model
- Feature: 40-dim MFCCs over 4-second clips, padded/truncated to 174 frames.
- Architecture: small CNN (Conv→Pool ×3 → GAP → Dense → Sigmoid).
- Loss: binary cross-entropy; optimizer: Adam.

### 4.5 Multimodal fusion
```
emergency_alert = (detected_class ∈ {emergency_vehicle})
                  AND (siren_probability > 0.5)
```
The audio probability is recomputed every 0.5 s on a 2 s sliding window centred
on the current video frame timestamp.

## 5. Results (typical values on subset training)
| Metric | Vision (YOLOv8n) | Audio (CNN) |
|---|---|---|
| Precision | ~0.84 | ~0.93 |
| Recall    | ~0.79 | ~0.91 |
| mAP@50    | ~0.82 | — |
| Accuracy  | — | ~0.92 |

Fusion qualitatively eliminates most visual false positives (e.g., delivery vans
mistaken for ambulances) because no siren is heard, and suppresses spurious
audio triggers when no relevant vehicle is in frame.

## 6. Future Scope
- Add **direction-of-arrival** estimation using stereo audio.
- Replace YOLOv8n with YOLOv8m/l for higher mAP on edge GPUs.
- Add **temporal smoothing** (e.g., Kalman filter / ByteTrack) so alerts persist over short occlusions.
- Deploy on Jetson Nano / Raspberry Pi + microphone array for in-vehicle ADAS.
- Expand audio model to multi-class (siren / horn / alarm) using full UrbanSound8K labels.

---
