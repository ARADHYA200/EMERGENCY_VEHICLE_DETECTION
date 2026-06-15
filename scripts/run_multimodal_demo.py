"""
Multimodal demo:
  - Reads a video (with audio)
  - Runs YOLOv8 on each frame
  - Runs siren CNN on a sliding audio window aligned with the current frame
  - Fusion rule:
        IF (vehicle in {ambulance, police, fire_truck}) AND (siren_prob > THRESH)
        => overlay "🚨 EMERGENCY VEHICLE DETECTED"
  - Writes annotated video to outputs/demo_annotated.mp4

Usage:
  python scripts/run_multimodal_demo.py --source path/to/video.mp4
"""
import os, argparse, tempfile, numpy as np, cv2, librosa
from ultralytics import YOLO
import tensorflow as tf
from moviepy.editor import VideoFileClip

ROOT = os.path.join(os.path.dirname(__file__), "..")
YOLO_W  = os.path.join(ROOT, "runs", "detect", "train", "weights", "best.pt")
AUDIO_M = os.path.join(ROOT, "models", "siren_cnn.h5")
OUT_DIR = os.path.join(ROOT, "outputs"); os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = ["emergency_vehicle", "normal_vehicle"]
EMERGENCY_IDS = {0}
COLORS = {0:(0,0,255), 1:(0,255,0)}  # BGR
SR = 22050; N_MFCC = 40; MAX_LEN = 174
WINDOW_SEC = 2.0
SIREN_THRESH = 0.5

def mfcc_window(audio, sr, t_center):
    half = WINDOW_SEC / 2
    s = int(max(0, (t_center - half) * sr))
    e = int(min(len(audio), (t_center + half) * sr))
    seg = audio[s:e]
    needed = int(WINDOW_SEC * sr)
    if len(seg) < needed:
        seg = np.pad(seg, (0, needed - len(seg)))
    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=N_MFCC)
    if mfcc.shape[1] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0,0),(0, MAX_LEN - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :MAX_LEN]
    return mfcc[None, ..., None].astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="input video path")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "demo_annotated.mp4"))
    args = ap.parse_args()

    assert os.path.exists(YOLO_W),  f"Missing {YOLO_W}. Train YOLO first."
    assert os.path.exists(AUDIO_M), f"Missing {AUDIO_M}. Train audio CNN first."

    # extract audio track
    audio = np.zeros(1, dtype=np.float32)
    tmp_wav = None
    try:
        with VideoFileClip(args.source) as clip:
            if clip.audio is not None:
                tmp_wav = tempfile.mktemp(suffix=".wav")
                clip.audio.write_audiofile(tmp_wav, fps=SR, logger=None, verbose=False)
                audio, _ = librosa.load(tmp_wav, sr=SR, mono=True)
            else:
                print("[WARN] No audio track found in the video; siren detection will remain disabled.")
    except Exception as exc:
        print(f"[WARN] Audio extraction failed: {exc}. Siren detection will remain disabled.")
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except Exception:
                pass

    yolo = YOLO(YOLO_W)
    audio_model = tf.keras.models.load_model(AUDIO_M)

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    f_idx = 0
    cached_siren = (0.0, -1)  # (prob, frame_at_which_computed)
    while True:
        ok, frame = cap.read()
        if not ok: break

        # vision
        res = yolo.predict(frame, verbose=False, imgsz=640, conf=0.35)[0]
        emergency_seen = False
        for box in res.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            color = COLORS.get(cls, (200,200,200))
            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
            label = f"{CLASS_NAMES[cls]} {conf:.2f}"
            cv2.putText(frame, label, (x1, max(20,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if cls in EMERGENCY_IDS:
                emergency_seen = True

        # audio (recompute every ~0.5s to save time)
        t = f_idx / fps
        if f_idx - cached_siren[1] > int(fps * 0.5):
            prob = float(audio_model.predict(mfcc_window(audio, SR, t), verbose=0)[0][0])
            cached_siren = (prob, f_idx)
        siren_prob = cached_siren[0]
        siren_on = siren_prob > SIREN_THRESH

        # HUD
        hud = f"Siren: {siren_prob:.2f} ({'YES' if siren_on else 'no'})"
        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # fusion
        if emergency_seen and siren_on:
            cv2.rectangle(frame, (0,0), (W, 60), (0,0,255), -1)
            cv2.putText(frame, "EMERGENCY VEHICLE DETECTED", (20, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3)

        writer.write(frame)
        f_idx += 1

    cap.release(); writer.release()
    print(f"Saved annotated video -> {args.out}")

if __name__ == "__main__":
    main()
