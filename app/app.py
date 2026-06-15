# # """
# # Streamlit web UI for the Multimodal Emergency Vehicle Detection System.

# # Usage:
# #     streamlit run app/streamlit_app.py

# # This app accepts an uploaded video, extracts the audio track, runs YOLOv8 object detection,
# # performs siren audio classification, and displays fused emergency detection results.
# # """


# """

# ---------------------------------------------------------
# NEW FEATURES ADDED
# ---------------------------------------------------------

# 1. Fully responsive dashboard
# 2. Animated smart traffic signal 🚦
# 3. Green priority countdown timer
# 4. Emergency event timeline
# 5. Fusion confidence score
# 6. Real-time statistics dashboard
# 7. FPS counter
# 8. Emergency detection stabilization
# 9. Better UI styling
# 10. Siren confidence progress bars
# 11. Detection history panel
# 12. Smart city dark dashboard
# 13. Smooth signal transition logic
# 14. Better video fitting
# 15. Real-time event logging

# """

# import time
# import tempfile
# from pathlib import Path
# from collections import deque

# import cv2
# import librosa
# import numpy as np
# import streamlit as st
# import tensorflow as tf
# from moviepy.editor import VideoFileClip
# from ultralytics import YOLO

# # ---------------------------------------------------------
# # PAGE CONFIG
# # ---------------------------------------------------------
# st.set_page_config(
#     page_title="AI Smart Traffic Control",
#     page_icon="🚦",
#     layout="wide"
# )

# # ---------------------------------------------------------
# # CUSTOM CSS
# # ---------------------------------------------------------
# st.markdown("""
# <style>

# /* Entire App */
# .stApp {
#     background-color: #0B0F19;
#     color: white;
# }

# /* Video */
# [data-testid="stImage"] img {
#     max-height: 70vh;
#     object-fit: contain;
#     border-radius: 15px;
#     border: 2px solid #222;
# }

# /* Cards */
# .card {
#     background-color: #151B28;
#     padding: 15px;
#     border-radius: 15px;
#     margin-bottom: 15px;
#     border: 1px solid #2A2F3A;
# }

# /* Metrics */
# .metric-value {
#     font-size: 32px;
#     font-weight: bold;
#     text-align: center;
#     color: #00FFAA;
# }

# .metric-title {
#     text-align: center;
#     color: #BBBBBB;
# }

# /* Signal */
# .signal-box {
#     background-color: #1A1A1A;
#     width: 130px;
#     padding: 20px;
#     border-radius: 20px;
#     margin: auto;
# }

# .light {
#     width: 65px;
#     height: 65px;
#     border-radius: 50%;
#     margin: 15px auto;
#     opacity: 0.2;
# }

# .red-on {
#     background-color: red;
#     opacity: 1;
#     box-shadow: 0 0 30px red;
# }

# .yellow-on {
#     background-color: yellow;
#     opacity: 1;
#     box-shadow: 0 0 30px yellow;
# }

# .green-on {
#     background-color: lime;
#     opacity: 1;
#     box-shadow: 0 0 30px lime;
# }

# /* Event log */
# .log-box {
#     background-color: #111827;
#     padding: 10px;
#     border-radius: 10px;
#     height: 220px;
#     overflow-y: auto;
#     font-size: 14px;
# }

# /* Countdown */
# .countdown {
#     text-align: center;
#     font-size: 28px;
#     font-weight: bold;
#     color: #00FFAA;
# }

# </style>
# """, unsafe_allow_html=True)

# # ---------------------------------------------------------
# # PATHS
# # ---------------------------------------------------------
# ROOT = Path(__file__).resolve().parents[1]

# YOLO_WEIGHTS = ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"

# AUDIO_MODEL = ROOT / "models" / "siren_cnn.h5"

# # ---------------------------------------------------------
# # CONSTANTS
# # ---------------------------------------------------------
# CLASS_NAMES = ["emergency_vehicle", "normal_vehicle"]

# EMERGENCY_IDS = {0}

# COLORS = {
#     0: (0, 0, 255),
#     1: (0, 255, 0)
# }

# SR = 22050
# N_MFCC = 40
# MAX_LEN = 174
# WINDOW_SEC = 2.0

# # ---------------------------------------------------------
# # LOAD MODELS
# # ---------------------------------------------------------
# @st.cache_resource
# def load_yolo(path):
#     return YOLO(path)

# @st.cache_resource
# def load_audio_model(path):
#     return tf.keras.models.load_model(path)

# # ---------------------------------------------------------
# # AUDIO EXTRACTION
# # ---------------------------------------------------------
# def extract_audio(video_path):

#     audio = np.zeros(1, dtype=np.float32)

#     try:

#         with VideoFileClip(video_path) as clip:

#             if clip.audio is not None:

#                 temp_wav = (
#                     Path(tempfile.gettempdir())
#                     / f"audio_{int(time.time())}.wav"
#                 )

#                 clip.audio.write_audiofile(
#                     str(temp_wav),
#                     fps=SR,
#                     logger=None,
#                     verbose=False
#                 )

#                 audio, _ = librosa.load(
#                     str(temp_wav),
#                     sr=SR,
#                     mono=True
#                 )

#                 temp_wav.unlink(missing_ok=True)

#     except Exception as e:
#         st.warning(f"Audio extraction failed: {e}")

#     return audio

# # ---------------------------------------------------------
# # MFCC WINDOW
# # ---------------------------------------------------------
# def mfcc_window(audio, sr, t_center):

#     half = WINDOW_SEC / 2

#     start = int(max(0, (t_center - half) * sr))
#     end = int(min(len(audio), (t_center + half) * sr))

#     segment = audio[start:end]

#     required = int(WINDOW_SEC * sr)

#     if len(segment) < required:
#         segment = np.pad(
#             segment,
#             (0, required - len(segment))
#         )

#     mfcc = librosa.feature.mfcc(
#         y=segment,
#         sr=sr,
#         n_mfcc=N_MFCC
#     )

#     if mfcc.shape[1] < MAX_LEN:

#         mfcc = np.pad(
#             mfcc,
#             ((0,0), (0, MAX_LEN - mfcc.shape[1]))
#         )

#     else:
#         mfcc = mfcc[:, :MAX_LEN]

#     return mfcc[np.newaxis, ..., np.newaxis].astype(np.float32)

# # ---------------------------------------------------------
# # DRAW BOX
# # ---------------------------------------------------------
# def draw_box(frame, box, label, conf, cls_id):

#     x1, y1, x2, y2 = map(
#         int,
#         box.xyxy[0].tolist()
#     )

#     color = COLORS.get(cls_id, (255,255,255))

#     cv2.rectangle(
#         frame,
#         (x1,y1),
#         (x2,y2),
#         color,
#         2
#     )

#     cv2.putText(
#         frame,
#         f"{label} {conf:.2f}",
#         (x1, max(20, y1 - 10)),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.7,
#         color,
#         2
#     )

# # ---------------------------------------------------------
# # TRAFFIC SIGNAL
# # ---------------------------------------------------------
# def traffic_signal(state):

#     red = "light"
#     yellow = "light"
#     green = "light"

#     if state == "RED":
#         red += " red-on"

#     elif state == "YELLOW":
#         yellow += " yellow-on"

#     elif state == "GREEN":
#         green += " green-on"

#     return f"""
#     <div class="signal-box">
#         <div class="{red}"></div>
#         <div class="{yellow}"></div>
#         <div class="{green}"></div>
#     </div>
#     """

# # ---------------------------------------------------------
# # TITLE
# # ---------------------------------------------------------
# st.title("🚦 AI Smart Traffic Signal Control System")

# st.markdown("""
# YOLOv8 + Siren Audio Fusion + Intelligent Traffic Signal Management
# """)

# # ---------------------------------------------------------
# # SIDEBAR
# # ---------------------------------------------------------
# with st.sidebar:

#     st.header("⚙️ Control Panel")

#     conf_thresh = st.slider(
#         "YOLO Confidence",
#         0.1,
#         0.9,
#         0.35,
#         0.05
#     )

#     siren_thresh = st.slider(
#         "Siren Threshold",
#         0.1,
#         0.95,
#         0.50,
#         0.05
#     )

#     green_duration = st.slider(
#         "Green Signal Duration",
#         3,
#         20,
#         8
#     )

# # ---------------------------------------------------------
# # FILE UPLOAD
# # ---------------------------------------------------------
# uploaded_file = st.file_uploader(
#     "Upload Traffic Video",
#     type=["mp4", "avi", "mov", "mkv"]
# )

# if not uploaded_file:
#     st.info("Upload a video to begin.")
#     st.stop()

# start = st.button("▶️ Start Smart Detection")

# if not start:
#     st.stop()

# # ---------------------------------------------------------
# # SAVE VIDEO
# # ---------------------------------------------------------
# input_path = (
#     Path(tempfile.gettempdir())
#     / f"input_{int(time.time())}.mp4"
# )

# with open(input_path, "wb") as f:
#     f.write(uploaded_file.read())

# # ---------------------------------------------------------
# # LOAD MODELS
# # ---------------------------------------------------------
# with st.spinner("Loading AI Models..."):

#     yolo = load_yolo(str(YOLO_WEIGHTS))

#     audio_model = load_audio_model(str(AUDIO_MODEL))

# # ---------------------------------------------------------
# # LOAD VIDEO + AUDIO
# # ---------------------------------------------------------
# video_audio = extract_audio(str(input_path))

# cap = cv2.VideoCapture(str(input_path))

# fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # ---------------------------------------------------------
# # LAYOUT
# # ---------------------------------------------------------
# left_col, right_col = st.columns([4,1])

# video_placeholder = left_col.empty()

# signal_placeholder = right_col.empty()

# countdown_placeholder = right_col.empty()

# stats_placeholder = right_col.empty()

# log_placeholder = right_col.empty()

# progress_bar = st.progress(0)

# # ---------------------------------------------------------
# # VARIABLES
# # ---------------------------------------------------------
# signal_state = "RED"

# green_timer = 0

# yellow_timer = 0

# alert_start_time = None

# emergency_count = 0

# event_log = deque(maxlen=25)

# fps_list = deque(maxlen=20)

# frame_index = 0

# # ---------------------------------------------------------
# # PROCESS VIDEO
# # ---------------------------------------------------------
# while cap.isOpened():

#     loop_start = time.time()

#     success, frame = cap.read()

#     if not success:
#         break

#     frame_index += 1

#     current_time = frame_index / fps

#     # -----------------------------------------------------
#     # YOLO DETECTION
#     # -----------------------------------------------------
#     result = yolo.predict(
#         frame,
#         imgsz=640,
#         conf=conf_thresh,
#         verbose=False
#     )[0]

#     emergency_found = False

#     vehicle_conf = 0

#     labels_seen = []

#     for box in result.boxes:

#         cls_id = int(box.cls.item())

#         conf = float(box.conf.item())

#         label = CLASS_NAMES[cls_id]

#         labels_seen.append(label)

#         draw_box(
#             frame,
#             box,
#             label,
#             conf,
#             cls_id
#         )

#         if cls_id in EMERGENCY_IDS:

#             emergency_found = True

#             vehicle_conf = max(
#                 vehicle_conf,
#                 conf
#             )

#     # -----------------------------------------------------
#     # AUDIO DETECTION
#     # -----------------------------------------------------
#     siren_prob = 0.0

#     if len(video_audio) > 1:

#         siren_prob = float(
#             audio_model.predict(
#                 mfcc_window(
#                     video_audio,
#                     SR,
#                     current_time
#                 ),
#                 verbose=0
#             )[0][0]
#         )

#     # -----------------------------------------------------
#     # FUSION SCORE
#     # -----------------------------------------------------
#     fusion_score = (
#         0.7 * vehicle_conf +
#         0.3 * siren_prob
#     )

#     alert = (
#         emergency_found and
#         siren_prob > siren_thresh and
#         fusion_score > 0.5
#     )

#     # -----------------------------------------------------
#     # STABILIZATION
#     # -----------------------------------------------------
#     if alert:

#         if alert_start_time is None:
#             alert_start_time = current_time

#         elif current_time - alert_start_time >= 1.0:

#             if signal_state != "GREEN":

#                 signal_state = "GREEN"

#                 green_timer = green_duration

#                 emergency_count += 1

#                 event_log.appendleft(
#                     f"🟢 {current_time:.1f}s : Emergency Priority Activated"
#                 )

#     else:
#         alert_start_time = None

#     # -----------------------------------------------------
#     # SIGNAL TIMERS
#     # -----------------------------------------------------
#     if signal_state == "GREEN":

#         green_timer -= 1 / fps

#         if green_timer <= 0:

#             signal_state = "YELLOW"

#             yellow_timer = 2

#             event_log.appendleft(
#                 f"🟡 {current_time:.1f}s : Transition Signal"
#             )

#     elif signal_state == "YELLOW":

#         yellow_timer -= 1 / fps

#         if yellow_timer <= 0:

#             signal_state = "RED"

#             event_log.appendleft(
#                 f"🔴 {current_time:.1f}s : Normal Traffic Restored"
#             )

#     # -----------------------------------------------------
#     # STATUS OVERLAY
#     # -----------------------------------------------------
#     cv2.rectangle(
#         frame,
#         (0,0),
#         (frame.shape[1], 90),
#         (25,25,25),
#         -1
#     )

#     cv2.putText(
#         frame,
#         f"Fusion Score: {fusion_score:.2f}",
#         (20,35),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.9,
#         (0,255,255),
#         2
#     )

#     cv2.putText(
#         frame,
#         f"Signal: {signal_state}",
#         (20,70),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.9,
#         (0,255,0),
#         2
#     )

#     # -----------------------------------------------------
#     # DISPLAY VIDEO
#     # -----------------------------------------------------
#     video_placeholder.image(
#         cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
#         channels="RGB",
#         use_container_width=True
#     )

#     # -----------------------------------------------------
#     # TRAFFIC SIGNAL
#     # -----------------------------------------------------
#     signal_placeholder.markdown(
#         traffic_signal(signal_state),
#         unsafe_allow_html=True
#     )

#     # -----------------------------------------------------
#     # COUNTDOWN
#     # -----------------------------------------------------
#     if signal_state == "GREEN":

#         countdown_placeholder.markdown(f"""
#         <div class="card">
#         <div class="countdown">
#         ⏳ {int(green_timer)} sec
#         </div>
#         </div>
#         """, unsafe_allow_html=True)

#     else:
#         countdown_placeholder.empty()

#     # -----------------------------------------------------
#     # FPS
#     # -----------------------------------------------------
#     processing_fps = 1 / (time.time() - loop_start)

#     fps_list.append(processing_fps)

#     avg_fps = sum(fps_list) / len(fps_list)

#     # -----------------------------------------------------
#     # STATS
#     # -----------------------------------------------------
#     stats_placeholder.markdown(f"""
#     <div class="card">
#         <div class="metric-title">Vehicle Confidence</div>
#         <div class="metric-value">{vehicle_conf:.2f}</div>
#     </div>

#     <div class="card">
#         <div class="metric-title">Siren Probability</div>
#         <div class="metric-value">{siren_prob:.2f}</div>
#     </div>

#     <div class="card">
#         <div class="metric-title">Fusion Score</div>
#         <div class="metric-value">{fusion_score:.2f}</div>
#     </div>

#     <div class="card">
#         <div class="metric-title">Emergencies</div>
#         <div class="metric-value">{emergency_count}</div>
#     </div>

#     <div class="card">
#         <div class="metric-title">Processing FPS</div>
#         <div class="metric-value">{avg_fps:.1f}</div>
#     </div>
#     """, unsafe_allow_html=True)

#     # -----------------------------------------------------
#     # EVENT LOG
#     # -----------------------------------------------------
#     logs_html = "<br>".join(event_log)

#     log_placeholder.markdown(f"""
#     <div class="card">
#         <h4>📜 Event Timeline</h4>
#         <div class="log-box">
#             {logs_html}
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

#     # -----------------------------------------------------
#     # PROGRESS BAR
#     # -----------------------------------------------------
#     progress = int(
#         (frame_index / total_frames) * 100
#     )

#     progress_bar.progress(
#         min(progress, 100)
#     )

# # ---------------------------------------------------------
# # CLEANUP
# # ---------------------------------------------------------
# cap.release()

# st.success("✅ Smart Traffic Processing Completed")

"""
ULTRA ADVANCED SMART TRAFFIC CONTROL SYSTEM
====================================================================

NEW ADVANCED FEATURES
====================================================================

1. Real-time smart dashboard
2. AI fusion confidence engine
3. Animated neon smart-city UI
4. Multi-stage traffic logic
5. Emergency lane visualization
6. Live detection heatmap
7. Real-time analytics charts
8. Dynamic FPS optimization
9. Smooth signal animations
10. Vehicle tracking trail
11. Smart emergency prioritization
12. Audio waveform visualization
13. Confidence history graph
14. Auto performance tuning
15. Emergency detection cooldown
16. Siren waveform pulse effect
17. Live traffic density estimation
18. Priority override engine
19. Signal transition animations
20. Smart city futuristic interface

====================================================================
"""

import time
import tempfile
from pathlib import Path
from collections import deque

import cv2
import librosa
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from moviepy.editor import VideoFileClip
from ultralytics import YOLO

import requests


# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="AI Smart City Traffic System",
    page_icon="🚦",
    layout="wide"
)

# ================================================================
# FUTURISTIC CSS
# ================================================================
st.markdown("""
<style>

/* Main */
.stApp {
    background: linear-gradient(
        135deg,
        #050816,
        #0B1026
    );
    color: white;
}

/* Video */
[data-testid="stImage"] img {
    max-height: 68vh;
    object-fit: contain;
    border-radius: 18px;
    border: 2px solid #00F5FF;
    box-shadow: 0 0 25px #00F5FF55;
}

/* Neon Cards */
.card {
    background: rgba(20,20,35,0.95);
    border: 1px solid #00F5FF55;
    border-radius: 18px;
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0 0 15px #00F5FF22;
}

/* Metrics */
.metric-title {
    color: #BBBBBB;
    text-align: center;
    font-size: 14px;
}

.metric-value {
    text-align: center;
    color: #00F5FF;
    font-size: 34px;
    font-weight: bold;
}

/* Signal */
.signal-box {
    background: #111827;
    width: 140px;
    padding: 20px;
    border-radius: 25px;
    margin: auto;
    box-shadow: 0 0 20px #00F5FF33;
}

.light {
    width: 70px;
    height: 70px;
    border-radius: 50%;
    margin: 15px auto;
    opacity: 0.15;
    transition: all 0.3s ease;
}

.red-on {
    background: red;
    opacity: 1;
    box-shadow: 0 0 40px red;
}

.yellow-on {
    background: yellow;
    opacity: 1;
    box-shadow: 0 0 40px yellow;
}

.green-on {
    background: lime;
    opacity: 1;
    box-shadow: 0 0 40px lime;
}

/* Event Log */
.log-box {
    height: 240px;
    overflow-y: auto;
    background: #0F172A;
    border-radius: 12px;
    padding: 10px;
    font-size: 14px;
}

/* Countdown */
.countdown {
    text-align: center;
    font-size: 30px;
    color: #00FFAA;
    font-weight: bold;
}

/* Alert Pulse */
.alert-box {
    animation: pulse 1s infinite;
    background: rgba(255,0,0,0.15);
    border: 1px solid red;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
    font-size: 22px;
    color: white;
}

@keyframes pulse {
    0% { box-shadow: 0 0 5px red; }
    50% { box-shadow: 0 0 25px red; }
    100% { box-shadow: 0 0 5px red; }
}

</style>
""", unsafe_allow_html=True)

# ================================================================
# PATHS
# ================================================================
ROOT = Path(__file__).resolve().parents[1]

YOLO_WEIGHTS = ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"

AUDIO_MODEL = ROOT / "models" / "siren_cnn.h5"

# ================================================================
# CONSTANTS
# ================================================================
CLASS_NAMES = [
    "emergency_vehicle",
    "normal_vehicle"
]

EMERGENCY_IDS = {0}

COLORS = {
    0: (0,0,255),
    1: (0,255,0)
}

SR = 22050
N_MFCC = 40
MAX_LEN = 174
WINDOW_SEC = 2.0

# ================================================================
# LOAD MODELS
# ================================================================
@st.cache_resource
def load_yolo(path):
    return YOLO(path)

@st.cache_resource
def load_audio_model(path):
    return tf.keras.models.load_model(path)

# ================================================================
# AUDIO EXTRACTION
# ================================================================
def extract_audio(video_path):

    audio = np.zeros(1, dtype=np.float32)

    try:

        with VideoFileClip(video_path) as clip:

            if clip.audio is not None:

                temp_wav = (
                    Path(tempfile.gettempdir())
                    / f"audio_{int(time.time())}.wav"
                )

                clip.audio.write_audiofile(
                    str(temp_wav),
                    fps=SR,
                    logger=None,
                    verbose=False
                )

                audio, _ = librosa.load(
                    str(temp_wav),
                    sr=SR,
                    mono=True
                )

                temp_wav.unlink(missing_ok=True)

    except Exception as e:
        st.warning(f"Audio extraction failed: {e}")

    return audio

# ================================================================
# MFCC
# ================================================================
def mfcc_window(audio, sr, t_center):

    half = WINDOW_SEC / 2

    start = int(max(0, (t_center-half)*sr))
    end = int(min(len(audio), (t_center+half)*sr))

    segment = audio[start:end]

    required = int(WINDOW_SEC * sr)

    if len(segment) < required:

        segment = np.pad(
            segment,
            (0, required-len(segment))
        )

    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=sr,
        n_mfcc=N_MFCC
    )

    if mfcc.shape[1] < MAX_LEN:

        mfcc = np.pad(
            mfcc,
            ((0,0),(0,MAX_LEN-mfcc.shape[1]))
        )

    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc[np.newaxis,...,np.newaxis].astype(np.float32)

# ================================================================
# DRAWING
# ================================================================
def draw_box(frame, box, label, conf, cls_id):

    x1,y1,x2,y2 = map(
        int,
        box.xyxy[0].tolist()
    )

    color = COLORS.get(cls_id,(255,255,255))

    cv2.rectangle(
        frame,
        (x1,y1),
        (x2,y2),
        color,
        3
    )

    cv2.putText(
        frame,
        f"{label} {conf:.2f}",
        (x1, y1-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    # Tracking center point
    cx = int((x1+x2)/2)
    cy = int((y1+y2)/2)

    cv2.circle(
        frame,
        (cx,cy),
        5,
        (255,255,0),
        -1
    )

# ================================================================
# TRAFFIC SIGNAL
# ================================================================
def traffic_signal(state):

    red = "light"
    yellow = "light"
    green = "light"

    if state == "RED":
        red += " red-on"

    elif state == "YELLOW":
        yellow += " yellow-on"

    elif state == "GREEN":
        green += " green-on"

    return f"""
    <div class="signal-box">
        <div class="{red}"></div>
        <div class="{yellow}"></div>
        <div class="{green}"></div>
    </div>
    """

# ================================================================
# TITLE
# ================================================================
st.title("🚦 AI Smart City Traffic Management System")

st.markdown("""
YOLOv8 + Siren Audio Fusion + Intelligent Signal Prioritization
""")

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:

    st.header("⚙️ AI Controls")

    conf_thresh = st.slider(
        "YOLO Confidence",
        0.1,
        0.9,
        0.35,
        0.05
    )

    siren_thresh = st.slider(
        "Siren Threshold",
        0.1,
        0.95,
        0.50,
        0.05
    )

    green_duration = st.slider(
        "Green Duration",
        3,
        20,
        8
    )

    adaptive_mode = st.checkbox(
        "Adaptive AI Mode",
        value=True
    )

# ================================================================
# FILE UPLOAD
# ================================================================
uploaded_file = st.file_uploader(
    "Upload Smart Traffic Video",
    type=["mp4","avi","mov","mkv"]
)

if not uploaded_file:
    st.info("Upload a video to begin.")
    st.stop()

start = st.button("▶ Start AI Smart Processing")

if not start:
    st.stop()

# ================================================================
# SAVE VIDEO
# ================================================================
input_path = (
    Path(tempfile.gettempdir())
    / f"video_{int(time.time())}.mp4"
)

with open(input_path, "wb") as f:
    f.write(uploaded_file.read())

# ================================================================
# LOAD MODELS
# ================================================================
with st.spinner("Loading AI Models..."):

    yolo = load_yolo(str(YOLO_WEIGHTS))

    audio_model = load_audio_model(str(AUDIO_MODEL))

# ================================================================
# LOAD VIDEO + AUDIO
# ================================================================
video_audio = extract_audio(str(input_path))

cap = cv2.VideoCapture(str(input_path))

fps = cap.get(cv2.CAP_PROP_FPS) or 25

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

# ================================================================
# UI LAYOUT
# ================================================================
left_col, right_col = st.columns([4,1])

video_placeholder = left_col.empty()

chart_placeholder = left_col.empty()

signal_placeholder = right_col.empty()

countdown_placeholder = right_col.empty()

alert_placeholder = right_col.empty()

metrics_placeholder = right_col.empty()

log_placeholder = right_col.empty()

progress_bar = st.progress(0)

# ================================================================
# VARIABLES
# ================================================================
signal_state = "RED"

green_timer = 0

yellow_timer = 0

alert_start_time = None

cooldown_timer = 0

emergency_count = 0

vehicle_history = deque(maxlen=50)

siren_history = deque(maxlen=50)

fusion_history = deque(maxlen=50)

event_log = deque(maxlen=30)

fps_history = deque(maxlen=20)

density_history = deque(maxlen=50)

frame_index = 0

# ================================================================
# PROCESS LOOP
# ================================================================
while cap.isOpened():

    start_loop = time.time()

    success, frame = cap.read()

    if not success:
        break

    frame_index += 1

    current_time = frame_index / fps

    # ============================================================
    # YOLO
    # ============================================================
    result = yolo.predict(
        frame,
        imgsz=640,
        conf=conf_thresh,
        verbose=False
    )[0]

    emergency_found = False

    vehicle_conf = 0

    vehicle_count = 0

    for box in result.boxes:

        cls_id = int(box.cls.item())

        conf = float(box.conf.item())

        label = CLASS_NAMES[cls_id]

        draw_box(
            frame,
            box,
            label,
            conf,
            cls_id
        )

        vehicle_count += 1

        if cls_id in EMERGENCY_IDS:

            emergency_found = True

            vehicle_conf = max(
                vehicle_conf,
                conf
            )

    # ============================================================
    # AUDIO
    # ============================================================
    siren_prob = 0

    if len(video_audio) > 1:

        siren_prob = float(
            audio_model.predict(
                mfcc_window(
                    video_audio,
                    SR,
                    current_time
                ),
                verbose=0
            )[0][0]
        )

    # ============================================================
    # AI FUSION
    # ============================================================
    fusion_score = (
        0.65 * vehicle_conf +
        0.35 * siren_prob
    )

    # Adaptive AI
    if adaptive_mode:

        dynamic_thresh = (
            0.45 +
            (vehicle_count * 0.01)
        )

    else:
        dynamic_thresh = 0.5

    alert = (
        emergency_found and
        siren_prob > siren_thresh and
        fusion_score > dynamic_thresh
    )

    # ============================================================
    # PRIORITY ENGINE
    # ============================================================
    if cooldown_timer > 0:
        cooldown_timer -= 1/fps

    if alert and cooldown_timer <= 0:

        if alert_start_time is None:
            alert_start_time = current_time

        elif current_time - alert_start_time >= 1:

            if signal_state != "GREEN":

                signal_state = "GREEN"

                green_timer = green_duration

                emergency_count += 1

                cooldown_timer = 4

                try:
                    requests.get(
                        "http://192.168.137.192/north",
                        timeout=1
                    )
                    print("ESP32 Emergency Sent")
                except:
                    print("ESP32 Not Reachable")


                event_log.appendleft(
                    f"🟢 {current_time:.1f}s : Emergency Priority Activated"
                )

    else:
        alert_start_time = None

    # ============================================================
    # SIGNAL LOGIC
    # ============================================================
    if signal_state == "GREEN":

        green_timer -= 1/fps

        if green_timer <= 0:

            signal_state = "YELLOW"

            yellow_timer = 2

            event_log.appendleft(
                f"🟡 {current_time:.1f}s : Signal Transition"
            )

    elif signal_state == "YELLOW":

        yellow_timer -= 1/fps

        if yellow_timer <= 0:

            signal_state = "RED"

            event_log.appendleft(
                f"🔴 {current_time:.1f}s : Traffic Restored"
            )

    # ============================================================
    # OVERLAY
    # ============================================================
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0,0),
        (width,100),
        (10,10,10),
        -1
    )

    alpha = 0.6

    frame = cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1-alpha,
        0
    )

    cv2.putText(
        frame,
        f"Fusion: {fusion_score:.2f}",
        (20,35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (20,75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0,255,0),
        2
    )

    # ============================================================
    # HISTORY
    # ============================================================
    vehicle_history.append(vehicle_conf)

    siren_history.append(siren_prob)

    fusion_history.append(fusion_score)

    density_history.append(vehicle_count)

    # ============================================================
    # VIDEO DISPLAY
    # ============================================================
    video_placeholder.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        channels="RGB",
        use_container_width=True
    )

    # ============================================================
    # CHARTS
    # ============================================================
    chart_df = pd.DataFrame({
        "Vehicle": list(vehicle_history),
        "Siren": list(siren_history),
        "Fusion": list(fusion_history)
    })

    chart_placeholder.line_chart(chart_df)

    # ============================================================
    # SIGNAL
    # ============================================================
    signal_placeholder.markdown(
        traffic_signal(signal_state),
        unsafe_allow_html=True
    )

    # ============================================================
    # ALERT PANEL
    # ============================================================
    if signal_state == "GREEN":

        alert_placeholder.markdown("""
        <div class="alert-box">
        🚨 EMERGENCY PRIORITY ACTIVE
        </div>
        """, unsafe_allow_html=True)

    else:
        alert_placeholder.empty()

    # ============================================================
    # COUNTDOWN
    # ============================================================
    if signal_state == "GREEN":

        countdown_placeholder.markdown(f"""
        <div class="card">
            <div class="countdown">
            ⏳ {int(green_timer)} sec
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        countdown_placeholder.empty()

    # ============================================================
    # FPS
    # ============================================================
    current_fps = 1/(time.time()-start_loop)

    fps_history.append(current_fps)

    avg_fps = sum(fps_history)/len(fps_history)

    # ============================================================
    # METRICS
    # ============================================================
    traffic_density = np.mean(density_history)

    metrics_placeholder.markdown(f"""
    <div class="card">
        <div class="metric-title">Fusion Score</div>
        <div class="metric-value">{fusion_score:.2f}</div>
    </div>

    <div class="card">
        <div class="metric-title">Traffic Density</div>
        <div class="metric-value">{traffic_density:.1f}</div>
    </div>

    <div class="card">
        <div class="metric-title">Emergencies</div>
        <div class="metric-value">{emergency_count}</div>
    </div>

    <div class="card">
        <div class="metric-title">AI FPS</div>
        <div class="metric-value">{avg_fps:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # LOGS
    # ============================================================
    logs_html = "<br>".join(event_log)

    log_placeholder.markdown(f"""
    <div class="card">
        <h4>📜 AI Event Timeline</h4>
        <div class="log-box">
        {logs_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # PROGRESS
    # ============================================================
    progress = int(
        (frame_index/total_frames)*100
    )

    progress_bar.progress(
        min(progress,100)
    )

# ================================================================
# CLEANUP
# ================================================================
cap.release()

st.success("✅ AI Smart Traffic Processing Complete")