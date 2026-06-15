#!/usr/bin/env python3
"""
Audio inference module — real-time microphone siren detection
(Simplified version without scaler)
"""

import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import threading
import queue
from pathlib import Path

MODEL_PATH = Path("models/siren_cnn.h5")  # ⚠️ FIXED NAME

SR         = 22050
DURATION   = 4.0
N_MFCC     = 40
MAX_LEN    = 174
THRESHOLD  = 0.70

class SirenDetector:
    def __init__(self):
        print("[INFO] Loading siren model...")
        self.model = tf.keras.models.load_model(str(MODEL_PATH))

        self._siren_active = False
        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue()

    @property
    def siren_active(self) -> bool:
        with self._lock:
            return self._siren_active

    def _predict_audio(self, y: np.ndarray):
        target_len = int(SR * DURATION)

        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)

        if mfcc.shape[1] < MAX_LEN:
            mfcc = np.pad(mfcc, ((0,0),(0, MAX_LEN - mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :MAX_LEN]

        inp = mfcc[np.newaxis, ..., np.newaxis]

        prob = float(self.model.predict(inp, verbose=0)[0][0])

        return prob >= THRESHOLD, prob

    def _audio_callback(self, indata, frames, time, status):
        self._q.put(indata[:, 0].copy())

    def start_realtime(self):
        def _loop():
            buffer = np.array([], dtype=np.float32)
            chunk = int(SR * DURATION)

            with sd.InputStream(samplerate=SR, channels=1,
                                callback=self._audio_callback, blocksize=1024):

                print("[AUDIO] Real-time detection started...")

                while True:
                    data = self._q.get()
                    buffer = np.concatenate([buffer, data])

                    if len(buffer) >= chunk:
                        segment = buffer[:chunk]
                        buffer = buffer[chunk:]

                        is_siren, conf = self._predict_audio(segment)

                        with self._lock:
                            self._siren_active = is_siren

                        status = "SIREN 🚨" if is_siren else "No siren"
                        print(f"[{status}] {conf:.2f}")

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        return t