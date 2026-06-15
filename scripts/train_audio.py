"""
Train CNN on MFCC features for siren vs non_siren classification
"""

import os, glob
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# ---------------- PATHS ----------------
ROOT = os.path.join(os.path.dirname(__file__), "..")
AUDIO_DIR = os.path.join(ROOT, "dataset", "audio")
OUT_DIR   = os.path.join(ROOT, "outputs")
MODEL_DIR = os.path.join(ROOT, "models")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- CONFIG ----------------
SR = 22050
DURATION = 4.0
N_MFCC = 40
MAX_LEN = 174

# ---------------- FEATURE EXTRACTION ----------------
def extract_mfcc(path):
    y, sr = librosa.load(path, sr=SR, duration=DURATION)

    if len(y) < int(SR * DURATION):
        y = np.pad(y, (0, int(SR * DURATION) - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_LEN - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc.astype(np.float32)

# ---------------- LOAD DATA ----------------
X, y = [], []

for label, name in enumerate(["non_siren", "siren"]):
    files = glob.glob(os.path.join(AUDIO_DIR, name, "*.wav"))
    print(f"{name}: {len(files)} files")

    for f in files:
        try:
            X.append(extract_mfcc(f))
            y.append(label)
        except Exception as e:
            print("skip:", f, e)

X = np.array(X)[..., None]
y = np.array(y)

print("Shape:", X.shape)
print("Class balance:", np.bincount(y))

# ---------------- NORMALIZATION ----------------
X = (X - np.mean(X)) / (np.std(X) + 1e-6)

# ---------------- SPLIT ----------------
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- CLASS WEIGHTS ----------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# ---------------- MODEL ----------------
model = models.Sequential([
    layers.Input(shape=X.shape[1:]),

    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPool2D(),

    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPool2D(),

    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),

    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),

    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- CALLBACK ----------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ---------------- TRAIN ----------------
hist = model.fit(
    Xtr, ytr,
    validation_data=(Xte, yte),
    epochs=25,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# ---------------- SAVE MODEL ----------------
model.save(os.path.join(MODEL_DIR, "audio_siren_cnn.h5"))

# ---------------- PLOT ACCURACY ----------------
plt.figure()
plt.plot(hist.history["accuracy"], label="train_acc")
plt.plot(hist.history["val_accuracy"], label="val_acc")
plt.legend()
plt.title("Audio CNN Accuracy")
plt.savefig(os.path.join(OUT_DIR, "audio_training_curves.png"))

# ---------------- EVALUATION ----------------
yp = (model.predict(Xte) > 0.5).astype(int).ravel()

print("\nClassification Report:\n")
print(classification_report(yte, yp, target_names=["non_siren", "siren"]))

cm = confusion_matrix(yte, yp)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["non_siren", "siren"],
            yticklabels=["non_siren", "siren"])
plt.title("Confusion Matrix")
plt.savefig(os.path.join(OUT_DIR, "audio_confusion_matrix.png"))

print("\n✅ Model + plots saved successfully!")