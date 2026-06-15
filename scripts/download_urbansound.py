"""
Download UrbanSound8K and split into:
  dataset/audio/siren/
  dataset/audio/non_siren/

UrbanSound8K must be downloaded manually (license click-through) from:
  https://urbansounddataset.weebly.com/urbansound8k.html

Place the extracted folder at: raw_data/UrbanSound8K/
This script then organizes wav files by class using metadata/UrbanSound8K.csv.
Class 'siren' -> siren/, all other 9 classes -> non_siren/.
"""
import os, shutil, pandas as pd
from tqdm import tqdm

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC  = os.path.join(ROOT, "raw_data", "UrbanSound8K")
META = os.path.join(SRC, "metadata", "UrbanSound8K.csv")
AUDIO_SRC = os.path.join(SRC, "audio")
DST = os.path.join(ROOT, "dataset", "audio")

os.makedirs(os.path.join(DST, "siren"), exist_ok=True)
os.makedirs(os.path.join(DST, "non_siren"), exist_ok=True)

assert os.path.exists(META), (
    f"Missing {META}\n"
    "Download UrbanSound8K manually and extract to raw_data/UrbanSound8K/"
)

df = pd.read_csv(META)
for _, row in tqdm(df.iterrows(), total=len(df), desc="organizing"):
    src = os.path.join(AUDIO_SRC, f"fold{row['fold']}", row["slice_file_name"])
    if not os.path.exists(src):
        continue
    cls = "siren" if row["class"] == "siren" else "non_siren"
    shutil.copy2(src, os.path.join(DST, cls, row["slice_file_name"]))

print("Done. Audio organised under dataset/audio/{siren,non_siren}/")
