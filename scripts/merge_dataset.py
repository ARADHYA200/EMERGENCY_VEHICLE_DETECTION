import os
import shutil
import random

# INPUT folders
SOURCE_DIRS = [
    "raw_data/roboflow/ambulance",
    "raw_data/roboflow/fire",
    "raw_data/roboflow/police",
    "raw_data/roboflow/normal"
]

# OUTPUT dataset
OUTPUT_DIR = "dataset"

TRAIN_RATIO = 0.8

# ---------------------- CREATE FOLDERS ----------------------
def create_dirs():
    for split in ["train", "val"]:
        os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

# ---------------------- FIX LABELS ----------------------
def fix_labels(label_path, dataset_type):
    new_lines = []

    if not os.path.exists(label_path):
        return new_lines

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        # skip corrupt lines
        if len(parts) < 5:
            continue

        try:
            cls = int(float(parts[0]))
        except:
            continue

        bbox = parts[1:]

        # 🔥 CLASS MAPPING
        if dataset_type in ["ambulance", "fire"]:
            cls = 0  # emergency

        elif dataset_type == "normal":
            if cls == 0:
                cls = 1  # normal
            elif cls == 1:
                cls = 0  # emergency

        elif dataset_type == "police":
            if cls != 0:
                continue  # ❌ skip unwanted classes
            cls = 0  # emergency

        new_lines.append(" ".join([str(cls)] + bbox))

    return new_lines

# ---------------------- COLLECT FILES ----------------------
def collect_files():
    image_files = []

    for src in SOURCE_DIRS:
        dataset_type = os.path.basename(src)  # ambulance / fire / police / normal

        for split in ["train", "valid", "test"]:
            img_path = os.path.join(src, split, "images")
            lbl_path = os.path.join(src, split, "labels")

            if not os.path.exists(img_path):
                continue

            for file in os.listdir(img_path):
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    image_files.append((
                        os.path.join(img_path, file),
                        os.path.join(lbl_path, file.replace(".jpg", ".txt").replace(".png", ".txt")),
                        dataset_type
                    ))

    return image_files

# ---------------------- SPLIT & COPY ----------------------
def split_and_copy(files):
    random.shuffle(files)
    split_idx = int(len(files) * TRAIN_RATIO)

    train_files = files[:split_idx]
    val_files = files[split_idx:]

    for split_name, file_list in [("train", train_files), ("val", val_files)]:
        for img_path, lbl_path, dataset_type in file_list:
            img_name = os.path.basename(img_path)
            lbl_name = img_name.rsplit(".", 1)[0] + ".txt"

            dest_img = f"{OUTPUT_DIR}/images/{split_name}/{img_name}"
            dest_lbl = f"{OUTPUT_DIR}/labels/{split_name}/{lbl_name}"

            # copy image
            shutil.copy(img_path, dest_img)

            # fix & write labels
            fixed_labels = fix_labels(lbl_path, dataset_type)

            if len(fixed_labels) > 0:
                with open(dest_lbl, "w") as f:
                    for line in fixed_labels:
                        f.write(line + "\n")

# ---------------------- MAIN ----------------------
def main():
    print("🧹 Cleaning old dataset...")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    print("📁 Creating folders...")
    create_dirs()

    print("📦 Collecting files...")
    files = collect_files()
    print(f"Total files: {len(files)}")

    print("🔀 Splitting & fixing labels...")
    split_and_copy(files)

    print("✅ Dataset merged & cleaned successfully!")

if __name__ == "__main__":
    main()