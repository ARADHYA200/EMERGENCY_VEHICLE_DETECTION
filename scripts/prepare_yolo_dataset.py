"""
Convert + merge Open Images subset and COCO subset into ONE unified YOLO dataset.

Final unified class mapping:
  0 -> emergency_vehicle
  1 -> normal_vehicle

Source class -> unified id:
  Open Images "Ambulance"   -> 0
  Open Images "Police car"  -> 1
  Open Images "Fire engine" -> 2
  COCO "car" / "bus" / "truck" -> 3

Output:
  dataset/images/{train,val}/*.jpg
  dataset/labels/{train,val}/*.txt   (YOLO format: cls cx cy w h, normalized)
"""
import os, json, shutil, random
from glob import glob
from PIL import Image
from tqdm import tqdm

ROOT = os.path.join(os.path.dirname(__file__), "..")
OI_DIR   = os.path.join(ROOT, "raw_data", "openimages")
COCO_DIR = os.path.join(ROOT, "raw_data", "coco")
OUT_IMG  = os.path.join(ROOT, "dataset", "images")
OUT_LBL  = os.path.join(ROOT, "dataset", "labels")
for s in ("train", "val"):
    os.makedirs(os.path.join(OUT_IMG, s), exist_ok=True)
    os.makedirs(os.path.join(OUT_LBL, s), exist_ok=True)

OI_MAP = {"Ambulance": 0, "Police car": 0, "Fire engine": 0}
COCO_MAP = {"car": 1, "bus": 1, "truck": 1}

def coco_to_yolo(box, w, h):
    x, y, bw, bh = box
    return ((x + bw/2)/w, (y + bh/2)/h, bw/w, bh/h)

def process_coco_export(coco_root, split_in, split_out, class_map):
    """Reads a FiftyOne COCO export and writes YOLO files into split_out."""
    ann_path = os.path.join(coco_root, split_in, "labels.json")
    img_dir  = os.path.join(coco_root, split_in, "data")
    if not os.path.exists(ann_path):
        print(f"  skip (missing): {ann_path}")
        return 0
    with open(ann_path) as f:
        coco = json.load(f)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    img_index = {im["id"]: im for im in coco["images"]}
    ann_by_img = {}
    for a in coco["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    n = 0
    for img_id, im in tqdm(img_index.items(), desc=f"{coco_root}:{split_in}->{split_out}"):
        anns = ann_by_img.get(img_id, [])
        keep = [(class_map[cats[a["category_id"]]], a["bbox"])
                for a in anns if cats[a["category_id"]] in class_map]
        if not keep:
            continue
        src_img = os.path.join(img_dir, im["file_name"])
        if not os.path.exists(src_img):
            continue
        # ensure unique filenames across sources
        base = os.path.splitext(os.path.basename(im["file_name"]))[0]
        prefix = "oi" if "openimages" in coco_root else "co"
        new_name = f"{prefix}_{base}"
        dst_img = os.path.join(OUT_IMG, split_out, new_name + ".jpg")
        try:
            Image.open(src_img).convert("RGB").save(dst_img, "JPEG", quality=92)
        except Exception:
            continue
        W, H = im["width"], im["height"]
        lines = []
        for cls, box in keep:
            cx, cy, bw, bh = coco_to_yolo(box, W, H)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(os.path.join(OUT_LBL, split_out, new_name + ".txt"), "w") as f:
            f.write("\n".join(lines))
        n += 1
    return n

random.seed(42)
total = 0
total += process_coco_export(OI_DIR,   "train",      "train", OI_MAP)
total += process_coco_export(OI_DIR,   "validation", "val",   OI_MAP)
total += process_coco_export(COCO_DIR, "train",      "train", COCO_MAP)
total += process_coco_export(COCO_DIR, "validation", "val",   COCO_MAP)

print(f"\nUnified YOLO dataset built. Images written: {total}")
print("Train images:", len(glob(os.path.join(OUT_IMG, 'train', '*.jpg'))))
print("Val   images:", len(glob(os.path.join(OUT_IMG, 'val',   '*.jpg'))))
