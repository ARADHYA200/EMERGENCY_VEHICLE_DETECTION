import os
import fiftyone as fo
import fiftyone.zoo as foz

CLASSES = ["car", "bus", "truck"]

EXPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "raw_data", "coco")
os.makedirs(EXPORT_DIR, exist_ok=True)

for split, max_samples in [("train", 1500), ("validation", 300)]:
    print(f"\n==> Downloading COCO split={split} for {CLASSES}")

    ds = foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        label_types=["detections"],
        classes=CLASSES,
        only_matching=True,
        max_samples=max_samples,
        dataset_name=f"coco_{split}",
        shuffle=True,
    )

    out = os.path.join(EXPORT_DIR, split)

    # ✅ FIX HERE
    ds.export(
        export_dir=out,
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth",
        classes=CLASSES,
    )

    print(f"   exported -> {out}")

print("\n✅ Done. COCO subset ready")