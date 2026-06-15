import os
import fiftyone as fo
import fiftyone.zoo as foz

CLASSES = ["Ambulance"]

EXPORT_DIR = r"D:\emergency_vehicle_detection\raw_data\openimages"
os.makedirs(EXPORT_DIR, exist_ok=True)

for split, max_samples in [("train", 500), ("validation", 100)]:
    print(f"\n==> Downloading Open Images split={split}")

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        label_types=["detections"],
        classes=CLASSES,
        only_matching=True,
        max_samples=max_samples,
        shuffle=True,
    )

    export_dir = os.path.join(EXPORT_DIR, split)

    # ✅ FIX HERE
    dataset.export(
        export_dir=export_dir,
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth",   # 👈 IMPORTANT
        classes=CLASSES,
    )

    print(f"   exported -> {export_dir}")

print("\n✅ Done")