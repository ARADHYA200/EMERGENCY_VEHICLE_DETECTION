
# # import os

# # label_folder = "raw_data/ambulances/train/labels"

# # for file in os.listdir(label_folder):
# #     if file.endswith(".txt"):
# #         path = os.path.join(label_folder, file)

# #         new_lines = []
# #         with open(path, "r") as f:
# #             for line in f:
# #                 parts = line.strip().split()
# #                 parts[0] = "0"   # 👈 class force 0
# #                 new_lines.append(" ".join(parts))

# #         with open(path, "w") as f:
# #             f.write("\n".join(new_lines))

# from ultralytics import YOLO
# import cv2
# import os

# model = YOLO("yolov8n.pt")

# input_folder = "raw_data/ambulances/val/images"
# output_folder = "raw_data/ambulances/val/labels"

# os.makedirs(output_folder, exist_ok=True)

# for img_name in os.listdir(input_folder):
#     if img_name.endswith((".jpg", ".png")):

#         img_path = os.path.join(input_folder, img_name)
#         results = model(img_path)

#         img = cv2.imread(img_path)
#         h, w, _ = img.shape

#         label_path = os.path.join(output_folder, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

#         with open(label_path, "w") as f:
#             for r in results[0].boxes:
#                 cls = int(r.cls[0])
#                 x, y, w_box, h_box = r.xywhn[0]

#                 f.write(f"{cls} {x} {y} {w_box} {h_box}\n")

import os

folders = [
    "raw_data/Vehicles/train/labels"
    
]

for folder in folders:

    for file in os.listdir(folder):

        if file.endswith(".txt"):

            path = os.path.join(folder, file)

            new_lines = []

            with open(path, "r") as f:

                for line in f:

                    parts = line.strip().split()

                    if len(parts) > 0:
                        parts[0] = "1"   # 👈 NORMAL VEHICLE

                    new_lines.append(" ".join(parts))

            with open(path, "w") as f:
                f.write("\n".join(new_lines))

print("All labels converted to class 1 (NORMAL)")