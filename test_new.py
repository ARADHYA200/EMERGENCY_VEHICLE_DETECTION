import os

folders = [
    "final_dataset/labels/train",
    "final_dataset/labels/val"
]

c0 = 0
c1 = 0

for folder in folders:

    for file in os.listdir(folder):

        if file.endswith(".txt"):

            path = os.path.join(folder, file)

            with open(path, "r") as f:

                for line in f:

                    cls = line.strip().split()[0]

                    if cls == "0":
                        c0 += 1

                    elif cls == "1":
                        c1 += 1

print("\n===== FINAL LABEL COUNT =====")
print(f"Emergency (0): {c0}")
print(f"Normal    (1): {c1}")

# import os
# import shutil

# # ==========================================
# # FINAL DATASET MERGE SCRIPT
# # ==========================================

# # -------- EMERGENCY DATASETS --------
# emergency_sources = [
#     "raw_data/ambulances",
#     "raw_data/roboflow/ambulance",
#     "raw_data/roboflow/fire",
#     "raw_data/roboflow/police"
# ]

# # -------- NORMAL DATASETS --------
# normal_sources = [
#     "raw_data/Vehicles"
# ]

# # -------- FINAL DESTINATION --------
# final_base = "final_dataset"

# # -------- CREATE FOLDERS --------
# folders = [
#     "final_dataset/images/train",
#     "final_dataset/images/val",
#     "final_dataset/labels/train",
#     "final_dataset/labels/val"
# ]

# for folder in folders:
#     os.makedirs(folder, exist_ok=True)

# # ==========================================
# # COPY FUNCTION
# # ==========================================

# image_ext = (".jpg", ".jpeg", ".png")

# def merge_dataset(source_list, class_name):

#     global train_count
#     global val_count

#     for source in source_list:

#         print(f"\nProcessing: {source}")

#         # support both val and valid
#         splits = {
#             "train": "train",
#             "val": ["val", "valid"]
#         }

#         for final_split, possible_splits in splits.items():

#             if isinstance(possible_splits, str):
#                 possible_splits = [possible_splits]

#             for split in possible_splits:

#                 img_dir = os.path.join(source, split, "images")
#                 lbl_dir = os.path.join(source, split, "labels")

#                 if not os.path.exists(img_dir):
#                     continue

#                 for file in os.listdir(img_dir):

#                     if file.lower().endswith(image_ext):

#                         img_path = os.path.join(img_dir, file)

#                         label_name = os.path.splitext(file)[0] + ".txt"
#                         lbl_path = os.path.join(lbl_dir, label_name)

#                         if not os.path.exists(lbl_path):
#                             continue

#                         # UNIQUE FILE NAME
#                         if final_split == "train":
#                             new_name = f"{train_count}_{file}"
#                             train_count += 1
#                         else:
#                             new_name = f"{val_count}_{file}"
#                             val_count += 1

#                         # DESTINATION PATHS
#                         dst_img = os.path.join(
#                             final_base,
#                             "images",
#                             final_split,
#                             new_name
#                         )

#                         dst_lbl = os.path.join(
#                             final_base,
#                             "labels",
#                             final_split,
#                             os.path.splitext(new_name)[0] + ".txt"
#                         )

#                         # COPY FILES
#                         shutil.copy2(img_path, dst_img)
#                         shutil.copy2(lbl_path, dst_lbl)

#     print(f"\n✅ {class_name} merged successfully")


# # ==========================================
# # START MERGING
# # ==========================================

# train_count = 0
# val_count = 0

# merge_dataset(emergency_sources, "EMERGENCY")
# merge_dataset(normal_sources, "NORMAL")

# # ==========================================
# # FINAL SUMMARY
# # ==========================================

# train_images = len(os.listdir("final_dataset/images/train"))
# val_images = len(os.listdir("final_dataset/images/val"))

# train_labels = len(os.listdir("final_dataset/labels/train"))
# val_labels = len(os.listdir("final_dataset/labels/val"))

# print("\n===================================")
# print(" FINAL DATASET CREATED SUCCESSFULLY ")
# print("===================================")

# print(f"\nTrain Images : {train_images}")
# print(f"Train Labels : {train_labels}")

# print(f"\nVal Images   : {val_images}")
# print(f"Val Labels   : {val_labels}")

# print("\nDataset Location: final_dataset")

# # import os

# # datasets = {
# #     "ambulances": "raw_data/ambulances",
# #     "vehicles": "raw_data/Vehicles",
# #     "roboflow_ambulance": "raw_data/roboflow/ambulance",
# #     "roboflow_fire": "raw_data/roboflow/fire",
# #     "roboflow_normal": "raw_data/roboflow/normal",
# #     "roboflow_police": "raw_data/roboflow/police",
# # }

# # image_ext = (".jpg", ".jpeg", ".png")

# # print("\n========== DETAILED DATASET SUMMARY ==========\n")

# # for name, base_path in datasets.items():

# #     print(f"\n{name.upper()}")

# #     total_class_0 = 0
# #     total_class_1 = 0

# #     for split in ["train", "val", "valid", "test"]:

# #         img_path = os.path.join(base_path, split, "images")
# #         lbl_path = os.path.join(base_path, split, "labels")

# #         if os.path.exists(img_path):

# #             image_count = len([
# #                 f for f in os.listdir(img_path)
# #                 if f.lower().endswith(image_ext)
# #             ])

# #             label_count = len([
# #                 f for f in os.listdir(lbl_path)
# #                 if f.endswith(".txt")
# #             ]) if os.path.exists(lbl_path) else 0

# #             class_0 = 0
# #             class_1 = 0

# #             if os.path.exists(lbl_path):

# #                 for txt_file in os.listdir(lbl_path):

# #                     if txt_file.endswith(".txt"):

# #                         txt_path = os.path.join(lbl_path, txt_file)

# #                         with open(txt_path, "r") as f:

# #                             for line in f:

# #                                 parts = line.strip().split()

# #                                 if len(parts) > 0:

# #                                     if parts[0] == "0":
# #                                         class_0 += 1

# #                                     elif parts[0] == "1":
# #                                         class_1 += 1

# #             total_class_0 += class_0
# #             total_class_1 += class_1

# #             print(f"\n  {split.upper()}")
# #             print(f"     Images : {image_count}")
# #             print(f"     Labels : {label_count}")
# #             print(f"     Class 0: {class_0}")
# #             print(f"     Class 1: {class_1}")

# #     print("\n  TOTAL")
# #     print(f"     Class 0: {total_class_0}")
# #     print(f"     Class 1: {total_class_1}")

# # print("\n==============================================\n")