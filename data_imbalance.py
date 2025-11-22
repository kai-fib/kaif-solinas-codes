# import os
# import shutil

# # Paths
# image_label_dir = r"D:\check"   # your main dataset folder
# output_dir = r"C:\Users\Kaif Ibrahim\Desktop\d_check_files"

# # Create output folder if it doesn’t exist
# os.makedirs(output_dir, exist_ok=True)

# # Get all files
# all_files = os.listdir(image_label_dir)

# # Separate images and labels
# images = {os.path.splitext(f)[0] for f in all_files if f.lower().endswith(".jpg")}
# labels = {os.path.splitext(f)[0] for f in all_files if f.lower().endswith(".txt")}

# # Find mismatched files
# only_images = images - labels   # images without labels
# only_labels = labels - images   # labels without images

# # Move unpaired files
# moved_count = 0
# for fname in all_files:
#     name, ext = os.path.splitext(fname)
#     if (name in only_images and ext.lower() == ".jpg") or (name in only_labels and ext.lower() == ".txt"):
#         src = os.path.join(image_label_dir, fname)
#         dst = os.path.join(output_dir, fname)
#         shutil.move(src, dst)
#         moved_count += 1
#         print(f"Moved: {fname}")

# # Summary
# print("\n--- SUMMARY ---")
# print(f"Total images in dataset: {len(images)}")
# print(f"Total labels in dataset: {len(labels)}")
# print(f"Images without labels: {len(only_images)}")
# print(f"Labels without images: {len(only_labels)}")
# print(f"Total moved to check folder: {moved_count}")


"""orphans in single folder both labels and images"""
# import os
# import shutil

# # Paths
# image_label_dir = r"D:\check"   # dataset folder
# output_dir = r"C:\Users\Kaif Ibrahim\Desktop\d_check_files\New folder"

# # Create output folder if it doesn’t exist
# os.makedirs(output_dir, exist_ok=True)

# # List all files in dataset
# all_files = os.listdir(image_label_dir)

# moved_count = 0

# for fname in all_files:
#     name, ext = os.path.splitext(fname)
#     ext = ext.lower()

#     if ext not in [".jpg", ".txt"]:
#         continue  # skip non-image/label files

#     # Build expected pair filename
#     pair_ext = ".txt" if ext == ".jpg" else ".jpg"
#     pair_file = name + pair_ext

#     # Check if pair exists
#     if pair_file not in all_files:
#         src = os.path.join(image_label_dir, fname)
#         dst = os.path.join(output_dir, fname)
#         shutil.move(src, dst)
#         moved_count += 1
#         print(f"Moved orphan: {fname}")

# print(f"\nDone! Total orphan files moved: {moved_count}")

# # import os
# # print("Dataset folder files:", len(os.listdir(r"D:\check")))
# # print("Check folder files:", len(os.listdir(r"C:\Users\Kaif Ibrahim\Desktop\d_check_files")))


import os
import shutil

# Paths
images_dir = r"C:\Users\Kaif Ibrahim\Desktop\solinas_downloads\Stage-III(Dataset)\wrc_sewer_2.2.0\wrc_sewer_2.2.0\images\val"
labels_dir = r"C:\Users\Kaif Ibrahim\Desktop\solinas_downloads\Stage-III(Dataset)\wrc_sewer_2.2.0\wrc_sewer_2.2.0\labels\val"
orphan_dir = r"C:\Users\Kaif Ibrahim\Desktop\check\orphan_val_2.2.0"

# Create orphan folder if it doesn’t exist
os.makedirs(orphan_dir, exist_ok=True)

# Get base filenames (without extensions)
image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))}
label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.lower().endswith(".txt")}

# Find orphans
orphan_images = image_files - label_files
orphan_labels = label_files - image_files

moved_count = 0

# Move orphan images
for name in orphan_images:
    for ext in [".jpg", ".png", ".jpeg"]:
        src = os.path.join(images_dir, name + ext)
        if os.path.exists(src):
            dst = os.path.join(orphan_dir, name + ext)
            shutil.move(src, dst)
            moved_count += 1
            print(f"Moved orphan image: {name + ext}")

# Move orphan labels
for name in orphan_labels:
    src = os.path.join(labels_dir, name + ".txt")
    if os.path.exists(src):
        dst = os.path.join(orphan_dir, name + ".txt")
        shutil.move(src, dst)
        moved_count += 1
        print(f"Moved orphan label: {name}.txt")

print(f"\n✅ Done! Total orphan files moved: {moved_count}")
