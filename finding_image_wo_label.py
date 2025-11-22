import os
import shutil

# 🔹 Set your dataset paths
images_dir = r"D:\dataset\wrc_sewer_2.3.1\wrc_sewer_2.3.1\images\val"
labels_dir = r"D:\dataset\wrc_sewer_2.3.1\wrc_sewer_2.3.1\labels\val"
output_dir = r"C:\Users\Kaif Ibrahim\Desktop\check\orphan"

# Supported image formats
image_exts = [".jpg", ".jpeg", ".png"]

# Create orphan output folder if it doesn’t exist
os.makedirs(output_dir, exist_ok=True)

# Get all image basenames (without extension)
image_basenames = set()
for file in os.listdir(images_dir):
    name, ext = os.path.splitext(file)
    if ext.lower() in image_exts:
        image_basenames.add(name)

# Find orphan label files
orphans = []
for label_file in os.listdir(labels_dir):
    name, ext = os.path.splitext(label_file)
    if ext.lower() == ".txt" and name not in image_basenames:
        orphans.append(label_file)
        shutil.move(os.path.join(labels_dir, label_file), os.path.join(output_dir, label_file))

# Show results
if orphans:
    print(f"⚠️ Found {len(orphans)} orphan label files (no matching image).")
    for f in orphans[:10]:  # show first 10 only for brevity
        print(" -", f)
    print(f"\n✅ All orphan labels moved to: {output_dir}")
else:
    print("✅ No orphan labels found — all labels have corresponding images.")
