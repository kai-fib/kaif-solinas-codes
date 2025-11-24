# import os
# import shutil

# def pull_labels_and_images(reference_dir, images_dir, extensions=['.txt']):
#     """
#     Ensure both .txt files and their corresponding image files are in reference_dir.
#     Looks at all .txt files in reference_dir and pulls matching images from images_dir.
    
#     :param reference_dir: Folder containing .txt files (like ob_prob)
#     :param images_dir: Folder where original images are stored
#     :param extensions: Possible image extensions
#     """
#     # Collect all base filenames from reference_dir (without extension)
#     reference_files = {os.path.splitext(f)[0] for f in os.listdir(reference_dir) if f.endswith('.jpg')}

#     for base in reference_files:
#         # Copy matching image (try all extensions)
#         for ext in extensions:
#             image_file = os.path.join(images_dir, base + ext)
#             if os.path.exists(image_file):
#                 shutil.move(image_file, os.path.join(reference_dir, base + ext))
#                 break  # Stop after copying the first found image

# # ---------------- Example usage ----------------
# reference_dir = r"D:\Remove\broken\br_2_sd" # Folder with filtered .txt files
# images_dir = r"D:\val_update\New folder"  

# pull_labels_and_images(reference_dir, images_dir)


import os
import shutil

def copy_matching_txts(image_dir, txt_source_dir, output_dir):
    """
    For every image in image_dir, find the corresponding .txt file
    in txt_source_dir and copy it to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all image base names (without extensions)
    image_bases = {os.path.splitext(f)[0] for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg'))}

    copied = 0
    for base in image_bases:
        txt_file = os.path.join(txt_source_dir, base + '.txt')
        if os.path.exists(txt_file):
            shutil.copy(txt_file, os.path.join(output_dir, base + '.txt'))
            copied += 1

    print(f"✅ Copied {copied} .txt files to '{output_dir}'")

# ---------------- Example usage ----------------
image_dir = r"D:\class_wise\2.3.4\New_11\train\d\old_txt" # pull the deform images next
txt_source_dir = r"D:\class_wise\2.3.4\old_11\val"
output_dir = r"D:\class_wise\2.3.4\New_11\train\d\old_txt"

copy_matching_txts(image_dir, txt_source_dir, output_dir)
