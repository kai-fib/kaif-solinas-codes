# import os
# import shutil

# def pull_class_id_files(input_dir, output_dir, target_class_id):
#     """
#     Move label files containing a given class ID — or empty files if target_class_id == 'empty'.
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     for filename in os.listdir(input_dir):
#         if filename.endswith('.txt'):
#             input_file = os.path.join(input_dir, filename)

#             with open(input_file, 'r') as file:
#                 lines = file.readlines()

#             # ✅ Case 1: move empty .txt files
#             if target_class_id == "empty":
#                 if not lines:  # empty file
#                     shutil.move(input_file, os.path.join(output_dir, filename))
#                 continue

#             # ✅ Case 2: normal class-based selection
#             for line in lines:
#                 parts = line.strip().split()
#                 if parts:
#                     # clean first token (remove non-digit chars)
#                     class_id_str = ''.join(filter(str.isdigit, parts[0]))
#                     if class_id_str.isdigit() and int(class_id_str) == target_class_id:
#                         shutil.move(input_file, os.path.join(output_dir, filename))
#                         break


# # ---------------- Example usage ----------------
# input_dir = r"D:\dataset\wrc_sewer_2.3.3\labels\train"
# output_dir = r"D:\class_wise\2.3.3\new\4"
# target_class_id = "4"  # use integer (e.g., 1) or "empty"

# pull_class_id_files(input_dir, output_dir, target_class_id)

import os
import shutil

def pull_class_id_files(input_dir, output_dir, target_class_id):
    # Create output folder if it doesn’t exist
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_file = os.path.join(input_dir, filename)

            with open(input_file, 'r') as file:
                lines = file.readlines()

            for line in lines:
                parts = line.strip().split()
                if parts:
                    # Clean first token (remove ':' or other non-digit chars)
                    class_id_str = ''.join(filter(str.isdigit, parts[0]))
                    
                    # Check if it's a valid integer and equals target_class_id
                    if class_id_str.isdigit() and int(class_id_str) == target_class_id:
                        shutil.move(input_file, os.path.join(output_dir, filename))
                        break


# ---------------- Example usage ----------------
input_dir = r"D:\dataset\wrc_sewer_2.3.4\wrc_sewer_2.3.4\labels\train"
output_dir = r"D:\class_wise\2.3.4\old_11\train"
target_class_id = 11

pull_class_id_files(input_dir, output_dir, target_class_id)
