import os
from collections import defaultdict

def update_class_ids(input_dir, output_dir, id_mapping):
    """
    Update multiple class IDs in YOLO label files based on a mapping dictionary.

    Args:
        input_dir (str): Directory containing input YOLO label .txt files.
        output_dir (str): Directory where updated label files will be saved.
        id_mapping (dict): Dictionary mapping old_class_id -> new_class_id.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    changes_counter = defaultdict(int)  # Track how many replacements per mapping

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)

            with open(input_file, 'r') as file:
                lines = file.readlines()

            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts:  # skip empty lines
                    class_id = int(parts[0])
                    if class_id in id_mapping:
                        changes_counter[(class_id, id_mapping[class_id])] += 1
                        parts[0] = str(id_mapping[class_id])  # apply mapping
                    updated_lines.append(' '.join(parts))

            with open(output_file, 'w') as file:
                file.write('\n'.join(updated_lines))

    # Print summary report
    print("\nSummary of changes:")
    if changes_counter:
        for (old_id, new_id), count in changes_counter.items():
            print(f"  {old_id} → {new_id} : {count} replacements")
    else:
        print("  No changes were made.")


# Example usage
input_dir = r"D:\dataset\roboflow\My First Project.v1i.yolov9\selected_deform"
output_dir = r"D:\dataset\roboflow\My First Project.v1i.yolov9\selected_deform"

# Mapping dictionary: old_id -> new_id
id_mapping = {
    # 19:0,
    # 13:2,
    # 17:5,
    # 6:6,
    # 9:7,
    # 10:8,
    # 11:9,
    # 12:10,
    # 14:11,
    # 15:12,
    # 7: 13,
    # 8: 14,

    # """2.2.0 to 3.3.0"""
    # 0:0,
    # 1:1,
    # 2:13,
    # 3:3,
    # 4:4,
    # 5:5,
    # 6:6,
    # 7:9,
    # 8:10,
    # 9:11,
    # 10:12,
    # 11:14,
    # 12:15,
    # 13:7,
    # 14:8,


    # #changes to be made 3.3.0 to 2.3.1
    # 0:0,
    # 1:1,
    # 2:2,
    # 3:3,
    # 4:4,
    # 5:5,
    # 6:6,
    # 7:9,
    # 8:10,
    # 9:11,
    # 10:12,
    # 11:14,
    # 12:13,
    # 13:7,
    # 14:8

    #changes for roboflow dataset deformation
    5:11,
    6:11

}

update_class_ids(input_dir, output_dir, id_mapping)


# import os
# from collections import defaultdict

# def remove_class_ids(input_dir, output_dir, remove_ids):
#     """
#     Remove lines from YOLO label files if the class ID matches any in remove_ids.

#     Args:
#         input_dir (str): Directory with input YOLO .txt label files.
#         output_dir (str): Directory where cleaned label files will be saved.
#         remove_ids (set): Set of class IDs to remove.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     removed_counter = defaultdict(int)

#     for filename in os.listdir(input_dir):
#         if filename.endswith('.txt'):
#             input_file = os.path.join(input_dir, filename)
#             output_file = os.path.join(output_dir, filename)

#             with open(input_file, 'r') as f:
#                 lines = f.readlines()

#             cleaned_lines = []
#             for line in lines:
#                 parts = line.strip().split()
#                 if not parts:
#                     continue

#                 class_id = int(parts[0])
#                 if class_id in remove_ids:
#                     removed_counter[class_id] += 1
#                     continue  # skip this line completely

#                 cleaned_lines.append(line.strip())

#             # Save updated label file
#             with open(output_file, 'w') as f:
#                 f.write("\n".join(cleaned_lines))

#     # Summary
#     print("\nRemoved class IDs:")
#     if removed_counter:
#         for cid, count in removed_counter.items():
#             print(f"  Class {cid}: removed {count} lines")
#     else:
#         print("  No matching classes found for removal.")


# # Example usage
# input_dir = r"D:\dataset\roboflow\My First Project.v1i.yolov9\selected_deform"
# output_dir = r"D:\dataset\roboflow\My First Project.v1i.yolov9\selected_deform"

# remove_ids = {0, 3, 7}  # classes to remove

# remove_class_ids(input_dir, output_dir, remove_ids)
