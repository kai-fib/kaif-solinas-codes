import os

import os
from collections import Counter

def count_class_ids(input_dir):
    class_counts = Counter()

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if parts:  # make sure line not empty
                        class_id = int(parts[0])  # first number is class id
                        class_counts[class_id] += 1

    # Print summary
    for class_id, count in sorted(class_counts.items()):
        print(f"{class_id} -> {count}")

# Example usage
count_class_ids(
    r"D:\dataset\wrc_sewer_2.3.3\labels\val"
)


# def update_class_id(input_dir, output_dir, old_class_id, new_class_id):
    
#     for filename in os.listdir(input_dir):
#         if filename.endswith('.txt'):
#             input_file = os.path.join(input_dir, filename)
#             output_file = os.path.join(output_dir, filename)
#             with open(input_file, 'r') as file:
#                 lines = file.readlines()
        
#             updated_lines = []
#             for line in lines:
#                 parts = line.strip().split()
#                 if parts:  # Ensure the line is not empty
#                     if int(parts[0]) == old_class_id:
#                         parts[0] = str(new_class_id)  # Update class ID
#                     updated_lines.append(' '.join(parts))
        
#             with open(output_file, 'w') as file:
#                 file.write('\n'.join(updated_lines))

# # Example usage
# #input_file = 'labels.txt'  # Input YOLO label file
# #output_file = 'updated_labels.txt'  # Output file with updated class IDs
# input_dir = r'C:\Users\Kaif Ibrahim\Desktop\solinas_downloads\Stage-III(Dataset)\wrc_sewer_3.0.0\labels\train'
# output_dir = r'C:\Users\Kaif Ibrahim\Desktop\solinas_downloads\Stage-III(Dataset)\wrc_sewer_3.0.0\labels\train'
# old_class_id = 19 # Class ID to replace
# new_class_id = 0  # New class ID

# update_class_id(input_dir, output_dir, old_class_id, new_class_id)


#   0: Attached_deposit(DEZ,3)         19 -> 0
#   1: Hole(H,3)                      nope sewer
#   2: Joint_Displacement(JD,3)        13 -> 2
#   3: Surface_damage(S,2)          **no change**
#   4: Root(R,3)                    **no change**
#   5: Settled_Deposits(DE,3)        17 -> 5
#   6: Other Obstacles(OB,3))        st(6),PB(7),CB(8) -> 6
#   7: Fracture(F,3)                 9 ->  7
#   8: Infiltration(I,4)             10 -> 8
#   9: Deformed(D,3)                 11 -> 9
#   10: Crack(C,2)                   12 -> 10
#   11: Broken(B,4)                  14 -> 11
#   12: Collapse(XP,5)               15 -> 12
#   13: Pb  PB(7)                    7  -> 13   
#   14: CB  CB(8)                    8  -> 14