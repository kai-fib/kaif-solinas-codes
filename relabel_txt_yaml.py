import os
import glob


labels_path = "C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Stage-III(Dataset)/WRC_bgrd_Sewer/labels/train/"
#D:\solinas work\pendrive\delete\train_folder\joint
#E:/delete/train_folder/fracture/
#
#D:/SOLINAS DOWNLOADS/NASSCO/labels/train
# Define your class mappings here
class_mapping = {
    20: 4,
          # Encrustation(ED,3) remains the same
          # Change class 10 to 1 for Infiltration (IP,3)
    # Add more mappings as needed
    # old_class_id: new_class_id
}

def relabel_files(path, class_mapping):
    # Recursively find all .txt label files in the directory
    label_files = glob.glob(os.path.join(path, '**', '*.txt'), recursive=True)
    for file in label_files:
        with open(file, 'r') as f:
            lines = f.readlines()
        with open(file, 'w') as f:
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                if class_id in class_mapping:
                    parts[0] = str(class_mapping[class_id])  # Change class ID
                f.write(' '.join(parts) + '\n')

# Relabel the dataset
relabel_files(labels_path, class_mapping)