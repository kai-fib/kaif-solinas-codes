import os
import glob

labels_path = 'C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Video_dataset/data/rename/stone/'

# Define class mappings
class_mapping = {
    7: 6
}

def relabel_files(path, class_mapping):
    label_files = glob.glob(os.path.join(path, '**', '*.txt'), recursive=True)
    
    for file in label_files:
        # Skip empty files
        if os.path.getsize(file) == 0:
            continue

        updated_lines = []
        with open(file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Skip lines that are empty or malformed
                if not parts or not parts[0].isdigit():
                    continue
                
                class_id = int(parts[0])
                if class_id in class_mapping:
                    parts[0] = str(class_mapping[class_id])  # Update class ID
                
                updated_lines.append(' '.join(parts))

        # Only rewrite if there were valid updates
        if updated_lines:
            with open(file, 'w') as f:
                f.write('\n'.join(updated_lines) + '\n')

relabel_files(labels_path, class_mapping)
