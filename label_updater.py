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
input_dir = r"C:\Users\Kaif Ibrahim\Desktop\solinas_downloads\Stage-III(Dataset)\wrc_sewer_3.0.0\labels\train"
output_dir = r"C:\Users\Kaif Ibrahim\Desktop\solinas_downloads\Stage-III(Dataset)\wrc_sewer_3.0.0\labels\train"

# Mapping dictionary: old_id -> new_id
id_mapping = {
    19: 0,
    18: 0,
    13: 2,
    17: 5,
    7: 6, 8: 6,
    9: 7,
    10: 8,
    11: 9,
    12: 10,
    14: 11,
    15: 12,
    16: 0
}

update_class_ids(input_dir, output_dir, id_mapping)
