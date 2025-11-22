import os

def delete_class_id_lines(input_dir, target_class_id):
    """
    Delete lines starting with target_class_id from all .txt files in input_dir.
    
    :param input_dir: Folder containing YOLO label .txt files
    :param target_class_id: Class ID to remove (int)
    """
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_dir, filename)

            with open(file_path, 'r') as file:
                lines = file.readlines()

            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts:
                    # keep the line only if its class id is not target_class_id
                    if parts[0].isdigit() and int(parts[0]) != target_class_id:
                        updated_lines.append(line.strip())

            # overwrite the file with cleaned lines
            with open(file_path, 'w') as file:
                file.write("\n".join(updated_lines))
                if updated_lines:  # add newline at the end if not empty
                    file.write("\n")

# ---------------- Example usage ----------------
input_dir = r'D:\wrc_2.2.1(ob)\six\val'   # folder with your .txt files
target_class_id = 16         # class id to delete

delete_class_id_lines(input_dir, target_class_id)
