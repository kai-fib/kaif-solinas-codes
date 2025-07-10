import os

def find_label_in_files(directory, target_label="20"):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)

            # Skip empty files
            if os.path.getsize(file_path) == 0:
                continue
            
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read().strip()
                if target_label in content.split():  # Ensures it matches as a standalone value
                    print(f"Label '{target_label}' found in: {filename}")

# Example usage
directory_path = "C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Stage-III(Dataset)/WRC_bgrd/labels/train/"  # Change this to your directory
find_label_in_files(directory_path)
