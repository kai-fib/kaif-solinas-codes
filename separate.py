import os
import re
import shutil

# Define the regex pattern
pattern = re.compile(r"Sluge Accumulation_(\d+)\.(\w+)")

# Define the source directory (cwd) and the target directory
cwd = 'C:/Users/Kaif Ibrahim/Desktop/3d_vid/sludge/sludge/'
target_folder = os.path.join(cwd, "arg_needed")

# Create the target directory if it doesn't exist
os.makedirs(target_folder, exist_ok=True)

# Iterate through all files in the current directory
for file_name in os.listdir(cwd):
    # Check if the file name matches the pattern
    if pattern.match(file_name):
        source_path = os.path.join(cwd, file_name)
        target_path = os.path.join(target_folder, file_name)
        # Move the file to the target directory
        shutil.move(source_path, target_path)
        print(f"Moved: {file_name} -> {target_path}")

print("File separation complete.")
