import os

def create_txt_files(num_files, directory="C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Video_dataset/Revamp-V9/images/prob/aug_back/"):
    os.makedirs(directory, exist_ok=True)
    for i in range(1, num_files + 1):
        file_path = os.path.join(directory, f"Background_{i}.txt")
        open(file_path, "w").close()
    print(f"{num_files} empty text files created in '{directory}' folder.")

# Generate 93 empty text files
create_txt_files(810)
