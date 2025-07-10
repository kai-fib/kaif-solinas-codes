# import os

# # Define the directory containing the files
# directory = 'C:/Users/Kaif Ibrahim/Desktop/roots/fine/'

# # Starting middle number
# start_middle_number = 0
# # Number of files per middle number
# files_per_group = 5

# # List all files in the directory and sort them
# files = sorted([f for f in os.listdir(directory) if f.startswith('Root_Blockage_') and f.endswith('.txt')])

# # Loop through all files and rename them with incrementing middle numbers every 6 files
# for i, filename in enumerate(files):
#     # Calculate the current middle number based on the file index
#     current_middle_number = start_middle_number + (i // files_per_group)
#     # Calculate the count part, formatted with leading zeros
#     count_part = f"{(i % files_per_group) + 1:05}"
    
#     # Create the new filename
#     new_filename = f"Fine_Roots_{current_middle_number}_{count_part}.txt"
    
#     # Rename the file
#     old_filepath = os.path.join(directory, filename)
#     new_filepath = os.path.join(directory, new_filename)
    
#     os.rename(old_filepath, new_filepath)
#     print(f"Renamed: {filename} -> {new_filename}")













import os

# Define the directory containing the files
directory = 'C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Video_dataset/Revamp-V9/water_2.1.0/images/ad_sewer/aug_ad/'

# List all .txt files in the directory and sort them
files = sorted([f for f in os.listdir(directory) if f.endswith('.jpg')])

# Loop through all files and rename them with a new sequential name
for i, filename in enumerate(files, start=156):
    new_filename = f"Attached_deposit_s_{i:05}.jpg"
    
    # Define full paths
    old_filepath = os.path.join(directory, filename)
    new_filepath = os.path.join(directory, new_filename)
    
    # Remove the existing file if it already exists (force rename)
    if os.path.exists(new_filepath):
        os.remove(new_filepath)  # Deletes the file with the same name
    
    # Rename the file
    os.rename(old_filepath, new_filepath)
    print(f"Renamed: {filename} -> {new_filename}")



