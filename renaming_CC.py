import os
import re

def rename_files(directory):
    #fracture_539_0005
    #joint_image_80_0001
    #joint_image_152_0009
    # Regular expression to match the pattern 'fracture_surface_damage_<count>.<ext>'

    #pattern = re.compile(r"Circumferential_Crack_(\d+_\d+)\.(\w+)")
    
    pattern = re.compile(r"cc_\((\d+)\)\.(\w+)")
    
    # Iterate over all files in the specified directory
    for filename in os.listdir(directory):
        # Check if the filename matches the pattern
        match = pattern.match(filename)
        if match:
            count, extension = match.groups()
    
            
            # Create the new filename with 'fracture_<count>.<ext>'
            new_filename = f"Gravels_{count}.{extension}"
        
            # Construct full file paths
            old_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_filepath, new_filepath)
            print(f"Renamed: {filename} -> {new_filename}")

# Usage example
directory = 'F:/gravel/' 
#D:/SOLINAS DOWNLOADS/Dataset to be Given/Make_3/Final/crack/aug_karo/   --1st
#D:\SOLINAS DOWNLOADS\Dataset to be Given\Make_3\pendrive\fracture\16th_oct --2nd
#D:/SOLINAS DOWNLOADS/Dataset to be Given/Make_3/infil/augment_cheyali/

rename_files(directory)