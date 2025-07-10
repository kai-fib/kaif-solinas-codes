import os

# def check_files_for_codes(input_dir, codes=(18,)):  # Add comma to make it a tuple
#     for filename in os.listdir(input_dir):
#         if filename.endswith('.txt'):
#             file_path = os.path.join(input_dir, filename)
#             with open(file_path, 'r') as file:
#                 for line in file:
#                     if any(line.startswith(f'{code} ') for code in codes):
#                         print(f'Code found in: {filename}')
#                         break

# check_files_for_codes(
#     'C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Stage-III(Dataset)/wrc_sewer_2.2.0/labels/val/',
#     codes=(19,)  # Example: scan for multiple class codes
# )

# import os

# def remove_lines_with_codes(input_dir, codes=(0)):
#     for filename in os.listdir(input_dir):
#         if filename.endswith('.txt'):
#             file_path = os.path.join(input_dir, filename)
#             with open(file_path, 'r') as file:
#                 lines = file.readlines()

#             # Filter out lines starting with any of the codes
#             filtered_lines = [
#                 line for line in lines 
#                 if not any(line.startswith(f'{code} ') for code in codes)
#             ]

#             # If any lines were removed, print filename and rewrite file
#             if len(filtered_lines) != len(lines):
#                 print(f'Removed lines from: {filename}')
#                 with open(file_path, 'w') as file:
#                     file.writelines(filtered_lines)

# remove_lines_with_codes(
#     'C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Stage-III(Dataset)/Wrc_Sewer_2.1.0/labels/train/'
# )

# import os

# def remove_lines_with_codes(input_dir, codes=(0,1,)):
#     for filename in os.listdir(input_dir):
#         if filename.endswith('.txt'):
#             file_path = os.path.join(input_dir, filename)
#             with open(file_path, 'r') as file:
#                 lines = file.readlines()

#             # Filter out lines starting with any of the codes
#             filtered_lines = [
#                 line for line in lines 
#                 if not any(line.startswith(f'{code} ') for code in codes)
#             ]

#             # If any lines were removed, print filename and rewrite file
#             if len(filtered_lines) != len(lines):
#                 print(f'Removed lines from: {filename}')
#                 with open(file_path, 'w') as file:
#                     file.writelines(filtered_lines)


# remove_lines_with_codes(
#     'C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Stage-III(Dataset)/Wrc_Sewer_2.1.0/labels/val/'
# )

def update_class_id(input_dir, output_dir, old_class_id, new_class_id):
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            with open(input_file, 'r') as file:
                lines = file.readlines()
        
            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts:  # Ensure the line is not empty
                    if int(parts[0]) == old_class_id:
                        parts[0] = str(new_class_id)  # Update class ID
                    updated_lines.append(' '.join(parts))
        
            with open(output_file, 'w') as file:
                file.write('\n'.join(updated_lines))

# Example usage
#input_file = 'labels.txt'  # Input YOLO label file
#output_file = 'updated_labels.txt'  # Output file with updated class IDs
input_dir = 'C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Stage-III(Dataset)/wrc_water_2.2.0/labels/val/'
output_dir = 'C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Stage-III(Dataset)/wrc_water_2.2.0/labels/val/'
old_class_id = 18  # Class ID to replace
new_class_id = 16  # New class ID

update_class_id(input_dir, output_dir, old_class_id, new_class_id)