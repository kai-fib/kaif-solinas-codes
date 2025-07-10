import os

def shorten_yolo_format(input_folder):
    """
    Shortens the YOLO format from six decimal places to three decimal places.
    
    Args:
        input_folder (str): The folder containing YOLO .txt files.
    """
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_file_path = os.path.join(input_folder, filename)
            
            # Read the content of the file
            with open(input_file_path, 'r') as file:
                lines = file.readlines()

            # Process each line and shorten the decimal places
            shortened_lines = []
            for line in lines:
                parts = line.strip().split()

                # Ensure the line contains the correct number of parts
                if len(parts) == 5:
                    class_id = parts[0]
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Format to three decimal places
                    shortened_line = f"{class_id} {x_center:.3f} {y_center:.3f} {width:.3f} {height:.3f}\n"
                    shortened_lines.append(shortened_line)

            # Write the shortened content back to the file
            with open(input_file_path, 'w') as file:
                file.writelines(shortened_lines)

# Example usage
input_folder = "C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/dataset_from_int/roboflow/Classes/Roots/"
shorten_yolo_format(input_folder)
