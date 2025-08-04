def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))


from PIL import Image

def apply_color_to_image(image_path, hex_code, output_path):
    # Convert hex to RGB
    rgb_color = hex_to_rgb(hex_code)
    
    # Open the image
    image = Image.open(image_path).convert('RGBA')
    
    # Create a new image with the same size and the specified color
    colored_image = Image.new('RGBA', image.size, rgb_color + (255,))
    
    # Blend the images
    blended_image = Image.blend(image, colored_image, alpha=0.5)
    
    # Save the result
    blended_image.save(output_path)

# Example usage
#apply_color_to_image('1.png', '#0C320E', 'output_1.png')

#apply_color_to_image('3.png', '#FFD69F', 'output_3.png')

#apply_color_to_image('3.png', '#0A3619', 'output_3.png')

apply_color_to_image('Defect 5.png', '#052707', 'output_Defect 5.png.png')


# def hex_to_rgb(hex):
#      x = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
#      return x

# kaif = hex_to_rgb('0c320e') # (255, 165, 1)
# print(kaif)


     
     



