import os
import xml.etree.ElementTree as ET

# Dictionary that maps class names to IDs

#wrc
class_name_to_id_mapping = {
    "Encrustation(DEE,3)": 0,
    "Encrustation(E,3)":0,
    "	0: Encrustation(DEE,3)":0,
    "Ferrule(H,4)": 1,
    "Ferrule(H,4)":1,
    "	1: Ferrule(H,4)":1,
    "Joint_Displacement_large(JDL,4)": 2,
    "Joint_Displacement(JDL,4)":2,
    "	2: Joint_Displacement(JDL,4)":2,
    "Surface_damage(SW,3)": 3,
    "Surface_damage(SW,2) ":3,#"
    "Surface_damage(SW,2)":3,
    "Surface_damage(SW,2)w":3,
    "Surface_damage(SW,2)":3,
    "	2: Encrustation(DEE,3)":2,
    "Root_Blockage(RF,3)": 4,
    "Sludge_Accumulation(DE,3)": 5,
    "Sludge_Accumulation(DER,3)":5,
    "Sludge_Accumulation(DES,3)":5,
    "	5: Sludge_Accumulation(DE,3)":5,
    "5: Sludge_Accumulation(DE,3)":5,
    "Stone(OBB,4)": 6,
    "	6: Stone(OBB,4)":6,
    "6: Stone(OBB,4)":6,
    "6:Stone (OBB,2)":6,
    "Partial_Blockage (B,4)": 7,
    "Partial_Blockage(OBP,3)": 7,
    "Partial_Blockage(OBP,3)":7,
    "	7: Partial_Blockage(OBP,3)":7,
    "Complete Blockage (B,4)": 8,
    "Complete Blockage(OBZ,4)":8,
    "Fracture(F,4)": 9,
    "Fracture_(F,4)": 9,
    "Infiltration (IG,4)": 10,
    "Infiltration (IR,3)": 10,
    "Deformed (D,3)": 11,
    "Deformed(D,4)":11,
    "Crack(CL,2)":12,
    "Crack (C,2)":12,
    "Crack(C,1)":12,
    "Joint_Displacement_medium(JDM,3)":13,
    "Broken(B,4)":14,
    "Collapse(XP,5)":15,
    "Fouling(DEF,3)":16,
    "Gravel(DER,3)":17,
    "	17: Gravel(INGG,3)":17,
    "Gravel(INGG,3)":17,
    "17: Gravel(INGG,3)":17,
    "	17: Gravel(INGG,3)":17,
    "	17: Gravel(INGG,3)":17,
    "	17: Gravel(INGG,3)":17,
    "Vermin(V,3)":18,
    "Attached_deposit(DEZ,3)":19,
    "Attached_deposit(DEZ,3)":19,
    "Fine_Roots(RF)":20

}



#NASSCO
# class_name_to_id_mapping = {
#     "Encrustation(DAE,3)": 0,
#     "Ferrule(H,2)": 1,
#     "Surface_damage(SRI,2)": 2,
#     "Root_Blockage(RFB,3)": 3,
#     "Sludge_Accumulation(DSGV,3)": 4,
#     "Fracture_(F,3)": 5,
#     "Infiltration (ID,3)": 6,
#     "Deformed (D,3)": 7,
#     "Crack_L (CL,1)": 8,
#     "Crack_C (CC,1)": 9,
# }


'''
#Europe
class_name_to_id_mapping = {
    "Encrustation(BBB,3)": 0,
    "Joint_Displacement(BAJ,4)": 1,
    "Surface_damage(BAF,2)": 2,
    "Root_Blockage(BBA,3)": 3,
    "Sludge_Accumulation(BBC,3)": 4, #do change to BBC
    "Partial_Blockage (BBE,4)": 5,
    "Complete Blockage (BBE,4)": 6,
    "Fracture_(BAC,3)": 7,
    "Infiltration (BBF,3)": 8,
    "Deformed (BAA,3)": 9,
    "Crack_L (BAB,1)":10,
    "Crack_C (BAB,1)":11,
}
'''
def convert_pascal_voc_to_yolo(xml_file, class_name_to_id_mapping):
    tree = ET.parse(xml_file)
    root = tree.getroot()

   # Assuming image dimensions are known or fixed
    #image_width = 640
    #image_height = 480
    
    # Extract image size from XML file
    size = root.find('size')
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)

    # Generate the output file path
    yolo_file = os.path.splitext(xml_file)[0] + ".txt"

    with open(yolo_file, "w") as f:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            
            if class_name not in class_name_to_id_mapping:
                continue
            
            class_id = class_name_to_id_mapping[class_name]

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # Calculate center coordinates and box dimensions
            width = xmax - xmin
            height = ymax - ymin
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2

            # Normalize coordinates
            x_center_normalized = x_center / image_width
            y_center_normalized = y_center / image_height
            width_normalized = width / image_width
            height_normalized = height / image_height

            # Write to file in YOLO format with three decimal places
            f.write(f"{class_id} {x_center_normalized:.3f} {y_center_normalized:.3f} {width_normalized:.3f} {height_normalized:.3f}\n")

def convert_folder_to_yolo(xml_folder, class_name_to_id_mapping):
    for filename in os.listdir(xml_folder):
        if filename.endswith(".xml"):
            xml_file = os.path.join(xml_folder, filename)
            convert_pascal_voc_to_yolo(xml_file, class_name_to_id_mapping)

# Example usage
xml_folder = 'C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/Video_dataset/Revamp-V9/water_2.1.0/images/stone/aug_stone/'

#D:/SOLINAS DOWNLOADS/Part-2 augmentation/sluge/xml_sluge_Nassco
convert_folder_to_yolo(xml_folder, class_name_to_id_mapping)

