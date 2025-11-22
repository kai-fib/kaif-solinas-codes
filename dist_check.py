# prev = '0000010.03M'
# import easyocr
# import re
# reader = easyocr.Reader(['en'])  # this needs to run only once to load the model into memory
# mid_frame = r"D:\no_dist\_(DE-S,3)_(NA)_(0-3-47)_(NA)_(L).png"
# results = reader.readtext(mid_frame, detail=0)
# print(results)
# dist_bot = "0.0 m"
# for j, text in enumerate(results):
#     text = text.replace('\xa0', ' ').replace('~', '').replace('O', '0').replace('o', '0').replace(";",'.').replace("?",".").replace(',','.').strip()
#     #print(f"index is : {j}, text is {text}")
#     #together_text = re.search(r"(\d+\s*\.?\s*\d*)\s*(m|ft|Miles)\b", text, flags=re.IGNORECASE)
#     together_text = re.search(r"(\b\d+\s*\.\s*\d+\s*)\s*(m|ft|Miles)\b", text, flags=re.IGNORECASE)
#     if together_text:
#         num = together_text.group(1)  
#         unit = together_text.group(2)  
#         num1 = re.sub(r"\s*\.\s*", ".", num)

#         dist_bot = f"{num1} {unit}"
#         break
    
#     #together_text1 = re.search(r"(ft|Feet|Miles|Mile)\s*(\d*\s*\.?\s*\d+)\b", text, flags=re.IGNORECASE)
#     together_text1 = re.search(r"(ft|Feet|Miles|Mile)\s*(\b\d+\s*\.\s*\d+\s*)\b", text, flags=re.IGNORECASE)
#     if together_text1:
#         unit = together_text1.group(1)  
#         num = together_text1.group(2)
#         num1 = re.sub(r"\s*\.\s*", ".", num)

#         dist_bot = f"{num1} {unit}"  
#         break
    
#     #initial_unit_equal = re.search(r"(Feet|MILES|Miles|miles|Mile|ft)\s*=\s*(\d*\s*\.?\s*\d+)", text, flags=re.IGNORECASE)
#     initial_unit_equal = re.search(r"(Feet|MILES|Miles|miles|Mile|ft)\s*=\s*(\b\d+\s*\.\s*\d+\s*)", text, flags=re.IGNORECASE)
#     if initial_unit_equal:
#         unit = initial_unit_equal.group(1)  
#         num = initial_unit_equal.group(2)  
#         num1 = re.sub(r"\s*\.\s*", ".", num)
#         dist_bot = f"{num1} {unit}"  
#         break
    
#     #initial_unit_colon = re.search(r"(Feet|Miles|MILES|Mile|ft)\s*:\s*(\d*\s*\.?\s*\d+)", text, flags=re.IGNORECASE)
#     initial_unit_colon = re.search(r"(Feet|Miles|MILES|Mile|ft)\s*:\s*(\b\d+\s*\.\s*\d+\s*)", text, flags=re.IGNORECASE)
#     if initial_unit_colon:
#         unit = initial_unit_colon.group(1)  
#         num = initial_unit_colon.group(2)  
#         num1 = re.sub(r"\s*\.\s*", ".", num)

#         dist_bot = f"{num1} {unit}"  
#         break
    

#     #Merge with next token if it's a unit
#     if j + 1 < len(results) and re.match(r"^(m|ft|Miles)$", results[j + 1].strip(),flags=re.IGNORECASE):
#         #text += results[j + 1].strip()
#         text = text + ' ' + results[j + 1].strip()
#         text = re.sub(r"\s*\.\s*", ".", text)
#     matchp = re.search(r"\b\d+\s*\.\s*\d+\s*(?:m|M|ft|FT|Ft)\b", text)    #
#     #matchp = re.search(r'\b\d+(?:\s*\.\s*\d+)?(?:\s*\d+)?\s*(?:m|ft)\b',text, flags=re.IGNORECASE)
#     if matchp:
#         dist_bot = matchp.group()
        
#     if (j != 0) and re.match(r"^(ft[:=]|Miles[:=])$", results[j - 1].strip(),flags=re.IGNORECASE):
#         #text += results[j + 1].strip()
#         text = text + ' ' + results[j - 1].strip()
#     matchp = re.search(r"\b\d+\s*\.\s*\d+\s*(?:m|M|ft|FT|Ft)\b", text)
#     #matchp = re.search(r'\b\d+(?:\s*\.\s*\d+)?(?:\s*\d+)?\s*(?:m|ft)\b',text, flags=re.IGNORECASE)
#     if matchp:
#         dist_bot = matchp.group()

# # print(dist_bot)
# """for j, text in enumerate(results):
#     text = text.replace('\xa0', ' ').replace('~', '').replace('O', '0').replace('o', '0').replace(",",".").replace(";",'.').replace("?",".").strip()
    
#     together_text = re.search(r"(\d+\.?\d*)(m|M|ft|FT|Ft)\b", text)
#     if together_text:
#         num = together_text.group(1)
#         unit = together_text.group(2)
#         dist_bot = f"{num} {unit}"
#         break
#     # Merge with next token if it's a unit
#     if j + 1 < len(results) and re.match(r"^(m|M|ft|FT|Ft)$", results[j + 1].strip()):
#         #text += results[j + 1].strip()
#         text = text + ' ' + results[j + 1].strip()
#     matchp = re.search(r"\b\d+\.?\d*\s*(?:m|M|ft|FT|Ft)\b", text)
#     if matchp:
#         num, unit = re.match(r"(\d+\.?\d*)\s*(m|M|ft|FT|Ft)", matchp.group()).groups()
#         dist_bot = (f"{num} {unit}")
#         #dist_bot = matchp.group()
#         break
# print(dist_bot)"""


# for j, text in enumerate(results):
#     matchp = re.search(r"(?:\b\s*\-?\d{1,5}\.\d{1,2}\s*(?:m|M|ft|FT|Ft)\b)|(?:(?:FEET|Feet|feet|Mile|MILE|mile)[:=]\s*\-?\d{1,5}\.\d{1,2}\b)", text)

#     if matchp:
#         dist_bot = matchp.group()
#         dist_float=float(re.search(r"[-+]?\d+\.\d+|[-+]?\d+",dist_bot).group())#to extract the distance
#         current_distance_found = True
#         break
#     text = text.replace('\xa0', ' ').replace('~', '').replace('O', '0').replace('o', '0').strip()

#     # Merge with next token if it's a unit
#     if j!=0 and re.match(r"^(FEET[:=]|Feet[:=]|feet[:=]|Mile[:=]|MILE[:=]|mile[:=])$", results[j - 1].strip()):
#         text+=results[j-1].strip()
#         matchp = re.search(r"(?:\b\s*\-?\d{1,5}\.\d{1,2}\s*(?:m|M|ft|FT|Ft)\b)|(?:(?:FEET|Feet|feet|Mile|MILE|mile)[:=]\s*\-?\d{1,5}\.\d{1,2}\b)", text)
#         if matchp:
#             dist_bot = matchp.group()
#             dist_float=float(re.search(r"[-+]?\d+\.\d+|[-+]?\d+",dist_bot).group())#to extract the float distance
#             current_distance_found = True
#             break

#     if j + 1 < len(results) and re.match(r"^(m|M|ft|FT|Ft)$", results[j + 1].strip()):##to conacatenate in case the next value in the list results is 'm','M' etc
#         text += results[j + 1].strip()
#         matchp = re.search(r"(?:\b\s*\-?\d{1,5}\.\d{1,2}\s*(?:m|M|ft|FT|Ft)\b)|(?:(?:FEET|Feet|feet|Mile|MILE|mile)[:=]\s*\-?\d{1,5}\.\d{1,2}\b)", text)
#         if matchp:
#             dist_bot = matchp.group()
#             dist_float=float(re.search(r"[-+]?\d+\.\d+|[-+]?\d+",dist_bot).group())#to extract the float distance
#             current_distance_found = True
#             break
# print(dist_bot)




# import easyocr
# import re
# reader = easyocr.Reader(['en'])  # this needs to run only once to load the model into memory
# mid_frame = r"C:\Users\Kaif Ibrahim\Downloads\Sludge_Accumulation(DES,3)_(0-0-40)_(0.0 m).png"
# results = reader.readtext(mid_frame, detail=0)
# dist_bot = "0.0 m"
# ref_format = None

# for j, text in enumerate(results):
#     text = text.replace('\xa0', ' ').replace('~', '').replace('O', '0').replace('o', '0').strip()

#     matchp = re.search(r"(?:\b\s*\-?\d{1,5}\.\d{1,2}\s*(?:m|M|ft|FT|Ft)\b)|(?:(?:FEET|Feet|feet|Mile|MILE|mile)[:=]\s*\-?\d{1,5}\.\d{1,2}\b)", text)
#     if matchp:
#         dist_bot = matchp.group()
#         dist_float=float(re.search(r"[-+]?\d+\.\d+|[-+]?\d+",dist_bot).group())#to extract the distance
#         current_distance_found = True
#         break

#     if j!=0 and re.match(r"^(FEET[:=]|Feet[:=]|feet[:=]|Mile[:=]|MILE[:=]|mile[:=])$", results[j - 1].strip()): #pre-fix 
#         text+=results[j-1].strip()
    
#     matchp = re.search(r"(?:\b\s*\-?\d{1,5}\.\d{1,2}\s*(?:m|M|ft|FT|Ft)\b)|(?:(?:FEET|Feet|feet|Mile|MILE|mile)[:=]\s*\-?\d{1,5}\.\d{1,2}\b)", text)
#     if matchp:
#         dist_bot = matchp.group()
#         break

#     # Merge with next token if it's a unit
#     if j + 1 < len(results) and re.match(r"^(m|M|ft|FT|Ft)$", results[j + 1].strip()): # suffix 
#         text += results[j + 1].strip()
    
#     matchp = re.search(r"(?:\b\s*\-?\d{1,5}\.\d{1,2}\s*(?:m|M|ft|FT|Ft)\b)|(?:(?:FEET|Feet|feet|Mile|MILE|mile)[:=]\s*\-?\d{1,5}\.\d{1,2}\b)", text)
#     if matchp:
#         dist_bot = matchp.group()

#                 # Store the format of the first valid result as reference
#         if ref_format is None:
#             ref_format = dist_bot  # e.g., '0202.50 ft'
#         break


#     elif ref_format is not None and re.match(r'^\d{6}\s*(ft|FT|Ft|m|M)?$', text):
#         digits = re.search(r'\d{6}', text).group()
#         unit = re.search(r'(ft|FT|Ft|m|M)', text)
#         unit = unit.group() if unit else ref_format[-2:]  # fallback to unit in reference

#         # Use decimal position from ref_format (e.g. 4 digits + 2 decimals)
#         parts = ref_format.split('.')[0]  # get '0202' from '0202.50 ft'
#         decimal_pos = len(parts)

#         dist_bot = f"{digits[:decimal_pos]}.{digits[decimal_pos:]} {unit}"

# print(results)
# print(dist_bot)

# print(results[1].strip())


# import easyocr
# import re
# import os

# reader = easyocr.Reader(['en'])  # Load once
# ref_format = None
# folder_path = r"C:\Users\Kaif Ibrahim\Desktop\sample"

# for filename in sorted(os.listdir(folder_path)):
#     if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#         continue

#     image_path = os.path.join(folder_path, filename)
#     print(f"\nProcessing: {filename}")
    
#     results = reader.readtext(image_path, detail=0)
#     dist_bot = "0.0 m"

#     for j, text in enumerate(results):
#         text = text.replace('\xa0', ' ').replace('~', '').replace('O', '0').replace('o', '0').strip()

#         # Combine with previous if unit keyword
#         if j != 0 and re.match(r"^(FEET[:=]|Feet[:=]|feet[:=]|Mile[:=]|MILE[:=]|mile[:=])$", results[j - 1].strip()):
#             text += results[j - 1].strip()

#         # Merge with next token if it's a unit
#         if j + 1 < len(results) and re.match(r"^(m|M|ft|FT|Ft)$", results[j + 1].strip()):
#             text += results[j + 1].strip()

#         # Match properly formatted distances
#         matchp = re.search(r"(-?\d{1,5}\.\d{1,2})\s*(m|M|ft|FT|Ft)|(?:FEET|Feet|feet|Mile|MILE|mile)[:=]\s*(-?\d{1,5}\.\d{1,2})", text)
#         if matchp:
#             dist_bot = matchp.group()
#             if ref_format is None:
#                 ref_format = dist_bot
#             break

#         # If bad format like '032220ft' and reference is available
#         elif ref_format is not None and re.match(r'^\d{6}\s*(ft|FT|Ft|m|M)?$', text):
#             digits = re.search(r'\d{6}', text).group()
#             unit = re.search(r'(ft|FT|Ft|m|M)', text)
#             unit = unit.group() if unit else ref_format[-2:]
#             decimal_pos = len(ref_format.split('.')[0])
#             dist_bot = f"{digits[:decimal_pos]}.{digits[decimal_pos:]} {unit}"
#             break
#     print("OCR Results:", results)

#     print("Detected Distance:", dist_bot)


# import easyocr
# import re
# # Initialize EasyOCR
# reader = easyocr.Reader(['en'])
# # Read text from image
# # results = reader.readtext('your_image.jpg')  # Replace with your image path
# results = ['feet3']
# normalized_distances = []
# # Define all supported units
# units = ['feet', 'foot', 'ft', 'm', 'cm', 'mm', 'km', 'inch', 'in']
# for text in results:
#     t = text.lower()
#     # Insert space between number and unit (e.g., 3m -> 3 m, 3.9mm -> 3.9 mm)
#     t = re.sub(r'(\d+(?:\.\d+)?)(?=' + '|'.join(units) + r')', r'\1 ', text)
#     # Insert space between unit and number (e.g., feet3 -> feet 3)
#     t = re.sub(r'(' + '|'.join(units) + r')(\d+(?:\.\d+)?)', r'\1 \2', text)
#     # Normalize spacing around '=' (e.g., feet=3 -> feet = 3)
#     t = re.sub(r'(' + '|'.join(units) + r')\s*=\s*(\d+(?:\.\d+)?)', r'\1 = \2', text)
#     # Check if this line contains a number and a unit
#     if re.search(r'\d', text) and any(u in t for u in units):
#         normalized_distances.append(text)
# print("Extracted & normalized distances:", normalized_distances)



"""DocTR using function """

# # import easyocr
# import re
# from os import listdir
# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor

# model = ocr_predictor(pretrained=True)
# # reader = easyocr.Reader(['en'])  # this needs to run only once to load the model into memory
# inpath = r"D:\no_dist/"
# A = listdir(inpath)
# for i in range(0,len(A)):
#     mid_frame = inpath + A[i]
# #     # mid_frame = r"D:\no_dist\_(DE-S,3)_(NA)_(0-1-22)_(NA)_(L).png"
#     doc = DocumentFile.from_images(mid_frame)

# # # Use OCR predictor
# #     #mid_frame = r"(DE-S,3)_(NA)_(0-0-4)_(NA)_(L)_(0)_(0, 318, 849, 713)_253_clock 7 to 10.png"
# #     # results = reader.readtext(mid_frame, detail=0)
# #     # Use OCR predictor
    

# # # Get the result
#     result = model(doc)
#     # Extract all text
#     results = []
#     for page in result.pages:
#         for block in page.blocks:
#             for line in block.lines:
#                 for word in line.words:
#                     results.append(word.value)
#     # # print(results)
#         #print(results)
#         # together_text = []
#         # together_text1 = []
#         # initial_unit = []
#         #dist_bot = "0.0 m"
#     dist_bot = "NA"
#     #print(f'length of result: {len(results)}')
#     for j, text in enumerate(results):
#         text = text.replace('\xa0', ' ').replace('~', '').replace('O', '0').replace('o', '0').replace(",",".").replace(";",'.').replace("?",".").replace("-"," ").strip()
#         #print(f"index is : {j}, text is {text}")
#         #together_text = re.search(r"(\d+\s*\.?\s*\d*)\s*(m|ft|Miles)\b", text, flags=re.IGNORECASE)
#         together_text = re.search(r"(\b\d+\s*\.\s*\d+\s*)\s*(m|ft|Miles)\b", text, flags=re.IGNORECASE)
#         if together_text:
#             num = together_text.group(1)
#             unit = together_text.group(2)
#             num1 = re.sub(r"\s*\.\s*", ".", num)
#             dist_bot = f"{num1} {unit}"
#             break
#         #together_text1 = re.search(r"(ft|Feet|Miles|Mile)\s*(\d*\s*\.?\s*\d+)\b", text, flags=re.IGNORECASE)
#         together_text1 = re.search(r"(ft|Feet|Miles|Mile)\s*(\b\d+\s*\.\s*\d+\s*)\b", text, flags=re.IGNORECASE)
#         if together_text1:
#             unit = together_text1.group(1)
#             num = together_text1.group(2)
#             num1 = re.sub(r"\s*\.\s*", ".", num)
#             dist_bot = f"{num1} {unit}"
#             break
#         #initial_unit_equal = re.search(r"(Feet|MILES|Miles|miles|Mile|ft)\s*=\s*(\d*\s*\.?\s*\d+)", text, flags=re.IGNORECASE)
#         initial_unit_equal = re.search(r"(Feet|MILES|Miles|miles|Mile|ft)\s*=\s*(\b\d+\s*\.\s*\d+\s*)", text, flags=re.IGNORECASE)
#         if initial_unit_equal:
#             unit = initial_unit_equal.group(1)
#             num = initial_unit_equal.group(2)
#             num1 = re.sub(r"\s*\.\s*", ".", num)
#             dist_bot = f"{num1} {unit}"
#             break
#         #initial_unit_colon = re.search(r"(Feet|Miles|MILES|Mile|ft)\s*:\s*(\d*\s*\.?\s*\d+)", text, flags=re.IGNORECASE)
#         initial_unit_colon = re.search(r"(Feet|Miles|MILES|Mile|ft)\s*:\s*(\b\d+\s*\.\s*\d+\s*)", text, flags=re.IGNORECASE)
#         if initial_unit_colon:
#             unit = initial_unit_colon.group(1)
#             num = initial_unit_colon.group(2)
#             num1 = re.sub(r"\s*\.\s*", ".", num)
#             dist_bot = f"{num1} {unit}"
#             break
#         #Merge with next token if it's a unit
#         if j + 1 < len(results) and re.match(r"^(m|ft|Miles)$", results[j + 1].strip(),flags=re.IGNORECASE):
#             #text += results[j + 1].strip()
#             #print('text is:',results[j-1])
#             te = re.findall(r'(?<!\d\.)\b\d{1,2}\b(?!\.\d)', text)
#             # print(te)
#             if not text:
#                text = results[j-1]

#             if len(te):
#                 text = results[j-1] + text 
            

#              #print('empty')
#             text1 = text + ' ' + results[j + 1].strip()
#             dist_bot = text1
            
#             #dist_bot = text
#         #matchp = re.search(r"\b\d+\s*\.\s*\d+\s*(?:m|M|ft|FT|Ft)\b", text)
#         #matchp = re.search(r'\b\d+(?:\s*\.\s*\d+)?(?:\s*\d+)?\s*(?:m|ft)\b',text, flags=re.IGNORECASE)
#         #if matchp:
#         #    dist_bot = matchp.group()
#         #if (j != 0) and re.match(r"^(ft[:=]|Miles[:=]|Miles)$", results[j - 1].strip(),flags=re.IGNORECASE):
#         if (j != 0) and re.match(r"^((ft|Miles)|(ft[:=]|Miles[:=]|Miles))$", results[j - 1].strip(),flags=re.IGNORECASE):

#             te = re.findall(r'(?<!\d\.)\b\d{1,2}\b(?!\.\d)', text)

#             if len(te):
#                 text = text + results[j+1]
#                 text1 = results[j - 1].strip() + ' ' + text
#                 #print('text1 whole number is', text1)


#             elif not text:
#                text = results[j+1]
#                text1 = results[j - 1].strip() + ' ' + text
#                #print('text1 space is', text1)

#             elif (text == ':' or text == '='):

#                 if len(te):
#                     text = results[j+1] + results[j+2]
#                 else:
#                     text = results[j+1]
#                 #text = results[j+1]
#                 text1 = results[j - 1].strip() + ' ' + text
#                 #print('text1 : is', text1)
#             else:

#                 text1 = results[j - 1].strip() + ' ' + text

#             #text += results[j + 1].strip()
#             dist_bot = text1
#             print(dist_bot)
#         #matchp = re.search(r"\b\d+\s*\.\s*\d+\s*(?:m|M|ft|FT|Ft)\b", text)
#         #matchp = re.search(r'\b\d+(?:\s*\.\s*\d+)?(?:\s*\d+)?\s*(?:m|ft)\b',text, flags=re.IGNORECASE)
#         #if matchp:
#         #    dist_bot = matchp.group()
#         #break
#     #print(f"dist_bot is {dist_bot}")
#     # print(f"image name is: {mid_frame},  dist_bot is {dist_bot}")
#         #print(f"Results: {results} text is {text}, and distbot:{dist_bot}")



"""DocTR"""
# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor

# # Load the image
# doc = DocumentFile.from_images(r"D:\no_dist\_(DE-S,3)_(NA)_(0-2-22)_(NA)_(L).png")

# # Use OCR predictor
# model = ocr_predictor(pretrained=True)

# # Get the result
# result = model(doc)

# # Extract all text
# results = []
# for page in result.pages:
#     for block in page.blocks:
#         for line in block.lines:
#             for word in line.words:
#                 results.append(word.value)

# # Print all detected text
# print("Detected text:")
# # for text in all_text:
# #     print(text)
# print(results)
# # # Or as a single string
# # full_text = "\n".join(all_text)
# # print("\nFull text:")
# # print(full_text)


"""new set of code"""
# from doctr.models import ocr_predictor
# # Load an image
# image_path = r"D:\no_dist\(DE-S,3)_(NA)_(0-0-4)_(NA)_(L)_(0)_(0, 318, 849, 713)_253_clock 7 to 10.png"

# # Create an OCR predictor
# predictor = ocr_predictor.create_predictor()

# # Perform OCR on the image
# result = predictor(image_path)

# # Print the extracted text
# print(result)

"""third new from perpelx docTR"""
# Install: pip install python-doctr[torch]
import time
start_time = time.time()
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import re

# Load the image
image_path = r"C:\Users\Kaif Ibrahim\Downloads\(H,3)_(0-0-21)_(NA)_(2,3) (1).png" # Replace with your image path
doc = DocumentFile.from_images(image_path)

# Load the OCR model (downloads automatically on first run)
model = ocr_predictor(pretrained=True)

# Perform OCR
ocr_output = model(doc)

# Extract text into an array (list)
results = []
# for page in ocr_output.pages:
#     for block in page.blocks:
#         for line in block.lines:
#             line_text = " ".join([word.value for word in line.words])
#             results.append(line_text)
results = ocr_output.render().splitlines()
# t1 =text_output.split('\n')
# results.append(text_output)
# print(ocr_output)
# Print the array
print(f"results:{results}")

dist_bot = "NA"

for j, text in enumerate(results):
    text = text.replace('\xa0', ' ').replace('~', '').replace('O', '0').replace('o', '0').replace(",",".").replace(";",'.').replace("?",".").replace("-"," ").strip()

    together_text = re.search(r"(\b\d+\s*\.\s*\d+\s*)\s*(m|ft|Miles)\b", text, flags=re.IGNORECASE)
    if together_text:
        num = together_text.group(1)  
        unit = together_text.group(2)
        
        num1 = re.sub(r"\s*\.\s*", ".", num)
        #dist_bot = f"{num1} {unit}"
        dist_bot = f"{num1}{unit}"
        break
    
    together_text1 = re.search(r"(ft|Feet|Miles|Mile)\s*(\b\d+\s*\.\s*\d+\s*)\b", text, flags=re.IGNORECASE)
    if together_text1:
        unit = together_text1.group(1)  
        num = together_text1.group(2)
        num1 = re.sub(r"\s*\.\s*", ".", num)

        dist_bot = f"{num1}{unit}"
        break
    
    initial_unit_equal = re.search(r"(Feet|MILES|Miles|miles|Mile|ft)\s*=\s*(\b\d+\s*\.\s*\d+\s*)", text, flags=re.IGNORECASE)
    if initial_unit_equal:
        unit = initial_unit_equal.group(1)  
        num = initial_unit_equal.group(2)  
        num1 = re.sub(r"\s*\.\s*", ".", num)
        dist_bot = f"{num1}{unit}" 
        break
    
    initial_unit_colon = re.search(r"(Feet|Miles|MILES|Mile|ft)\s*:\s*(\b\d+\s*\.\s*\d+\s*)", text, flags=re.IGNORECASE)
    if initial_unit_colon:
        unit = initial_unit_colon.group(1)  
        num = initial_unit_colon.group(2)  
        num1 = re.sub(r"\s*\.\s*", ".", num)

        dist_bot = f"{num1}{unit}"  
        break

    if j + 1 < len(results) and re.match(r"^(m|ft|Miles)$", results[j + 1].strip(),flags=re.IGNORECASE):
        te = re.findall(r'(?<!\d\.)\b\d{1,2}\b(?!\.\d)', text)
        # print(te)
        if not text:
            text = results[j-1]
        if len(te):
            text = results[j-1] + text
        
        text1 = text + results[j + 1].strip()

        dist_bot = text1
    
    if (j != 0) and re.match(r"^((ft|Miles)|(ft[:=]|Miles[:=]|Miles))$", results[j - 1].strip(),flags=re.IGNORECASE):
        te = re.findall(r'(?<!\d\.)\b\d{1,2}\b(?!\.\d)', text)
        if len(te):
            text = text + results[j+1]
            text1 = results[j - 1].strip() +  text
            
        elif not text:
            text = results[j+1]
            text1 = results[j - 1].strip() + text
            
        elif (text == ':' or text == '='):
            if len(te):
                text = results[j+1] + results[j+2]
            else:
                text = results[j+1]
            
            text1 = results[j - 1].strip() + text
            
        else:
            text1 = results[j - 1].strip() +  text
    
        dist_bot = text1
    
final_text = re.search(r"(\b\d+\s*\.\s*\d+\s*)\s*(m|ft|Miles|Feet|Ft|MILES|Mile|Meter|METER|M)\b", dist_bot, flags=re.IGNORECASE)

if final_text :
    unit_final = final_text.group(2)

    if ((unit_final == 'm')| (unit_final == 'meter')|(unit_final == 'M')|(unit_final == 'Meter')|(unit_final == 'METER')|(unit_final == 'Mile')|(unit_final == 'MILES')|(unit_final == 'Miles')):
        unit_final = 'm'
    elif ((unit_final == 'ft')| (unit_final == 'feet')|(unit_final == 'Feet')|(unit_final == 'Ft')|(unit_final == 'FEET')):
        unit_final = 'ft'

    Final_distance = f"{final_text.group(1)}{unit_final}"

else:
    Final_distance = 'NA'
print("--- %s seconds ---" % (time.time() - start_time))
print(dist_bot)
# Example output: ['First line of text', 'Second line of text', 'Third line', ...]



"""padleocrv5"""

# from paddleocr import PaddleOCR
# ocr = PaddleOCR(use_angle_cls=True, lang='en')
# img_path = r"D:\no_dist\(DE-S,3)_(NA)_(0-0-4)_(NA)_(L)_(0)_(0, 318, 849, 713)_253_clock 7 to 10.png"
# result = ocr.predict(img_path)
# # if None not in result:
# #     for line in result[0]:
# #         print(line[1][0])
# for res in result:
#     if 'rec_texts' in res:
#         print("\n".join(res['rec_texts']))

"""easyocr"""
# import time
# start_time = time.time()
# import easyocr
# import re
# reader = easyocr.Reader(['en'])  # this needs to run only once to load the model into memory
# mid_frame = r"D:\no_dist\(SW,2)_(0-10-43)_(NA)_(7,8).png"
# results = reader.readtext(mid_frame, detail=0)

# print(f"results:{results}")

# dist_bot = "NA"

# for j, text in enumerate(results):
#     text = text.replace('\xa0', ' ').replace('~', '').replace('O', '0').replace('o', '0').replace(",",".").replace(";",'.').replace("?",".").replace("-"," ").strip()

#     together_text = re.search(r"(\b\d+\s*\.\s*\d+\s*)\s*(m|ft|Miles)\b", text, flags=re.IGNORECASE)
#     if together_text:
#         num = together_text.group(1)  
#         unit = together_text.group(2)
        
#         num1 = re.sub(r"\s*\.\s*", ".", num)
#         #dist_bot = f"{num1} {unit}"
#         dist_bot = f"{num1}{unit}"
#         break
    
#     together_text1 = re.search(r"(ft|Feet|Miles|Mile)\s*(\b\d+\s*\.\s*\d+\s*)\b", text, flags=re.IGNORECASE)
#     if together_text1:
#         unit = together_text1.group(1)  
#         num = together_text1.group(2)
#         num1 = re.sub(r"\s*\.\s*", ".", num)

#         dist_bot = f"{num1}{unit}"
#         break
    
#     initial_unit_equal = re.search(r"(Feet|MILES|Miles|miles|Mile|ft)\s*=\s*(\b\d+\s*\.\s*\d+\s*)", text, flags=re.IGNORECASE)
#     if initial_unit_equal:
#         unit = initial_unit_equal.group(1)  
#         num = initial_unit_equal.group(2)  
#         num1 = re.sub(r"\s*\.\s*", ".", num)
#         dist_bot = f"{num1}{unit}" 
#         break
    
#     initial_unit_colon = re.search(r"(Feet|Miles|MILES|Mile|ft)\s*:\s*(\b\d+\s*\.\s*\d+\s*)", text, flags=re.IGNORECASE)
#     if initial_unit_colon:
#         unit = initial_unit_colon.group(1)  
#         num = initial_unit_colon.group(2)  
#         num1 = re.sub(r"\s*\.\s*", ".", num)

#         dist_bot = f"{num1}{unit}"  
#         break

#     if j + 1 < len(results) and re.match(r"^(m|ft|Miles)$", results[j + 1].strip(),flags=re.IGNORECASE):
#         te = re.findall(r'(?<!\d\.)\b\d{1,2}\b(?!\.\d)', text)
#         # print(te)
#         if not text:
#             text = results[j-1]
#         if len(te):
#             text = results[j-1] + text
        
#         text1 = text + results[j + 1].strip()

#         dist_bot = text1
    
#     if (j != 0) and re.match(r"^((ft|Miles)|(ft[:=]|Miles[:=]|Miles))$", results[j - 1].strip(),flags=re.IGNORECASE):
#         te = re.findall(r'(?<!\d\.)\b\d{1,2}\b(?!\.\d)', text)
#         if len(te):
#             text = text + results[j+1]
#             text1 = results[j - 1].strip() +  text
            
#         elif not text:
#             text = results[j+1]
#             text1 = results[j - 1].strip() + text
            
#         elif (text == ':' or text == '='):
#             if len(te):
#                 text = results[j+1] + results[j+2]
#             else:
#                 text = results[j+1]
            
#             text1 = results[j - 1].strip() + text
            
#         else:
#             text1 = results[j - 1].strip() +  text
    
#         dist_bot = text1
    
# final_text = re.search(r"(\b\d+\s*\.\s*\d+\s*)\s*(m|ft|Miles|Feet|Ft|MILES|Mile|Meter|METER|M)\b", dist_bot, flags=re.IGNORECASE)

# if final_text :
#     unit_final = final_text.group(2)

#     if ((unit_final == 'm')| (unit_final == 'meter')|(unit_final == 'M')|(unit_final == 'Meter')|(unit_final == 'METER')|(unit_final == 'Mile')|(unit_final == 'MILES')|(unit_final == 'Miles')):
#         unit_final = 'm'
#     elif ((unit_final == 'ft')| (unit_final == 'feet')|(unit_final == 'Feet')|(unit_final == 'Ft')|(unit_final == 'FEET')):
#         unit_final = 'ft'

#     Final_distance = f"{final_text.group(1)}{unit_final}"

# else:
#     Final_distance = 'NA'
# print("--- %s seconds ---" % (time.time() - start_time))
# print(dist_bot)