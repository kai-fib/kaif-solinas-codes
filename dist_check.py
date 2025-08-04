prev = '0000010.03M'
import easyocr
import re
reader = easyocr.Reader(['en'])  # this needs to run only once to load the model into memory
mid_frame = r"C:\Users\Kaif Ibrahim\Pictures\vlcsnap-2025-08-04-14h22m14s775.png"
results = reader.readtext(mid_frame, detail=0)
print(results)
dist_bot = "0.0 m"
for j, text in enumerate(results):
    text = text.replace('\xa0', ' ').replace('~', '').replace('O', '0').replace('o', '0').strip()
    together_text = re.search(r"(\d+\.?\d*)(m|M|ft|FT|Ft)\b", prev)
    if together_text:
        num = together_text.group(1)
        unit = together_text.group(2)
        dist_bot = f"{num} {unit}"
        break
    # Merge with next token if it's a unit
    if j + 1 < len(results) and re.match(r"^(m|M|ft|FT|Ft)$", results[j + 1].strip()):
        #text += results[j + 1].strip()
        text = text + ' ' + results[j + 1].strip()
    matchp = re.search(r"\b\d+\.?\d*\s*(?:m|M|ft|FT|Ft)\b", prev)
    if matchp:
        num, unit = re.match(r"(\d+\.?\d*)\s*(m|M|ft|FT|Ft)", matchp.group()).groups()
        dist_bot = (f"{num} {unit}")
        #dist_bot = matchp.group()
        break
print(dist_bot)


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