# import easyocr
# import re
# reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
# img = r"C:\Users\Kaif Ibrahim\Desktop\solinas_downloads\check_3.png"
# results = reader.readtext(img, detail = 0)
# for text in results:
#     matchp = re.search(pattern, text.replace('\xa0', ' ').strip())
#     if matchp is None:
#         dist_bot = "0.0 m"
#     else:
#         dist_bot =  matchp.group()
# print(results)





# import easyocr
# import re

# reader = easyocr.Reader(['en'])  # Load model
# img = r"C:\Users\Kaif Ibrahim\Desktop\solinas_downloads\sample_1.png"
# pattern = r"\d+\.\d+\s*(?:[mM]|[Ff][Tt])"

# results = reader.readtext(img, detail=0)

# print("OCR Results:")
# for text in results:
#     print(f"- OCR Text: '{text}'")  # Print raw text
#     matchp = re.search(pattern, text.replace('\xa0', ' ').strip())
#     if matchp:
#         print(f"  ✅ Match: {matchp.group()}")
#     else:
#         print("  ❌ No match")

# # Optional: return the first valid match (instead of overwriting dist_bot)
# dist_bot = "0.0 m"
# for text in results:
#     matchp = re.search(pattern, text.replace('\xa0', ' ').strip())
#     if matchp:
#         dist_bot = matchp.group()
#         break

# print("Final extracted distance:", dist_bot)

# import easyocr
# import re

# reader = easyocr.Reader(['en'])  # Load the OCR model
# img = r"C:\Users\Kaif Ibrahim\Desktop\solinas_downloads\check_2.png"

# # Define both patterns
# pattern1 = r"\d+\.\d+\s*(?:[mM]|[Ff][Tt])"        # e.g., 003.0 m or 12.5 FT
# pattern2 = r"[O0]{2,3}\.0\s*[mM]"                # e.g., OO3.0 m (OCR mistake)

# results = reader.readtext(img, detail=0)

# dist_bot = "0.0 m"  # default if no match found

# for text in results:
#     cleaned_text = text.replace('\xa0', ' ').strip()

#     match1 = re.search(pattern1, cleaned_text)
#     match2 = re.search(pattern2, cleaned_text)

#     if match1:
#         dist_bot = match1.group()
#         break
#     elif match2:
#         dist_bot = match2.group()
#         break

# print("Final extracted distance:", dist_bot)


# import easyocr
# import re

# reader = easyocr.Reader(['en'])  # this needs to run only once to load the model into memory
# img = r"C:\Users\Kaif Ibrahim\Desktop\solinas_downloads\check_3.png"

# # Robust regex to match distance with unit like '001.1 m', '003.0m', '000.11M', '1.2 ft', etc.
# pattern = r"\b\d{1,4}\.\d{1,2}\s*(?:m|M|ft|FT|Ft)\b"

# results = reader.readtext(img, detail=0)

# for text in results:
#     text = text.replace('\xa0', ' ').replace('~', '').replace('O', '0').replace('o', '0').strip()
#     matchp = re.search(pattern, text)
#     if matchp is None:
#         dist_bot = "0.0 m"
#     else:
#         dist_bot = matchp.group()
#         break

# print(results)
# print("Distance detected:", dist_bot)


import easyocr
import re
# reader = easyocr.Reader(['en'])  # this needs to run only once to load the model into memory
# mid_frame = r"C:\Users\Kaif Ibrahim\Desktop\solinas_downloads\sample_1.png"

# results = reader.readtext(mid_frame, detail=0)
dist_bot = "0.0 m"
results = ['05/28/2025 12.37.37', 'BrokenB,.4L', 'OOOOO.11M','mumbai']

for j, text in enumerate(results):
    text = text.replace('\xa0', ' ').replace('~', '').replace('O', '0').replace('o', '0').strip()
    
    # Merge with next token if it's a unit
    if j + 1 < len(results) and re.match(r"^(m|M|ft|FT|Ft)$", results[j + 1].strip()):
        text += results[j + 1].strip()
    
    matchp = re.search(r"\b\d{1,5}\.\d{1,2}\s*(?:m|M|ft|FT|Ft)\b", text)
    if matchp:
        dist_bot = matchp.group()
        break
print(results)
print(dist_bot)

# results = ['SOLINAS', '11/06/2025 06:15.33', '003.1', 'm', 'ID 123456789', 'Pre-Con, GIS 19.863924,75.316400', '200 mm AC Water pipe', 'Northeast', 'Bansilal NAgar, Raj Nagar', 'Pipe 19.863924,75.316400', 'Oo', 'Se9; 19.863924,75.316400', '00']

# print(results[1].strip())