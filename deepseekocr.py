"""Deepseek_Ocr"""
# # from deepseek_ocr_encoder import DeepSeekOCREncoder

# # # One-line initialization - automatically handles device, dtype, and model loading
# # encoder = DeepSeekOCREncoder.from_pretrained("deepseek-ai/DeepSeek-OCR")

# # # Encode an image
# # vision_tokens = encoder(r"D:\no_dist\(DE-S,3)_(NA)_(0-0-4)_(NA)_(L)_(0)_(0, 318, 849, 713)_253_clock 7 to 10.png")
# # # Returns: torch.Tensor of shape [1, N, 1024] where N=256 for 1024x1024 input

# from transformers import AutoModel, AutoTokenizer
# import torch
# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# model_name = 'deepseek-ai/DeepSeek-OCR'

# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
# model = model.eval().cuda().to(torch.bfloat16)

# # Prompt to find all text in the image
# prompt = "<image>\nFind all text present in the image."
# image_file = r"D:\no_dist\(DE-S,3)_(NA)_(0-0-4)_(NA)_(L)_(0)_(0, 318, 849, 713)_253_clock 7 to 10.png"

# # Run inference and get text result
# res = model.infer(
#     tokenizer, 
#     prompt=prompt, 
#     image_file=image_file, 
#     base_size=1024, 
#     image_size=640, 
#     crop_mode=True,
#     save_results=False,  # Don't save files, just return text
#     test_compress=True
# )

# # Print the extracted text
# print("All text found in the image:")
# print("=" * 50)
# print(res)
# print("=" * 50)
"""TRocr"""
"""this is not working didnt check properly"""
# # Install: pip install transformers sentencepiece

# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from PIL import Image

# # Load the image
# image_path = r"D:\no_dist\(DE-S,3)_(NA)_(0-0-4)_(NA)_(L)_(0)_(0, 318, 849, 713)_253_clock 7 to 10.png" # Replace with your image path
# image = Image.open(image_path).convert("RGB")

# # Load processor and model (downloads automatically on first run)
# processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')
# model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed')

# # Process the image
# pixel_values = processor(image, return_tensors="pt").pixel_values

# # Generate text
# generated_ids = model.generate(pixel_values)

# # Decode to get the result
# result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# # Print result
# print(result)

"""New code for trocr"""
# https://github.com/rsommerfeld/trocr

from PIL import Image
from trocr.src.main import TrocrPredictor

# load images
image_names = ["data/img1.png", "data/img2.png"]
images = [Image.open(img_name) for img_name in image_names]

# directly predict on Pillow Images or on file names
model = TrocrPredictor()
predictions = model.predict_images(images)
predictions = model.predict_for_file_names(image_names)

# print results
for i, file_name in enumerate(image_names):
    print(f'Prediction for {file_name}: {predictions[i]}')