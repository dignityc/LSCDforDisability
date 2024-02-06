import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os

# Function to extract text from a single page
os.environ['TESSDATA_PREFIX'] = '/scratch/users/k2369089/LSCDforDisability'
custom_config = r'--psm 3'

# Rest of your script...
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# Convert PDF to images
images = convert_from_path('PDF/Sample.pdf')

# Extract text from each image
for i, image in enumerate(images):
    text = extract_text_from_image(image)
    with open(f'datasets/extracted_text_page_{i+1}.txt', 'w') as file:
        file.write(text)

