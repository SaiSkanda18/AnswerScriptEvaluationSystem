import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np

def extract_text_from_image(image_path, split_ratio=0.5):
    # Load the image from the specified path
    image = cv2.imread(image_path)

    # Split the image into typed and handwritten sections
    height = image.shape[0]
    split_point = int(height * split_ratio)
    typed_image = image[:split_point, :]
    handwritten_image = image[split_point:, :]

    # Preprocess the images (convert to grayscale, apply thresholding)
    typed_gray = cv2.cvtColor(typed_image, cv2.COLOR_BGR2GRAY)
    handwritten_gray = cv2.cvtColor(handwritten_image, cv2.COLOR_BGR2GRAY)
    _, typed_thresh = cv2.threshold(typed_gray, 150, 255, cv2.THRESH_BINARY_INV)
    _, handwritten_thresh = cv2.threshold(handwritten_gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Use Tesseract to extract text from the preprocessed images
    typed_text = pytesseract.image_to_string(typed_thresh, config='--psm 6')
    handwritten_text = pytesseract.image_to_string(handwritten_thresh, config='--psm 6')

    return typed_text, handwritten_text

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Resize the image for better OCR performance
    resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    return resized_image

def save_extracted_text_to_file(text, output_file):
    with open(output_file, 'w') as file:
        file.write(text)

def extract_text_from_pdf(pdf_file):
    images = convert_from_path(pdf_file)
    text = ""
    for image in images:
        # Convert the image to a numpy array
        image_np = np.array(image)
        # Use Tesseract to extract text from the image
        text += pytesseract.image_to_string(image_np, config='--psm 6')
    return text