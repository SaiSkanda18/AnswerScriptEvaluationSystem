import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from text_extraction.extract_text import extract_text_from_pdf

def extract_reference_text(pdf_path="Answer key.pdf"):
    """Extract reference text from the answer-key PDF using existing OCR pipeline."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Reference PDF not found: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    return text
