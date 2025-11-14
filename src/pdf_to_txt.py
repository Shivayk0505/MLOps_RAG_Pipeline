import os
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from src.logger import get_logger

log = get_logger()

def extract_text_from_pdf(pdf_path, txt_folder="./data/docs/"):
    """
    Extract text for just ONE pdf
    """
    os.makedirs(txt_folder, exist_ok=True)

    filename = os.path.basename(pdf_path)
    txt_filename = os.path.splitext(filename)[0] + ".txt"
    txt_path = os.path.join(txt_folder, txt_filename)

    # skip if already exists
    if os.path.exists(txt_path):
        log.info(f"Skip extraction — {txt_filename} already exists")
        return txt_path

    log.info(f"Starting extraction for: {filename}")
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text += page_text + "\n"
            else:
                images = convert_from_path(pdf_path, first_page=i+1, last_page=i+1)
                for image in images:
                    ocr_text = pytesseract.image_to_string(image, lang='eng', config='--psm 3')
                    text += ocr_text + "\n"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    log.info(f"Extracted and saved: {txt_filename}")
    return txt_path
