import os
import pdfplumber
from pdf2image import convert_from_path
import pytesseract

def extract_text_from_pdfs(pdf_folder="data/pdfs", txt_folder="data/docs"):
    """
    Extract text from PDFs using pdfplumber.
    If page has no text, fallback to OCR (Tesseract).
    """
    os.makedirs(txt_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in {pdf_folder}")
        return

    for filename in pdf_files:
        pdf_path = os.path.join(pdf_folder, filename)
        text = ""

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text += page_text + "\n"
                else:
                    # OCR fallback
                    images = convert_from_path(pdf_path, first_page=i+1, last_page=i+1)
                    for image in images:
                        ocr_text = pytesseract.image_to_string(image, lang='eng', config='--psm 3')
                        text += ocr_text + "\n"

        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(txt_folder, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"✅ Extracted {filename} → {txt_filename}")

if __name__ == "__main__":
    extract_text_from_pdfs()
