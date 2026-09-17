import re
from pypdf import PdfReader


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = clean_text(text)

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages