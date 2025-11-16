# utils/pdf_reader.py

import fitz  # PyMuPDF

def read_pdf(path: str) -> str:
    """
    將 PDF 每頁合併成一段文字
    """
    doc = fitz.open(path)
    texts = []

    for page in doc:
        text = page.get_text("text")
        texts.append(text)

    doc.close()
    return "\n".join(texts)

