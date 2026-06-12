"""
parse_book.py — Two-pass PDF extractor for Coffee Can Investing.
Uses PyMuPDF (fitz) — installed as pymupdf. pdfplumber not available.
Outputs: docs/complete_book_parsed.txt with --- PAGE X --- markers.
"""
import fitz
import os

PDF_PATH = r"s:\Stock Scan (Build)\Other Resources\Coffee_Can_Investing_The_Low_Risk_Road_to_Stupendous_Wealth_-_Saurabh_Mukherjea.pdf"
OUT_PATH = r"s:\Stock Scan (Build)\docs\complete_book_parsed.txt"

doc = fitz.open(PDF_PATH)
total = doc.page_count
print(f"Opened: {total} pages")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for i in range(total):
        page = doc[i]
        text = page.get_text("text")          # pass 1: native text layer
        if not text.strip():
            # pass 2: try block extraction for tables / rotated text
            blocks = page.get_text("blocks")
            text = "\n".join(b[4] for b in blocks if isinstance(b[4], str))
        f.write(f"\n--- PAGE {i + 1} ---\n")
        f.write(text if text.strip() else "[IMAGE/SCAN — no extractable text]\n")

doc.close()
size_kb = os.path.getsize(OUT_PATH) // 1024
print(f"Written: {OUT_PATH}  ({size_kb} KB)")
