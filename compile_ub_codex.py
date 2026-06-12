"""
compile_ub_codex.py — Layout-aware PDF extractor for The Unusual Billionaires.
Uses PyMuPDF (fitz). Outputs docs/unusual_billionaires_master_compiled.txt
with --- PAGE X --- markers. Marks scan/image pages as [SCAN_PAGE_INDEX_X].
"""
import fitz
import os

PDF_PATH = r"s:\Stock Scan (Build)\Other Resources\The_Unusual_Billionaires_-_Saurabh_Mukherjea.pdf"
OUT_PATH = r"s:\Stock Scan (Build)\docs\unusual_billionaires_master_compiled.txt"

doc = fitz.open(PDF_PATH)
total = doc.page_count
print(f"Opened: {total} pages")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for i in range(total):
        page = doc[i]
        text = page.get_text("text")
        if not text or len(text.strip()) < 100:
            # Fallback: block extraction
            blocks = page.get_text("blocks")
            text = "\n".join(b[4] for b in blocks if isinstance(b[4], str))
        f.write(f"\n--- PAGE {i + 1} ---\n")
        if text and len(text.strip()) >= 30:
            f.write(text)
        else:
            f.write(f"[SCAN_PAGE_INDEX_{i + 1}]\n")

doc.close()
size_kb = os.path.getsize(OUT_PATH) // 1024
print(f"Written: {OUT_PATH}  ({size_kb} KB, {total} pages)")
