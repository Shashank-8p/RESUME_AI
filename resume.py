"""
Step 1: PDF Extractor & Text Sanitizer
---------------------------------------
Extracts raw text from a PDF using PyMuPDF and cleans it
using RegEx into a single, sanitized string.
"""

import fitz  # PyMuPDF
import re

# --- CONFIGURATION ---
PDF_PATH = "sanjeet-1.pdf"


def extract_text_from_pdf(path: str) -> str:
    """Opens a PDF and extracts raw text from every page."""
    raw_pages = []
    with fitz.open(PDF_PATH) as doc:
        for page in doc:
            raw_pages.append(page.get_text("text"))
    return "\n".join(raw_pages)


def clean_text(raw: str) -> str:
    """
    Applies a series of RegEx cleaning steps:
    1. Strip zero-width & invisible Unicode characters
    2. Normalize unicode whitespace variants to a standard space
    3. Remove non-printable / control characters (except newlines/tabs)
    4. Collapse multiple blank lines into a single blank line
    5. Strip leading/trailing whitespace from each line
    6. Fix erratic mid-word line breaks (e.g., "Soft\nware" -> "Software")
    7. Final strip of the whole string
    """
    # 1. Remove zero-width characters (U+200B, U+FEFF, U+00AD, etc.)
    text = re.sub(r'[\u200b\u200c\u200d\u00ad\ufeff\u2060]', '', raw)

    # 2. Normalize unicode whitespace variants to standard space
    text = re.sub(r'[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]', ' ', text)

    # 3. Remove non-printable control characters, keeping \n and \t
    text = re.sub(r'[^\x09\x0a\x0d\x20-\x7e\u00c0-\u024f]', '', text)

    # 4. Strip leading/trailing whitespace from every line
    text = "\n".join(line.strip() for line in text.splitlines())

    # 5. Fix erratic line breaks inside words:
    #    If a line ends with a lowercase letter and the next starts with one,
    #    it's likely a broken word — join with a space.
    text = re.sub(r'([a-z,;])\n([a-z])', r'\1 \2', text)

    # 6. Collapse 3+ consecutive newlines into exactly 2 (one blank line)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 7. Final strip
    return text.strip()


if __name__ == "__main__":
    print(f"[*] Extracting text from: {PDF_PATH}\n")

    raw_text = extract_text_from_pdf(PDF_PATH)
    clean = clean_text(raw_text)

    print("=" * 60)
    print("CLEANED RESUME TEXT:")
    print("=" * 60)
    print(clean)
    print("=" * 60)
    print(f"\n[✓] Done. Total characters after cleaning: {len(clean)}")
