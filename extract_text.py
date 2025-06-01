# scripts/extract_text.py

import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """
    Returns a single string with each slide labeled:
      "----- Slide 1 -----\n<slide 1 text>\n\n----- Slide 2 -----\n<slide 2 text>\n\n..."
    """
    doc = fitz.open(pdf_path)
    lines = []
    for page in doc:
        text = page.get_text().strip()
        lines.append(f"----- Slide {page.number+1} -----\n{text}\n")
    return "\n".join(lines)


if __name__ == "__main__":
    # If run directly, process all PDFs in input_decks/ and write .txt to ground_truth/
    import os
    INPUT_FOLDER = "input_decks"
    OUTPUT_FOLDER = "ground_truth"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for fname in os.listdir(INPUT_FOLDER):
        if not fname.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(INPUT_FOLDER, fname)
        plain = extract_text_from_pdf(pdf_path)
        base = os.path.splitext(fname)[0]
        out_txt = os.path.join(OUTPUT_FOLDER, f"{base}.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(plain)
        print(f"Extracted text for {fname} → {base}.txt")
