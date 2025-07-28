# In src/pdf_parser.py

import fitz  # PyMuPDF
import re


def is_header(span, body_font_size):
    """Determines if a span of text is likely a header."""
    if not span["text"].strip():
        return False

    # Heuristic 1: Font size is significantly larger than body text
    is_large_font = span["size"] > (body_font_size + 1.5)

    # Heuristic 2: Font is bold
    is_bold = "bold" in span["font"].lower()

    # Heuristic 3: Text is short and doesn't end with a period.
    is_short_and_clean = len(span["text"].strip()) < 100 and not span[
        "text"
    ].strip().endswith(".")

    # A good header satisfies at least two of these conditions
    return (is_large_font and is_short_and_clean) or (is_bold and is_short_and_clean)


def extract_sections(pdf_path):
    """Extracts sections from a PDF using improved heuristics."""
    doc = fitz.open(pdf_path)
    sections = []
    current_section = {"title": "Introduction", "content": "", "page_number": 1}

    # Get the most common font size to identify body text
    font_counts = {}
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        size = round(span["size"])
                        font_counts[size] = font_counts.get(size, 0) + 1

    body_font_size = max(font_counts, key=font_counts.get) if font_counts else 12

    # Now iterate and extract sections
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                full_line_text = "".join(
                    span["text"] for span in block["lines"][0]["spans"]
                ).strip()

                # Use the first span of the line to check for header properties
                first_span = block["lines"][0]["spans"][0]

                if is_header(first_span, body_font_size):
                    # If we found a new header, save the previous section
                    if (
                        len(current_section["content"].strip()) > 150
                    ):  # Only save meaningful sections
                        sections.append(current_section)

                    # Start a new section
                    current_section = {
                        "title": full_line_text,
                        "content": "",
                        "page_number": page_num,
                    }
                else:
                    # Append the block's text to the current section's content
                    for line in block["lines"]:
                        line_text = "".join(span["text"] for span in line["spans"])
                        current_section["content"] += line_text + "\n"

    # Add the last processed section if it's meaningful
    if len(current_section["content"].strip()) > 150:
        sections.append(current_section)

    return sections
