import fitz  # PyMuPDF
import json
import os
import re
from collections import Counter

# --- Path Handling ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
except NameError:
    script_dir = os.getcwd()
    project_root = script_dir

INPUT_DIR = os.path.join(project_root, "input")
OUTPUT_DIR = os.path.join(project_root, "output")


def profile_document(doc):
    """Analyzes document to find body style and potential heading styles."""
    style_catalog = Counter()
    for page in doc:
        blocks = page.get_text(
            "dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_WHITESPACE
        )["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        is_bold = "bold" in span["font"].lower()
                        style_key = (round(span["size"]), is_bold)
                        style_catalog[style_key] += len(text)

    if not style_catalog:
        return {"body_style": (10, False), "heading_styles": set()}

    body_style = style_catalog.most_common(1)[0][0]
    # A heading style is any style that is larger or bolder than the body style.
    heading_styles = {
        s for s in style_catalog if s[0] > body_style[0] or (s[1] and not body_style[1])
    }

    return {"body_style": body_style, "heading_styles": heading_styles}


def score_block(block, doc_profile):
    """Scores a text block on its likelihood of being a heading."""
    block_text = " ".join(
        "".join(s.get("text", "") for s in l.get("spans", [])).strip()
        for l in block.get("lines", [])
    ).strip()
    if not block_text or not block.get("lines") or not block["lines"][0].get("spans"):
        return 0, None

    # --- Feature Extraction ---
    first_line = block["lines"][0]
    span = first_line["spans"][0]
    font_size = round(span["size"])
    is_bold = "bold" in span["font"].lower()
    block_style = (font_size, is_bold)
    word_count = len(block_text.split())

    # --- Scoring ---
    score = 0
    # Style Score: Must be a pre-identified heading style
    if block_style in doc_profile["heading_styles"]:
        score += 10
        score += (font_size - doc_profile["body_style"][0]) * 2
        if is_bold:
            score += 5
    else:
        return 0, None  # If not a heading style, score is 0

    # Structure & Content Score
    if len(block.get("lines", [])) < 3:
        score += 5  # Bonus for being short
    if re.match(r"^\d+(\.\d+)*\s", block_text):
        score += 15  # High bonus for numbered headings
    if block_text.isupper() and word_count > 1:
        score += 5

    # Penalty for sentence-like text
    if word_count > 25 or (
        block_text.endswith(".") and not re.match(r"^\d+\.", block_text)
    ):
        score -= 15

    candidate = {
        "text": block_text,
        "score": score,
        "style": block_style,
        "page": block.get("page_num", 0),
    }
    return score, candidate


def process_pdf(pdf_path):
    """The main processing pipeline for high-accuracy heading extraction."""
    doc = fitz.open(pdf_path)
    if not len(doc):
        return {"title": "", "outline": []}

    # Pass 1: Profile the document's styles
    doc_profile = profile_document(doc)

    # Pass 2: Score all blocks to find candidates
    heading_candidates = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text(
            "dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_WHITESPACE
        )["blocks"]
        for block in blocks:
            block["page_num"] = page_num
            score, candidate = score_block(block, doc_profile)
            if candidate and score > 10:
                heading_candidates.append(candidate)

    if not heading_candidates:
        return {"title": "", "outline": []}

    # Pass 3: Identify Title and filter candidates
    heading_candidates.sort(key=lambda x: x["score"], reverse=True)
    title = ""
    page_one_candidates = [c for c in heading_candidates if c["page"] == 0]
    if page_one_candidates:
        title = page_one_candidates[0]["text"]

    outline_candidates = [
        c for c in heading_candidates if c["text"].lower() != title.lower()
    ]

    # Pass 4: Assign Levels (H1-H3 Capped)
    if not outline_candidates:
        return {"title": title, "outline": []}

    # Group candidates by their style
    style_clusters = {}
    for c in outline_candidates:
        if c["style"] not in style_clusters:
            style_clusters[c["style"]] = []
        style_clusters[c["style"]].append(c)

    # Rank clusters by prominence (font size is the primary factor)
    # The style key is (size, is_bold)
    sorted_styles = sorted(style_clusters.keys(), key=lambda s: s[0], reverse=True)

    # Assign H1, H2, and cap everything else at H3
    level_map = {}
    for i, style in enumerate(sorted_styles):
        if i == 0:
            level_map[style] = "H1"
        elif i == 1:
            level_map[style] = "H2"
        else:
            level_map[style] = "H3"

    outline = []
    for cand in outline_candidates:
        if cand["style"] in level_map:
            outline.append(
                {
                    "level": level_map[cand["style"]],
                    "text": re.sub(r"^(\d+(\.\d+)*)\s*", "", cand["text"]).strip(),
                    "page": cand["page"],
                }
            )

    # Pass 5: Final Cleanup and Sorting
    final_outline = []
    processed_texts = set()
    outline.sort(key=lambda x: (x["page"], x["text"]))
    for item in outline:
        if item["text"].lower() not in processed_texts:
            final_outline.append(item)
            processed_texts.add(item["text"].lower())

    doc.close()
    return {"title": title, "outline": final_outline}


def main():
    """Main function to run the batch processing."""
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(INPUT_DIR, filename)
            print(f"🚀 Processing {filename}...")
            try:
                structured_data = process_pdf(pdf_path)
                json_filename = os.path.splitext(filename)[0] + ".json"
                output_path = os.path.join(OUTPUT_DIR, json_filename)

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(structured_data, f, indent=4, ensure_ascii=False)

                print(f"✅ Successfully created {output_path}")
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")
                import traceback

                traceback.print_exc()


if __name__ == "__main__":
    main()
