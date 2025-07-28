import json
import os
from datetime import datetime
from pdf_parser import extract_sections
from analysis import get_initial_ranking, summarize_text_abstractive
from reranker import mmr_rerank


def process_documents(input_path):
    """
    Processes a collection of documents based on a persona and job-to-be-done,
    extracting and summarizing the most relevant, diverse sections.
    """
    # 1. Load Input
    with open(input_path, "r") as f:
        input_data = json.load(f)

    doc_infos = input_data["documents"]
    persona = input_data["persona"]["role"]
    job_to_be_done = input_data["job_to_be_done"]["task"]

    # Assume PDFs are in the same directory as the input JSON
    base_path = os.path.dirname(input_path)

    # 2. Parse all documents and collect sections
    print("Step 1/5: Parsing all PDF documents...")
    all_sections = []
    for doc_info in doc_infos:
        pdf_path = os.path.join(base_path, doc_info["filename"])
        if os.path.exists(pdf_path):
            sections = extract_sections(pdf_path)
            for section in sections:
                section["document"] = doc_info["filename"]
            all_sections.extend(sections)
        else:
            print(f"Warning: Document not found at {pdf_path}")

    if not all_sections:
        print("Error: No sections were extracted from any documents. Exiting.")
        return

    # 3. Get Initial Ranking based on semantic relevance
    print("Step 2/5: Calculating initial relevance scores...")
    query_embedding, section_embeddings, all_sections = get_initial_ranking(
        all_sections, persona, job_to_be_done
    )

    # 4. Rerank for Diversity using MMR to get the top sections
    print("Step 3/5: Reranking results for diversity with MMR...")
    top_sections = mmr_rerank(
        query_embedding, section_embeddings, all_sections, lambda_val=0.6, top_n=5
    )

    # 5. Prepare the output dictionary
    print("Step 4/5: Generating final JSON output...")
    output_data = {
        "metadata": {
            "input_documents": [d["filename"] for d in doc_infos],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.utcnow().isoformat()
            + "Z",  # Adding Z for UTC
        }
    }

    # Populate "extracted_sections" from the reranked list
    output_data["extracted_sections"] = [
        {
            "document": s["document"],
            "section_title": s["title"],
            "importance_rank": s["importance_rank"],
            "page_number": s["page_number"],
        }
        for s in top_sections
    ]

    # 6. Generate Abstractive Summaries for the top sections
    print("Step 5/5: Creating abstractive summaries...")
    subsection_analysis_list = []
    for s in top_sections:
        print(f"  -> Summarizing section '{s['title']}' from {s['document']}...")
        refined_text = summarize_text_abstractive(s["content"])
        subsection_analysis_list.append(
            {
                "document": s["document"],
                "refined_text": refined_text,
                "page_number": s["page_number"],
            }
        )

    output_data["subsection_analysis"] = subsection_analysis_list

    # 7. Save the output
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "challenge1b_output.json")

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"\nProcessing complete. Output saved to {output_path}")


if __name__ == "__main__":
    # The script expects the input JSON path to be in the 'data' folder
    # relative to the project root.
    input_json_path = "data/challenge1b_input.json"
    if not os.path.exists(input_json_path):
        print(f"Error: Input file not found at '{input_json_path}'")
    else:
        process_documents(input_json_path)
