# src/analysis.py
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
from transformers import T5Tokenizer, T5ForConditionalGeneration  # Import T5

# --- MODEL LOADING ---
# Load sentence model (for ranking and keywords)
sentence_model = SentenceTransformer("./models/all-MiniLM-L6-v2")
kw_model = KeyBERT(model=sentence_model)

# Load summarization model and tokenizer from local files
tokenizer = T5Tokenizer.from_pretrained("./models/t5-small-tokenizer")
summarization_model = T5ForConditionalGeneration.from_pretrained(
    "./models/t5-small-model"
)


def get_initial_ranking(sections, persona, job_to_be_done):
    """
    Generates an initial ranking of sections based on semantic similarity.
    The final ranking will be done by the MMR reranker.
    """
    context_doc = f"{persona}. {job_to_be_done}"
    keywords = kw_model.extract_keywords(
        context_doc,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        use_maxsum=True,
        nr_candidates=20,
        top_n=5,
    )
    just_keywords = " ".join([kw[0] for kw in keywords])
    query = f"{context_doc} Key concepts are: {just_keywords}"

    query_embedding = sentence_model.encode(query, convert_to_tensor=True)
    section_texts = [f"{s['title']}. {s['content']}" for s in sections]
    section_embeddings = sentence_model.encode(section_texts, convert_to_tensor=True)

    # Return everything needed for the reranker
    return query_embedding, section_embeddings, sections


def summarize_text_abstractive(text, max_length=150, min_length=40):
    """
    Generates a concise, abstractive summary of the given text using T5.
    """
    # Prepare the text for T5 by adding the "summarize: " prefix
    input_text = f"summarize: {text}"

    input_ids = tokenizer.encode(
        input_text, return_tensors="pt", max_length=1024, truncation=True
    )

    summary_ids = summarization_model.generate(
        input_ids,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
