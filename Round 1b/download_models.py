# download_models.py
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

# Define model names
EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L6-v2'
LLM_MODEL = 'google/flan-t5-base'

# Set a persistent cache directory within the Docker image
# This ensures the models are saved in a predictable location
CACHE_DIR = "/persistent_cache"
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['SENTENCE_TRANSFORMERS_HOME'] = CACHE_DIR

print(f"Downloading models to {CACHE_DIR}...")

# Download and cache the embedding model
print(f"Downloading {EMBEDDING_MODEL}...")
SentenceTransformer(EMBEDDING_MODEL, cache_folder=CACHE_DIR)

# Download and cache the reranker model
print(f"Downloading {RERANKER_MODEL}...")
CrossEncoder(RERANKER_MODEL, cache_folder=CACHE_DIR)

# Download and cache the LLM and its tokenizer
print(f"Downloading {LLM_MODEL}...")
T5Tokenizer.from_pretrained(LLM_MODEL, cache_dir=CACHE_DIR)
T5ForConditionalGeneration.from_pretrained(LLM_MODEL, cache_dir=CACHE_DIR)

print("--- All models downloaded successfully! ---")

# Note: The 'unstructured' library also downloads models (like 'yolox').
# Running the main script once during the build process, or including
# a small PDF partition here, would also cache its dependencies.
