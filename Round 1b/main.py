# --- Complete Code for Solution Bravo ---

import os
import faiss
import numpy as np
from unstructured.partition.pdf import partition_pdf
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import T5Tokenizer, T5ForConditionalGeneration
import warnings

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

class RAGSystemBravo:
    """
    An advanced RAG system implementing a retrieve-and-rerank pipeline.
    """
    def __init__(self, pdf_directory: str,
                 embedding_model: str = 'sentence-transformers/all-mpnet-base-v2',
                 reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L6-v2',
                 llm_model: str = 'google/flan-t5-base'):

        self.pdf_directory = pdf_directory
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.llm_model_name = llm_model

        print("Initializing models...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.reranker_model = CrossEncoder(self.reranker_model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(self.llm_model_name)
        self.llm = T5ForConditionalGeneration.from_pretrained(self.llm_model_name)

        self.index = None
        self.chunk_texts = []
        self.chunk_elements = []

    def _ingest_documents(self):
        """
        Private method to parse PDFs and enrich with metadata.
        """
        print(f"Ingesting documents from: {self.pdf_directory}")
        all_elements = []
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith(".pdf")]
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in directory: {self.pdf_directory}")

        for filename in pdf_files:
            filepath = os.path.join(self.pdf_directory, filename)
            try:
                elements = partition_pdf(
                    filename=filepath,
                    strategy="hi_res",
                    infer_table_structure=True,
                    model_name="yolox"
                )
                for element in elements:
                    element.metadata.source = filename
                    all_elements.append(element)
                print(f"Successfully processed {filename}")
            except Exception as e:
                print(f"Could not process {filename} with 'hi_res' strategy. Error: {e}. Falling back to 'fast' strategy.")
                elements = partition_pdf(filename=filepath, strategy="fast")
                for element in elements:
                    element.metadata.source = filename
                    all_elements.append(element)

        self.chunk_elements = all_elements
        self.chunk_texts = [str(el) for el in self.chunk_elements]
        print(f"Ingestion complete. Found {len(self.chunk_texts)} chunks.")

    def _create_embeddings_and_index(self):
        """
        Private method to create embeddings and build the FAISS index.
        """
        if not self.chunk_texts:
            raise ValueError("No chunks to embed. Run ingestion first.")

        print("Creating embeddings for document chunks...")
        embeddings = self.embedding_model.encode(self.chunk_texts, show_progress_bar=True)

        print("Building FAISS vector index...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings, dtype=np.float32))
        print("Index built successfully.")

    def setup(self):
        """
        Runs the full ingestion and indexing pipeline.
        """
        self._ingest_documents()
        self._create_embeddings_and_index()
        print("--- RAG System Bravo is ready. ---")

    def query(self, question: str, retrieve_k: int = 50, rerank_n: int = 5):
        """
        Performs a query against the RAG system using the retrieve-and-rerank workflow.
        """
        if self.index is None:
            raise RuntimeError("System not set up. Please run the.setup() method first.")

        print(f"\n--- Querying for: '{question}' ---")

        # 1. Retrieve initial candidates
        print(f"1. Retrieving top {retrieve_k} candidates...")
        query_embedding = self.embedding_model.encode([question])
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), retrieve_k)
        retrieved_chunks = [self.chunk_texts[i] for i in indices[0]]

        # 2. Rerank the candidates
        print(f"2. Reranking candidates to find top {rerank_n}...")
        query_chunk_pairs = [[question, chunk] for chunk in retrieved_chunks]
        scores = self.reranker_model.predict(query_chunk_pairs, show_progress_bar=False)

        chunk_score_pairs = list(zip(retrieved_chunks, scores))
        chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)

        reranked_chunks = [chunk for chunk, score in chunk_score_pairs[:rerank_n]]

        print(f"Identified top {len(reranked_chunks)} most relevant chunks after reranking.")

        # 3. Generate answer
        print("3. Synthesizing answer...")
        prompt_template = """
        You are an expert assistant. Answer the following question based ONLY on the provided context.
        If the context does not contain enough information to answer the question, state that the information is not available in the provided documents.
        Be concise and accurate.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """

        context_str = "\n\n---\n\n".join(reranked_chunks)
        prompt = prompt_template.format(context=context_str, question=question)

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
        outputs = self.llm.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("4. Final Answer:")
        print(answer)
        return answer

# --- Example Usage ---
if __name__ == '__main__':
    corpus_directory = "Pdf2"
    if not os.path.exists(corpus_directory) or not os.listdir(corpus_directory):
        print(f"Error: The directory '{corpus_directory}' does not exist or is empty.")
        print("Please create it and place your PDF files inside.")
    else:
        rag_system = RAGSystemBravo(pdf_directory=corpus_directory)
        rag_system.setup()
        
        # You can now interact with the system programmatically
        # or by calling the query method from another script.
        # For example:
        # rag_system.query("Your question here")
