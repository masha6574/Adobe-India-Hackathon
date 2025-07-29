Of course. Here is the approach_explanation.md file for your Adobe Hackathon submission, based on the provided code.

Solution Bravo: Advanced RAG with Retrieve-and-Rerank
This document outlines the technical approach for Solution Bravo, a sophisticated Retrieval-Augmented Generation (RAG) system designed for high-accuracy question-answering over a corpus of PDF documents. Our core strategy is a two-stage retrieve-and-rerank pipeline that ensures the Large Language Model (LLM) receives the most relevant and precise context possible.

## 1. Core Philosophy 
Standard RAG pipelines retrieve documents based on semantic similarity and directly feed them to an LLM. This can sometimes result in suboptimal context, as the initial retrieval (while fast) may not perfectly capture the nuances of the query.

Our approach enhances this process by introducing a reranking step. We first retrieve a broad set of potentially relevant document chunks and then use a more powerful, fine-grained model to re-score and select the absolute best candidates before generation. This significantly improves the relevance of the context, leading to more accurate and reliable answers.

The entire workflow can be summarized in four stages:

Ingestion: Parse and chunk PDF documents intelligently.

Retrieval: Use a fast bi-encoder and a vector index to fetch an initial set of relevant chunks.

Reranking: Employ a more powerful cross-encoder to refine and re-order the retrieved chunks for maximum relevance.

Generation: Synthesize a final, coherent answer using an instruction-tuned LLM, grounded in the high-quality, reranked context.

## 2. System Architecture & Workflow ⚙️
Our system is implemented in the RAGSystemBravo class, which encapsulates the entire pipeline.

### Stage 1: Intelligent Document Ingestion
The foundation of any RAG system is the quality of its data. We use the unstructured.io library to process PDFs.

Strategy: We primarily use the "hi_res" partitioning strategy with the "yolox" model. This allows for a deep analysis of the PDF structure, accurately extracting not just text but also understanding layouts and parsing tables.

Robustness: We've built in a fallback mechanism. If the hi_res strategy fails on a complex or malformed PDF, the system automatically switches to the "fast" strategy to ensure all documents are processed.

Metadata: Each extracted chunk is enriched with metadata, specifically the source filename, which is crucial for traceability.

### Stage 2: Semantic Retrieval (The "Retrieve" Step)
This first-pass retrieval is designed for speed and broad semantic matching.

Embedding Model: We use sentence-transformers/all-mpnet-base-v2, a powerful bi-encoder model that generates high-quality vector embeddings for text chunks. It maps each chunk to a dense vector in a 768-dimensional space.

Vector Store: The generated embeddings are stored in a faiss.IndexFlatL2 index. FAISS (Facebook AI Similarity Search) is a library optimized for efficient similarity search. IndexFlatL2 performs an exhaustive, exact search using L2 (Euclidean) distance, guaranteeing that we find the true nearest neighbors for a given query vector.

Process: When a query is received, it's encoded into a vector, and this index is searched to retrieve the top k=50 candidate chunks.

### Stage 3: Relevance Refinement (The "Rerank" Step)
This is the key step that elevates our solution's accuracy. While the bi-encoder is great for finding semantically similar chunks, it doesn't always capture contextual relevance perfectly.

Reranker Model: We use a cross-encoder/ms-marco-MiniLM-L6-v2 model. Unlike a bi-encoder, a cross-encoder takes both the query and a document chunk as a single input and outputs a relevance score. This allows it to pay closer attention to the interactions between the query and chunk terms.

Process: The 50 chunks retrieved in the previous step are individually paired with the user's query. The cross-encoder scores each pair. We then sort the chunks by this new, more accurate relevance score and select the top n=5 chunks.

### Stage 4: Context-Aware Generation (The "Generate" Step)
With the most relevant context now identified, the final step is to generate a human-readable answer.

LLM: We use google/flan-t5-base, an instruction-tuned model known for its ability to follow instructions and perform well on reasoning tasks.

Prompt Engineering: The final prompt is carefully constructed to guide the LLM. It explicitly instructs the model to answer the question based only on the provided context and to state when the answer is not available. This is a critical guardrail against model hallucination and ensures the answers are grounded in the source documents.

Process: The top 5 reranked chunks are combined into a single context block, inserted into the prompt template along with the question, and fed to the LLM to generate the final answer.

## 3. Key Technical Choices & Justification
Component	Technology Used	Justification
PDF Parsing	unstructured.io	Excels at handling complex layouts and tables, providing cleaner, more contextually complete chunks than simple text extraction.
Embedding	all-mpnet-base-v2	A top-tier SentenceTransformer model offering excellent performance for semantic search tasks.
Vector Index	faiss.IndexFlatL2	Provides a simple, fast, and exact similarity search, which is perfect for a corpus of this size. No loss of accuracy from approximation.
Reranking	ms-marco-MiniLM-L-6-v2	The crucial component for accuracy. This lightweight cross-encoder significantly boosts relevance over the initial retrieval at a minimal computational cost.
Generation	google/flan-t5-base	A powerful and reliable instruction-tuned LLM that excels at following prompts and synthesizing information, reducing the risk of hallucination.

Export to Sheets
## 4. How to Run the Solution
Create a directory named Pdf2 in the same folder as the script.

Place all your PDF documents inside the Pdf2 directory.

Run the Python script. The system will automatically build the index.

Once the message --- RAG System Bravo is ready. --- appears, you can use the rag_system.query("Your question here") method to ask questions.
