Solution Bravo: An Advanced Question-Answering System
Welcome to Solution Bravo, a powerful Python system designed to read and understand your PDF documents. You can ask it questions, and it will find the most relevant information within your files to provide a clear and accurate answer.

This project is built on a modern AI technique called Retrieval-Augmented Generation (RAG). It goes beyond simple search by using a sophisticated "retrieve-and-rerank" pipeline to ensure the answers are of the highest quality and relevance.

Key Features
Advanced PDF Processing: Automatically reads and understands the layout of your PDFs, including text, paragraphs, and tables.

A Smart Two-Step Search: This system first retrieves a broad set of possible answers and then carefully reranks them to select the very best matches for your question. This ensures both speed and accuracy.

Intelligent Answer Generation: Uses a powerful language model to form a concise, human-readable answer based on the information it finds.

Powered by Standard Models: Built with well-regarded, reliable models from Hugging Face and Sentence Transformers for dependable performance.

How It Works
The system operates in two main phases: a one-time setup and the ongoing query process.

1. The Setup Phase
When you first run the setup() method, the system prepares all your documents for questioning.

Ingestion: The script scans your specified folder for PDF files and breaks each one down into smaller, manageable chunks of text.

Indexing: Each text chunk is converted into a numerical representation (known as an "embedding") that captures its meaning. These are stored in a high-speed search index, which allows the system to find relevant information almost instantly.

2. The Query Phase
When you ask a question using the query() method, the following happens:

Retrieve: The system takes your question and uses the index to find the top 50 text chunks that are most similar in meaning. This is a fast, broad search to gather all possible evidence.

Rerank: The system then uses a more advanced model to carefully compare your question against each of the 50 retrieved chunks. It scores them for relevance and selects the top 5. This step is crucial for accuracy.

Generate: Finally, the top 5 chunks are given to a large language model. It is instructed to compose a final answer based only on the provided text, ensuring the answer is factual and grounded in your documents.

Getting Started
Follow these steps to get the system up and running.

1. Prerequisites
You will need Python 3.8 or a newer version installed on your computer.

2. Installation
Project Structure: First, set up your folders. Save the code as rag_system.py. In that same directory, create a new folder named Pdf2. Place all the PDF files you want the system to read inside this Pdf2 folder.

Install Libraries: Open your terminal or command prompt and install the necessary Python libraries by running the following command:

Bash

pip install "unstructured[pdf,yolox]" sentence-transformers faiss-cpu transformers torch accelerate
(Note: If your computer has a supported NVIDIA GPU, you can install faiss-gpu instead of faiss-cpu for a significant performance increase.)

3. How to Use the System
The main script is designed to handle the setup process automatically when you run it.

Run the Setup:
From your terminal, navigate to the project directory and run the Python script:

Bash

python rag_system.py
The script will begin by initializing the models, processing all the documents in the Pdf2 folder, and building the search index. You will see status messages printed to the console during this process.

Ask a Question:
To ask a question, you can edit the bottom of the rag_system.py script. Find the if __name__ == '__main__': block and add your call to the query method.

Example:

Python

(rest of the code)

if __name__ == '__main__':
    # ... (setup code is here)

    else:
        rag_system = RAGSystemBravo(pdf_directory=corpus_directory)
        rag_system.setup()

        # --- ASK YOUR QUESTIONS HERE ---
        rag_system.query("What were the key takeaways from the annual meeting?")
        rag_system.query("Provide a summary of the project's background.")
After adding your questions, save the file and run the script again from your terminal. The system will now use the prepared index to find and generate answers.
