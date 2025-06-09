# Retrieval-Augmented Generation (RAG) with LangChain, Ollama, and Pinecone

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using LangChain, Ollama, and Pinecone for semantic search and question answering over custom documents (text and PDF).

## Features
- **Document Ingestion:** Load and split text or PDF documents into manageable chunks.
- **Embeddings:** Generate vector embeddings using Ollama's embedding models.
- **Vector Store:** Store and retrieve document embeddings using Pinecone (for text) or FAISS (for PDF).
- **RAG Pipeline:** Retrieve relevant context and generate answers using a custom prompt and Ollama LLM.

## Project Structure
- `ingestion.py` — Loads, splits, embeds, and ingests text documents into Pinecone.
- `main.py` — Runs the RAG pipeline: retrieves context from Pinecone and generates answers to user queries.
- `for_pdf.py` — Loads, splits, embeds, and retrieves from PDF documents using FAISS.
- `mediumblog1.txt` — Example text document used for ingestion.
- `ReactPaper.pdf` — Example PDF document for PDF pipeline.
- `.env` — Environment variables for API keys and configuration.

## Setup
1. **Clone the repository** and navigate to the project directory.
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
   (Make sure you have Python 3.10+ and the required packages: langchain, ollama, pinecone, python-dotenv, etc.)
3. **Set up environment variables:**
   - Edit `.env` and fill in your Pinecone and LangChain API keys.
   - Example `.env`:
     ```env
     INDEX_NAME=your-pinecone-index-name
     PINECONE_API_KEY=your-pinecone-api-key
     LANGCHAIN_API_KEY=your-langchain-api-key
     LANGCHAIN_TRACING_V2=true
     LANGCHAIN_PROJECT=Your Project Name
     ```
4. **Prepare your documents:**
   - Place your text files (e.g., `mediumblog1.txt`) and/or PDF files (e.g., `ReactPaper.pdf`) in the project directory.

## Usage
### 1. Ingest Text Documents
Run the ingestion script to load, split, embed, and store your text documents in Pinecone:
```sh
python ingestion.py
```

### 2. Run the RAG Pipeline on Text
Ask questions using the main script:
```sh
python main.py
```
The script will retrieve relevant context from Pinecone and generate an answer using Ollama.

### 3. Ingest and Query PDF Documents
To process and query a PDF file using FAISS:
```sh
python for_pdf.py
```
This will build a FAISS index from the PDF and answer a sample question.

## Customization
- **Change the document:** Replace `mediumblog1.txt` or `ReactPaper.pdf` with your own files.
- **Modify the prompt:** Edit the prompt template in `main.py` or `for_pdf.py` to suit your needs.
- **Switch models:** Change the Ollama model names in the scripts for different embedding or LLM models.

## Requirements
- Python 3.10+
- [Ollama](https://ollama.com/) (for local LLM and embedding models)
- [Pinecone](https://www.pinecone.io/) account and API key (for text RAG)
- [LangChain](https://python.langchain.com/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [FAISS](https://github.com/facebookresearch/faiss) (for PDF RAG)

## References
- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Documentation](https://ollama.com/docs)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [FAISS Documentation](https://faiss.ai/)

---

**Note:** This project is for educational purposes. Make sure to secure your API keys and follow best practices for production deployments.
