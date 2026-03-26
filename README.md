# Semantic Code Search Engine

A semantic code search engine that takes your natural language query, maps it to semantic space, and returns the most relevant code snippets from the Hugging Face `CodeSearchNet` dataset. It leverages a modern two-stage retrieval pipeline with generative descriptions.

## Architecture

1. **Ingestion Phase (`ingest.py`)**: 
   - Downloads a subset of Python snippets from the `CodeSearchNet` dataset.
   - Generates natural-language summaries for each snippet using a local LLM through **Ollama** (`qwen2.5-coder:7b`). This runs smoothly on consumer GPUs like the RTX 3050 (6GB VRAM) and operates entirely locally without API costs.
   - Embeds those summaries into a vector space with `sentence-transformers` (`all-MiniLM-L6-v2`).
   - Stores the embeddings + metadata (code, description, local path) into an easily deployable local `ChromaDB`.

2. **Search Phase (`search.py`)**: 
   - Receives a user query and embeds it instantly using the same bi-encoder.
   - Retrieves the top-20 nearest vectors (candidate snippets) using `ChromaDB`.
   - Passes the `(query, description)` pairs through a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for highly accurate re-ranking.
   - Returns the top 5 most contextually relevant snippets.

3. **REST API (`api.py`)**: 
   - Provides a FastAPI POST `/search` endpoint to easily consume the code search functionality in other applications.

## Setup Instructions

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install and Configure Ollama**:
   Since this project uses a local quantized model to stay cost-free, you need to install [Ollama](https://ollama.com/) if you haven't already.
   
   Once installed, open a terminal and pull the `qwen2.5-coder` 7B model locally:
   ```bash
   ollama run qwen2.5-coder:7b
   ```
   *Note: Our `ingest.py` script utilizes the `openai` Python wrapper pointed to Ollama's local OpenAI-compatible endpoint at `http://localhost:11434` for a drop-in integration!*

3. **Run Ingestion**:
   Generates the DB structure locally. It limits ingestion to 500 samples by default (see `.env`) so the pipeline finishes reasonably fast on local hardware. Ensure the Ollama app is running in the background.
   ```bash
   python ingest.py
   ```

4. **Testing Search via CLI**:
   ```bash
   python search.py "read content of a json file"
   ```

5. **Start API Server**:
   ```bash
   python api.py
   ```
   Alternatively, run via uvicorn directly:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```
   You can access the generated API docs at [http://localhost:8000/docs](http://localhost:8000/docs).
