# Semantic Code Search Engine

A locally-hosted semantic code search engine that uses natural language to search through Python code snippets. Built with entirely open-source, local models.

## Architecture

This engine works in a two-stage pipeline:
1. **Ingestion**: Uses the `CodeSearchNet` dataset (Python subset). For each snippet, a local instance of Ollama (`qwen2.5-coder`) generates a concise, one-sentence description. This description is then embedded using `sentence-transformers` (`all-MiniLM-L6-v2`) and stored in a local ChromaDB instance along with the original code and metadata.
2. **Search**: Given a natural language query, it is embedded using `all-MiniLM-L6-v2`. We retrieve the top 20 candidate snippets from ChromaDB. These candidates are then re-ranked using a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to return the most relevant top 5 results to the user.

## Requirements & Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   Create or modify the `.env` file in the root directory to configure model endpoints and settings.

3. **Setup Ollama**:
   Ensure you have [Ollama](https://ollama.com/) installed and running locally. We recommend using `qwen2.5-coder:7b` (fits nicely in ~5GB of VRAM).
   ```bash
   ollama pull qwen2.5-coder:7b
   ```

## Usage

### 1. Ingestion
Run the ingestion script. It will connect to Ollama to generate descriptions and embed them into the local ChromaDB.
```bash
python ingest.py
```

### 2. Search (CLI)
You can search the ingested snippets directly from the terminal.
```bash
python search.py "how do I parse a JSON string?"
```

### 3. Search (API)
Start the FastAPI server:
```bash
python api.py
```
Or start via uvicorn directly:
```bash
uvicorn api:app --reload
```
You can now make POST requests to the `/search` endpoint:
```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "parse json string", "top_k": 5}'
```

Visit `http://localhost:8000/docs` in your browser to test the API via the interactive Swagger UI.
