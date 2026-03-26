import os
import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder:7b")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
LIMIT = int(os.getenv("INGEST_LIMIT", "500"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "code_search")

print(f"Initializing OpenAI client for Ollama at {OLLAMA_BASE_URL}...")
client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama" # required by the OpenAI package, but unused by Ollama
)

print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print(f"Initializing ChromaDB at {CHROMA_DB_PATH}...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def generate_description(code: str) -> str:
    """Uses Ollama to generate a one-sentence natural language description of the code."""
    prompt = f"""You are a senior software engineer. Please provide a brief, one-sentence natural language description explaining what the following Python snippet does. Return ONLY the description, nothing else.

Code:
```python
{code}
```

Description:"""
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that describes code snippets concisely."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"\nError generating description for snippet: {e}")
        return ""


def process_snippet(i: int, sample: dict):
    # Create a unique, deterministic ID
    doc_id = f"snippet_{i}"
    
    # Check if already processed to avoid duplicate work on restart
    existing = collection.get(ids=[doc_id])
    if existing and existing['ids']:
        return True # already processed
        
    code = sample.get('original_string', '')
    if not code:
        return False
        
    description = generate_description(code)
    if not description:
        return False
        
    # Generate the vector embedding using sentence-transformers
    vector = embedding_model.encode(description).tolist()
    
    metadata = {
        "original_code": code,
        "language": "python",
        "llm_description": description,
        "repository": sample.get("repository_name", ""),
        "func_name": sample.get("func_name", "")
    }
    
    # Store directly in ChromaDB
    collection.add(
        ids=[doc_id],
        embeddings=[vector],
        metadatas=[metadata],
        documents=[description]
    )
    return True


def main():
    print("Loading CodeSearchNet dataset (Python subset)...")
    dataset = load_dataset("code_search_net", "python", trust_remote_code=True)
    train_data = dataset["train"]
    
    num_samples = min(LIMIT, len(train_data))
    print(f"Processing first {num_samples} samples from the dataset...")
    
    success_count = 0
    for i in tqdm(range(num_samples), desc="Ingesting snippets"):
        success = process_snippet(i, train_data[i])
        if success:
            success_count += 1
            
    print(f"\nDone! {success_count} valid snippets have been ingested into ChromaDB.")


if __name__ == "__main__":
    main()
