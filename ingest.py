import os
import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load config from .env
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", 500))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 10))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "code_search")

# Initialize models and clients
# We use the OpenAI client pointed to Ollama's local wrapper for drop-in compatibility
client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama" # Required for the client but ignored by Ollama
)

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

def generate_description(code_snippet: str) -> str:
    """Generate a concise natural language description of the code using local LLM via Ollama."""
    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert programmer. Give a concise, 1-2 sentence description of what the following Python code does. Focus on the core functionality."
                },
                {"role": "user", "content": code_snippet}
            ],
            max_tokens=100,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating description: {e}")
        return ""

def main():
    print("Loading CodeSearchNet dataset...")
    # Load the Python split of CodeSearchNet
    dataset = load_dataset("code_search_net", "python", split="train")
    
    # Take the first MAX_SAMPLES
    subset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))
    
    print(f"Processing {len(subset)} snippets...")
    
    for i in tqdm(range(0, len(subset), BATCH_SIZE), desc="Processing Batches"):
        batch = subset[i:i+BATCH_SIZE]
        
        descriptions = []
        valid_indices = []
        codes = []
        
        # Dataset field containing the code
        code_keys = ["func_code_string", "whole_func_string", "code"]
        code_key = next((k for k in code_keys if k in batch), None)
        if not code_key:
            print(f"Dataset keys: {batch.keys()}")
            print("Could not find code field in dataset!")
            break
            
        code_list = batch[code_key]
        
        # Generate descriptions sequentially for this batch 
        for j, code in enumerate(code_list):
            desc = generate_description(code)
            if desc:
                descriptions.append(desc)
                valid_indices.append(j)
                codes.append(code)
        
        if not descriptions:
            continue
            
        # Embed descriptions using SentenceTransformers
        embeddings = embedding_model.encode(descriptions).tolist()
        
        # Prepare for ChromaDB
        ids = [f"doc_{i+j}" for j in valid_indices]
        metadatas = [{"code": codes[j], "language": "python", "description": descriptions[j]} for j in range(len(descriptions))]
        documents = descriptions  # the actual document content is the description
        
        # Upsert to ChromaDB
        collection.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
    print(f"\\nIngestion complete. Vector DB saved to {CHROMA_DB_PATH}")

if __name__ == "__main__":
    # Check if Ollama seems reachable (basic check)
    import requests
    try:
        requests.get("http://localhost:11434")
        main()
    except requests.exceptions.ConnectionError:
        print("Wait! Ollama doesn't seem to be running on localhost:11434.")
        print("Please start Ollama and try running ingest.py again.")
