import os
import argparse
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "code_search")

print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print(f"Loading cross-encoder model '{CROSS_ENCODER_MODEL_NAME}'...")
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)

print(f"Connecting to ChromaDB at {CHROMA_DB_PATH}...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
except ValueError:
    print(f"Collection '{COLLECTION_NAME}' not found. Please run ingest.py first.")
    exit(1)

def search_code(query: str, top_k: int = 5, retrieve_k: int = 20):
    """Search for code snippets matching the natural language query."""
    
    # 1. Embed the user query
    query_vector = embedding_model.encode(query).tolist()
    
    # 2. Retrieve top `retrieve_k` candidates from ChromaDB
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=retrieve_k
    )
    
    if not results['ids'] or not results['ids'][0]:
        return []
        
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    ids = results['ids'][0]
    
    # 3. Re-rank using cross-encoder
    # CrossEncoder expects pairs of (query, document)
    cross_inp = [[query, doc] for doc in documents]
    cross_scores = cross_encoder.predict(cross_inp)
    
    # Sort results by score in descending order
    # Create a list of tuples: (score, id, document, metadata)
    scored_results = list(zip(cross_scores, ids, documents, metadatas))
    scored_results.sort(key=lambda x: x[0], reverse=True)
    
    # 4. Return top `top_k` results
    top_results = []
    for score, doc_id, doc, meta in scored_results[:top_k]:
        top_results.append({
            "id": doc_id,
            "score": float(score),
            "description": doc,
            "code": meta.get("original_code", ""),
            "language": meta.get("language", ""),
            "repository": meta.get("repository", ""),
            "func_name": meta.get("func_name", "")
        })
        
    return top_results

def main():
    parser = argparse.ArgumentParser(description="Semantic Code Search Engine")
    parser.add_argument("query", type=str, help="Natural language search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()
    
    print(f"\nSearching for: '{args.query}'...\n")
    results = search_code(args.query, top_k=args.top_k)
    
    if not results:
        print("No results found.")
        return
        
    for i, res in enumerate(results, 1):
        print(f"{'='*50}")
        print(f"Rank {i} (Score: {res['score']:.4f})")
        print(f"Document ID: {res['id']}")
        if res['func_name']:
            print(f"Function: {res['func_name']} ({res['repository']})")
        print(f"Description: {res['description']}")
        print(f"\nCode ({res['language']}):\n{res['code']}")
        
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
