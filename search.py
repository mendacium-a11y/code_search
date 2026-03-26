import os
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "code_search")

print("Initializing search models...")
# Initialize embedding and re-ranking models
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

def search_code(query: str, top_k_retrieve: int = 20, top_k_return: int = 5):
    """
    1. Embed query using sentence-transformers
    2. Retrieve top_k_retrieve from ChromaDB
    3. Re-rank using CrossEncoder
    4. Return top_k_return results
    """
    # 1. Embed query
    query_embedding = embedding_model.encode([query]).tolist()[0]
    
    # 2. Retrieve candidates
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k_retrieve,
        include=["metadatas", "documents"]
    )
    
    if not results['documents'] or not results['documents'][0]:
        return []
        
    retrieved_docs = results['documents'][0]
    retrieved_metadatas = results['metadatas'][0]
    
    # 3. Re-rank utilizing cross-encoder
    # Form pairs of (query, document)
    cross_inp = [[query, doc] for doc in retrieved_docs]
    cross_scores = cross_encoder.predict(cross_inp)
    
    # Rank documents by cross-encoder score
    scored_results = []
    for i in range(len(cross_scores)):
        scored_results.append({
            "score": float(cross_scores[i]),
            "description": retrieved_metadatas[i]["description"],
            "code": retrieved_metadatas[i]["code"],
            "language": retrieved_metadatas[i]["language"]
        })
        
    # Sort descending by re-ranker score
    scored_results = sorted(scored_results, key=lambda x: x["score"], reverse=True)
    
    # 4. Return top_k_return
    return scored_results[:top_k_return]

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "parse json file"
    print(f"Searching for: '{query}'")
    results = search_code(query)
    for i, res in enumerate(results):
        print(f"\\n--- Result {i+1} (Score: {res['score']:.4f}) ---")
        print(f"Description: {res['description']}")
        print(f"Code:\\n{res['code'][:300]}") # truncate code output to keep it neat
        if len(res['code']) > 300:
            print("... (truncated)")
