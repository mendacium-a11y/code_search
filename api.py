from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Import the search function from our search module
from search import search_code

app = FastAPI(
    title="Semantic Code Search API",
    description="API for the semantic code search engine relying on local LLM ingestion, ChromaDB, and Cross-Encoder re-ranking.",
    version="1.0.0"
)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    retrieve_k: int = 20

class SearchResult(BaseModel):
    id: str
    score: float
    description: str
    code: str
    language: str
    repository: Optional[str] = ""
    func_name: Optional[str] = ""

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

@app.post("/search", response_model=SearchResponse)
async def api_search(request: SearchRequest):
    try:
        results = search_code(
            query=request.query, 
            top_k=request.top_k, 
            retrieve_k=request.retrieve_k
        )
        return SearchResponse(
            query=request.query,
            results=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
