"""
Indexes a local codebase into a named ChromaDB collection.
Each codebase gets its own isolated collection: codebase_<name>
"""

import hashlib
import os
from pathlib import Path
from typing import Optional

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from chunker import chunk_file, SUPPORTED_EXTENSIONS

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Directories to always skip
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt", "coverage", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".idea", ".vscode",
}

_llm_client: Optional[OpenAI] = None
_embed_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None


def _get_clients():
    global _llm_client, _embed_model, _chroma_client
    if _llm_client is None:
        _llm_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _llm_client, _embed_model, _chroma_client


def _file_hash(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _generate_description(llm: OpenAI, code: str, language: str) -> str:
    try:
        resp = llm.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are an expert {language} programmer. "
                        "Give a concise, 1-2 sentence description of what the following code does. "
                        "Focus on the core functionality. Do not include code in your answer."
                    ),
                },
                {"role": "user", "content": code[:2000]},  # cap to avoid huge prompts
            ],
            max_tokens=120,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return ""


def _walk_codebase(root: Path) -> list[Path]:
    """Yield all supported source files under root, skipping ignored dirs."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped dirs in-place so os.walk won't descend into them
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in filenames:
            p = Path(dirpath) / fname
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(p)
    return files


def _collection_name(codebase_name: str) -> str:
    # ChromaDB collection names must be alphanumeric + hyphens/underscores
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in codebase_name)
    return f"codebase_{safe}"


def get_all_codebase_collections(chroma: chromadb.PersistentClient) -> list[str]:
    return [c.name for c in chroma.list_collections() if c.name.startswith("codebase_")]


def index_codebase(
    path: str,
    name: str,
    progress_callback=None,   # callable(current, total, file_path)
) -> dict:
    """
    Index all supported source files under `path` into collection `codebase_<name>`.
    Returns a summary dict: {total_files, skipped_files, total_chunks, new_chunks}
    """
    llm, embed, chroma = _get_clients()
    root = Path(path).resolve()
    col = chroma.get_or_create_collection(name=_collection_name(name))

    files = _walk_codebase(root)
    total_files = len(files)
    skipped = 0
    total_chunks = 0
    new_chunks = 0

    for i, filepath in enumerate(files):
        if progress_callback:
            progress_callback(i + 1, total_files, str(filepath))

        fhash = _file_hash(str(filepath))
        rel_path = str(filepath.relative_to(root))

        # Check if this file's hash is already in the DB (incremental indexing)
        existing = col.get(where={"file_path": rel_path}, include=["metadatas"])
        if existing["ids"] and existing["metadatas"][0].get("file_hash") == fhash:
            skipped += 1
            continue

        # Delete old entries for this file (handles re-indexing after edits)
        if existing["ids"]:
            col.delete(ids=existing["ids"])

        chunks = chunk_file(str(filepath))
        total_chunks += len(chunks)

        for j, chunk in enumerate(chunks):
            code = chunk["code"]
            if not code.strip():
                continue

            desc = _generate_description(llm, code, chunk["language"])
            if not desc:
                continue

            embedding = embed.encode([desc]).tolist()[0]
            doc_id = f"{name}::{rel_path}::{chunk['function_name']}::{j}"

            col.upsert(
                ids=[doc_id],
                documents=[desc],
                embeddings=[embedding],
                metadatas=[{
                    "code": code,
                    "language": chunk["language"],
                    "description": desc,
                    "function_name": chunk["function_name"],
                    "file_path": rel_path,
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "file_hash": fhash,
                    "codebase": name,
                }],
            )
            new_chunks += 1

    return {
        "total_files": total_files,
        "skipped_files": skipped,
        "total_chunks": total_chunks,
        "new_chunks": new_chunks,
    }


def search_codebase(
    query: str,
    codebase_name: Optional[str] = None,
    top_k_retrieve: int = 20,
    top_k_return: int = 5,
) -> list[dict]:
    """
    Search one or all codebases.
    If codebase_name is None, merges results across all codebase_* collections.
    Re-ranks with cross-encoder after retrieval.
    """
    from sentence_transformers import CrossEncoder
    import os
    CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    _, embed, chroma = _get_clients()
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    query_embedding = embed.encode([query]).tolist()[0]

    # Determine which collections to search
    if codebase_name:
        col_names = [_collection_name(codebase_name)]
    else:
        col_names = get_all_codebase_collections(chroma)

    if not col_names:
        return []

    candidates = []
    for col_name in col_names:
        try:
            col = chroma.get_collection(col_name)
        except Exception:
            continue
        results = col.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k_retrieve, col.count()),
            include=["metadatas", "documents"],
        )
        if results["documents"] and results["documents"][0]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                candidates.append({"document": doc, "metadata": meta})

    if not candidates:
        return []

    # Re-rank all candidates together
    pairs = [[query, c["document"]] for c in candidates]
    from sentence_transformers import CrossEncoder
    scores = cross_encoder.predict(pairs)

    scored = sorted(
        [{"score": float(s), **c["metadata"]} for s, c in zip(scores, candidates)],
        key=lambda x: x["score"],
        reverse=True,
    )
    return scored[:top_k_return]
