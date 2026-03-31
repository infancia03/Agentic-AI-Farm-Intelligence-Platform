"""
AgriFarm — RAG: Ingest + Retrieve
ChromaDB with local sentence-transformers embeddings (free, no API key).
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from loguru import logger

CHROMA_DIR       = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
COLLECTION_NAME  = "agrifarm_knowledge"
KNOWLEDGE_FILE   = Path("./data/agri_knowledge.json")


def _get_collection():
    import chromadb
    from chromadb.utils import embedding_functions
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"   # 22 MB, downloads once, fully free
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_knowledge_base(force: bool = False) -> int:
    """Load agri_knowledge.json → ChromaDB. Idempotent."""
    if not KNOWLEDGE_FILE.exists():
        logger.warning(f"{KNOWLEDGE_FILE} not found. Run: python data/seed_data.py")
        return 0

    col = _get_collection()
    if col.count() > 0 and not force:
        logger.info(f"ChromaDB already has {col.count()} docs — skipping ingest.")
        return col.count()

    with open(KNOWLEDGE_FILE) as f:
        docs = json.load(f)

    BATCH = 50
    for i in range(0, len(docs), BATCH):
        batch = docs[i:i + BATCH]
        col.upsert(
            ids=[d["id"] for d in batch],
            documents=[d["content"] for d in batch],
            metadatas=[{"type": d.get("type",""), "title": d.get("title",""), "crop": d.get("crop","general")} for d in batch],
        )
    logger.info(f"✓ ChromaDB: {col.count()} documents")
    return col.count()


def query_knowledge_base(
    query: str,
    n_results: int = 3,
    crop_filter: str | None = None,
) -> list[dict]:
    """Semantic search → list of {content, metadata, distance}."""
    col = _get_collection()
    if col.count() == 0:
        ingest_knowledge_base()

    kwargs: dict = dict(
        query_texts=[query],
        n_results=min(n_results, max(col.count(), 1)),
        include=["documents", "metadatas", "distances"],
    )
    if crop_filter:
        kwargs["where"] = {"crop": crop_filter}

    try:
        res = col.query(**kwargs)
        return [
            {"content": doc, "metadata": meta, "distance": round(dist, 4)}
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]
    except Exception as e:
        logger.error(f"ChromaDB query failed: {e}")
        return []


if __name__ == "__main__":
    n = ingest_knowledge_base(force=True)
    print(f"✅ Ingested {n} documents")
