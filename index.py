"""
Typesense Indexing & Multi-Modal Search Module for Customer Support RAG Bot.

INTERVIEW RATIONALE & DESIGN CHOICES:
-------------------------------------
1. Why Typesense for RAG Search?
   - Typesense is an open-source, in-memory C++ search engine with native support for
     both BM25 keyword search, typo-tolerance, and HNSW dense vector search in a single engine.
   - Eliminates the need for running separate systems (e.g. Elasticsearch for keywords + Chroma for vectors).

2. Field Weighting in Keyword Search (title:3, heading_path:2, content:1):
   - When a user query matches the article title or section heading, that document is usually
     much more relevant than a document where the keyword appears only once in body text.

3. Hybrid Search Mechanics & Alpha Parameter:
   - Typesense combines keyword rank and vector cosine similarity:
     `vector_query = "embedding:([vec...], k: top_k, alpha: 0.7)"`
   - Alpha = 1.0: Pure Dense Vector search (semantic only).
   - Alpha = 0.0: Pure BM25 Keyword search (exact terms and error codes).
   - Alpha = 0.7: 70% Vector + 30% Keyword. Catches semantic nuances while strictly respecting
     exact technical tokens (e.g. `st.cache_resource`, `secrets.toml`, `403 Forbidden`).

4. Resilient Fallback Engine:
   - Includes an embedded in-memory Hybrid search fallback to ensure testing and offline
     evaluations execute seamlessly without downtime.
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import typesense

from config import settings
from embeddings import EmbeddingManager


def get_typesense_client(
    host: Optional[str] = None,
    port: Optional[str] = None,
    protocol: Optional[str] = None,
    api_key: Optional[str] = None
) -> typesense.Client:
    """Initialize Typesense Client from settings."""
    return typesense.Client({
        "nodes": [{
            "host": host or settings.typesense_host,
            "port": str(port or settings.typesense_port),
            "protocol": protocol or settings.typesense_protocol
        }],
        "api_key": api_key or settings.typesense_api_key,
        "connection_timeout_seconds": 3
    })


def is_typesense_alive(client: Optional[typesense.Client] = None) -> bool:
    """Check if Typesense server is reachable."""
    try:
        c = client or get_typesense_client()
        health = c.operations.is_healthy()
        return bool(health)
    except Exception:
        return False


def get_collection_schema(
    collection_name: Optional[str] = None,
    dimension: Optional[int] = None
) -> Dict:
    """Define Typesense schema with text fields, metadata, and dense vector field."""
    col_name = collection_name or settings.typesense_collection_name
    dim = dimension or settings.embedding_dimension

    return {
        "name": col_name,
        "fields": [
            {"name": "id", "type": "string", "facet": False},
            {"name": "doc_id", "type": "string", "facet": True},
            {"name": "title", "type": "string", "facet": True, "sort": False},
            {"name": "heading_path", "type": "string", "facet": False, "sort": False},
            {"name": "section_title", "type": "string", "facet": False, "sort": False},
            {"name": "content", "type": "string", "facet": False, "sort": False},
            {"name": "raw_content", "type": "string", "facet": False, "index": False},
            {"name": "url", "type": "string", "facet": False, "index": False},
            {"name": "char_count", "type": "int32", "facet": False},
            {"name": "token_estimate", "type": "int32", "facet": False},
            {
                "name": "embedding",
                "type": "float[]",
                "num_dim": dim,
                "vec_dist": "cosine"
            }
        ],
        "default_sorting_field": "char_count"
    }


def create_or_recreate_collection(
    client: typesense.Client,
    collection_name: Optional[str] = None,
    dimension: Optional[int] = None,
    force_recreate: bool = True
) -> None:
    """Create Typesense collection with vector index."""
    col_name = collection_name or settings.typesense_collection_name
    schema = get_collection_schema(col_name, dimension)

    # Check if collection exists
    try:
        client.collections[col_name].retrieve()
        if force_recreate:
            print(f"🔄 Dropping existing collection '{col_name}'...")
            client.collections[col_name].delete()
            print(f"✨ Creating collection '{col_name}' (dim={schema['fields'][-1]['num_dim']})...")
            client.collections.create(schema)
        else:
            print(f"✓ Collection '{col_name}' already exists.")
    except Exception:
        print(f"✨ Creating new collection '{col_name}' (dim={schema['fields'][-1]['num_dim']})...")
        client.collections.create(schema)


def index_chunks_to_typesense(
    chunks: List[Dict],
    client: Optional[typesense.Client] = None,
    collection_name: Optional[str] = None,
    batch_size: int = 100
) -> int:
    """Bulk index document chunks with dense vectors into Typesense."""
    c = client or get_typesense_client()
    col_name = collection_name or settings.typesense_collection_name

    dim = len(chunks[0]["embedding"]) if chunks and "embedding" in chunks[0] else settings.embedding_dimension
    create_or_recreate_collection(c, collection_name=col_name, dimension=dim, force_recreate=True)

    print(f"📦 Indexing {len(chunks)} documents into Typesense collection '{col_name}'...")
    total_indexed = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        results = c.collections[col_name].documents.import_(batch, {"action": "upsert"})
        # Verify success
        success_count = sum(1 for r in results if r.get("success", False))
        total_indexed += success_count
        print(f"  Indexed {total_indexed}/{len(chunks)} chunks...")

    print(f"✅ Successfully indexed {total_indexed} chunks into Typesense!")
    return total_indexed


# ==============================================================================
# Search Implementations: Keyword (BM25), Vector (Cosine), and Hybrid
# ==============================================================================

class TypesenseSearcher:
    """
    Search engine wrapper supporting Keyword, Vector, and Hybrid search over Typesense.
    Includes seamless local fallback if Typesense daemon is starting or offline.
    """
    def __init__(
        self,
        client: Optional[typesense.Client] = None,
        collection_name: Optional[str] = None,
        chunks_file: Optional[Path] = None
    ):
        self.client = client or get_typesense_client()
        self.collection_name = collection_name or settings.typesense_collection_name
        self.embedding_mgr = EmbeddingManager()
        self.is_live = is_typesense_alive(self.client)

        self.chunks_file = chunks_file or (settings.processed_data_dir / "chunks.json")
        self._local_chunks: Optional[List[Dict]] = None
        self._local_embeddings: Optional[np.ndarray] = None

    def _ensure_local_cache(self):
        """Load local chunks into memory for zero-downtime fallback search."""
        if self._local_chunks is None and self.chunks_file.exists():
            with open(self.chunks_file, "r", encoding="utf-8") as f:
                self._local_chunks = json.load(f)
            if self._local_chunks and "embedding" in self._local_chunks[0]:
                self._local_embeddings = np.array([c["embedding"] for c in self._local_chunks], dtype=np.float32)

    def search(
        self,
        query: str,
        search_mode: Optional[str] = None,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None
    ) -> List[Dict]:
        """
        Unified search endpoint:
        - mode: "keyword", "vector", or "hybrid"
        - top_k: number of chunks to return
        - alpha: weight for hybrid search (0.0=keyword, 1.0=vector, 0.7=default)
        """
        mode = search_mode or settings.search_mode
        k = top_k or settings.top_k
        alph = alpha if alpha is not None else settings.hybrid_alpha

        if is_typesense_alive(self.client):
            if mode == "keyword":
                return self._search_typesense_keyword(query, top_k=k)
            elif mode == "vector":
                return self._search_typesense_vector(query, top_k=k)
            else:
                return self._search_typesense_hybrid(query, top_k=k, alpha=alph)
        else:
            # Embedded In-Memory Fallback
            return self._search_local_fallback(query, mode=mode, top_k=k, alpha=alph)

    def _search_typesense_keyword(self, query: str, top_k: int = 4) -> List[Dict]:
        """Pure BM25 Keyword Search with field weighting."""
        search_params = {
            "q": query,
            "query_by": "title,heading_path,content",
            "query_by_weights": "3,2,1",
            "per_page": top_k,
            "highlight_full_fields": "content",
            "num_typos": 2
        }
        resp = self.client.collections[self.collection_name].documents.search(search_params)
        return self._format_typesense_results(resp, search_type="keyword")

    def _search_typesense_vector(self, query: str, top_k: int = 4) -> List[Dict]:
        """Pure Dense Vector Search using cosine distance via multi_search POST."""
        q_vec = self.embedding_mgr.embed_query(query)
        vec_str = ",".join(f"{x:.6f}" for x in q_vec)
        search_params = {
            "collection": self.collection_name,
            "q": "*",
            "vector_query": f"embedding:([{vec_str}], k:{top_k})",
            "per_page": top_k
        }
        resp = self.client.multi_search.perform({"searches": [search_params]}, {})
        first_result = resp.get("results", [{}])[0]
        return self._format_typesense_results(first_result, search_type="vector")

    def _search_typesense_hybrid(self, query: str, top_k: int = 4, alpha: float = 0.7) -> List[Dict]:
        """
        Hybrid Search combining BM25 Keyword match with Dense Vector similarity.
        alpha=0.7 gives 70% vector + 30% keyword weight via multi_search POST.
        """
        q_vec = self.embedding_mgr.embed_query(query)
        vec_str = ",".join(f"{x:.6f}" for x in q_vec)
        search_params = {
            "collection": self.collection_name,
            "q": query,
            "query_by": "title,heading_path,content",
            "query_by_weights": "3,2,1",
            "vector_query": f"embedding:([{vec_str}], k:{top_k}, alpha:{alpha})",
            "per_page": top_k
        }
        resp = self.client.multi_search.perform({"searches": [search_params]}, {})
        first_result = resp.get("results", [{}])[0]
        return self._format_typesense_results(first_result, search_type="hybrid")

    @staticmethod
    def _format_typesense_results(resp: Dict, search_type: str = "hybrid") -> List[Dict]:
        """Format Typesense API search hits into uniform chunk dictionaries."""
        hits = resp.get("hits", [])
        formatted = []
        for rank, hit in enumerate(hits, 1):
            doc = hit.get("document", {})
            score = hit.get("hybrid_search_info", {}).get("rank_score") or hit.get("vector_distance") or hit.get("text_match", 0.0)
            formatted.append({
                "rank": rank,
                "id": doc.get("id"),
                "doc_id": doc.get("doc_id"),
                "title": doc.get("title"),
                "url": doc.get("url"),
                "heading_path": doc.get("heading_path"),
                "section_title": doc.get("section_title"),
                "content": doc.get("content"),
                "raw_content": doc.get("raw_content", doc.get("content")),
                "score": float(score) if score is not None else 0.0,
                "search_type": search_type
            })
        return formatted

    def _search_local_fallback(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 4,
        alpha: float = 0.7
    ) -> List[Dict]:
        """In-memory local search fallback (Keyword BM25 approximation + Vector Cosine)."""
        self._ensure_local_cache()
        if not self._local_chunks or self._local_embeddings is None:
            return []

        q_terms = set(re.findall(r"\w+", query.lower()))
        q_vec = np.array(self.embedding_mgr.embed_query(query), dtype=np.float32)

        # 1. Compute Vector Cosine Similarity
        norm_q = np.linalg.norm(q_vec) or 1.0
        norm_chunks = np.linalg.norm(self._local_embeddings, axis=1)
        norm_chunks[norm_chunks == 0] = 1.0
        vector_scores = np.dot(self._local_embeddings, q_vec) / (norm_chunks * norm_q)

        # 2. Compute Keyword Overlap Score
        keyword_scores = np.zeros(len(self._local_chunks), dtype=np.float32)
        for i, c in enumerate(self._local_chunks):
            content_lower = c["content"].lower()
            title_lower = c["title"].lower()
            heading_lower = c["heading_path"].lower()

            matches = sum(1 for term in q_terms if term in content_lower)
            title_matches = sum(2 for term in q_terms if term in title_lower)
            heading_matches = sum(1.5 for term in q_terms if term in heading_lower)

            total_score = matches + title_matches + heading_matches
            keyword_scores[i] = total_score / (len(q_terms) or 1.0)

        # Normalize keyword scores to [0, 1]
        max_kw = np.max(keyword_scores) if np.max(keyword_scores) > 0 else 1.0
        keyword_scores = keyword_scores / max_kw

        # 3. Combine scores based on mode
        if mode == "vector":
            final_scores = vector_scores
        elif mode == "keyword":
            final_scores = keyword_scores
        else:  # hybrid
            final_scores = alpha * vector_scores + (1.0 - alpha) * keyword_scores

        top_indices = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            chunk = self._local_chunks[idx]
            results.append({
                "rank": rank,
                "id": chunk["id"],
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "url": chunk["url"],
                "heading_path": chunk["heading_path"],
                "section_title": chunk.get("section_title", ""),
                "content": chunk["content"],
                "raw_content": chunk.get("raw_content", chunk["content"]),
                "score": float(final_scores[idx]),
                "search_type": f"{mode}_local"
            })

        return results


def run_indexing_pipeline(
    chunks_path: Optional[Path] = None,
    collection_name: Optional[str] = None
) -> None:
    """Orchestrate collection schema creation and document indexing."""
    input_file = chunks_path or (settings.processed_data_dir / "chunks.json")
    if not input_file.exists():
        raise FileNotFoundError(f"Processed chunks file not found at {input_file}. Please run chunking.py first.")

    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    client = get_typesense_client()
    if is_typesense_alive(client):
        index_chunks_to_typesense(chunks, client=client, collection_name=collection_name)
    else:
        print(f"⚠️ Typesense server not reachable at {settings.typesense_protocol}://{settings.typesense_host}:{settings.typesense_port}.")
        print(f"✓ {len(chunks)} chunks are prepared in {input_file} and ready for both Typesense indexing and in-memory search.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index chunks into Typesense or test search modes.")
    parser.add_argument("--index", action="store_true", help="Run indexing into Typesense")
    parser.add_argument("--query", type=str, default="How to deploy Streamlit app with secrets?", help="Test query")
    parser.add_argument("--mode", type=str, choices=["hybrid", "vector", "keyword"], default="hybrid", help="Search mode")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results")
    args = parser.parse_args()

    if args.index:
        run_indexing_pipeline()

    print(f"\n🔎 Testing Search (Mode: {args.mode}, Query: '{args.query}'):")
    searcher = TypesenseSearcher()
    hits = searcher.search(query=args.query, search_mode=args.mode, top_k=args.top_k)

    for h in hits:
        print(f"\n[Rank {h['rank']}] Score: {h['score']:.4f} | {h['title']}")
        print(f"  Path: {h['heading_path']}")
        print(f"  URL: {h['url']}")
        print(f"  Snippet: {h['content'][:140]}...")
