"""
Embedding Management Module for Customer Support RAG Bot.

INTERVIEW RATIONALE & DESIGN CHOICES:
-------------------------------------
1. Dense Vector Embeddings:
   - Dense embeddings map textual meaning into a continuous vector space, allowing
     semantic search to find relevant passages even when user queries use different
     synonyms (e.g. "cancel plan" vs "terminate subscription").

2. HuggingFace (all-MiniLM-L6-v2) vs OpenAI (text-embedding-3-small):
   - sentence-transformers/all-MiniLM-L6-v2:
     * 384 dimensions (4x smaller vector footprint in RAM / disk compared to 1536-dim models).
     * Runs locally on CPU/GPU with zero API cost, zero rate-limit constraints, and high throughput.
   - OpenAI text-embedding-3-small:
     * 1536 dimensions, highly effective across complex domain vocabularies, but requires an API key and internet connectivity.
   - DESIGN CHOICE: We make the embedding model fully configurable via .env or CLI so the system
     can run completely free & offline or upgraded to commercial cloud models.

3. Normalized Embeddings & Distance Metrics:
   - Embeddings are L2-normalized, allowing cosine similarity to be computed efficiently
     via dot product in vector databases like Typesense.
"""

import os
from typing import List, Optional
from config import settings

# Graceful loader for LangChain & SentenceTransformer embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HAS_LANGCHAIN_HF = True
except ImportError:
    HAS_LANGCHAIN_HF = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from langchain_openai import OpenAIEmbeddings
    HAS_OPENAI_EMBEDDINGS = True
except ImportError:
    HAS_OPENAI_EMBEDDINGS = False


class EmbeddingManager:
    """
    Unified manager for text embeddings supporting both HuggingFace (SentenceTransformers)
    and OpenAI models with automatic fallback.
    """
    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None
    ):
        self.provider = provider or settings.embedding_provider
        self.model_name = model_name or settings.embedding_model_name
        self.dimension = dimension or settings.embedding_dimension
        self._model = None

    def _get_model(self):
        """Lazy loader for the selected embedding model."""
        if self._model is not None:
            return self._model

        if self.provider == "openai":
            if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
                print("⚠️ OpenAI API key not set. Falling back to local HuggingFace embeddings.")
                self.provider = "huggingface"
                self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
                self.dimension = 384
            else:
                try:
                    if HAS_OPENAI_EMBEDDINGS:
                        self._model = OpenAIEmbeddings(
                            model=self.model_name,
                            openai_api_key=settings.openai_api_key
                        )
                        print(f"✓ Initialized OpenAI embeddings: {self.model_name}")
                        return self._model
                except Exception as e:
                    print(f"⚠️ Failed to init OpenAI embeddings ({e}). Falling back to local.")
                    self.provider = "huggingface"

        # HuggingFace / Local SentenceTransformers
        try:
            if HAS_LANGCHAIN_HF:
                self._model = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    encode_kwargs={"normalize_embeddings": True}
                )
                print(f"✓ Initialized LangChain HuggingFace embeddings: {self.model_name}")
                return self._model
            elif HAS_SENTENCE_TRANSFORMERS:
                self._model = SentenceTransformer(self.model_name)
                print(f"✓ Initialized SentenceTransformer: {self.model_name}")
                return self._model
        except Exception as e:
            print(f"⚠️ Could not load sentence-transformers directly ({e}). Using deterministic fallback.")

        return None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a batch of text chunks."""
        model = self._get_model()
        if model is None:
            # Fallback deterministic pseudo-embedding for testing environment before torch loads
            return [self._generate_fallback_vector(t, self.dimension) for t in texts]

        try:
            if hasattr(model, "embed_documents"):
                return model.embed_documents(texts)
            elif hasattr(model, "encode"):
                embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                return [e.tolist() for e in embeddings]
        except Exception as e:
            print(f"⚠️ Embedding generation error: {e}. Using fallback vector.")
            return [self._generate_fallback_vector(t, self.dimension) for t in texts]

        return [self._generate_fallback_vector(t, self.dimension) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """Compute embedding for a single user query."""
        model = self._get_model()
        if model is None:
            return self._generate_fallback_vector(text, self.dimension)

        try:
            if hasattr(model, "embed_query"):
                return model.embed_query(text)
            elif hasattr(model, "encode"):
                embedding = model.encode(text, normalize_embeddings=True)
                return embedding.tolist()
        except Exception as e:
            print(f"⚠️ Query embedding error: {e}. Using fallback vector.")
            return self._generate_fallback_vector(text, self.dimension)

        return self._generate_fallback_vector(text, self.dimension)

    @staticmethod
    def _generate_fallback_vector(text: str, dim: int = 384) -> List[float]:
        """
        Deterministic normalized vector generator based on text hash.
        Used as zero-crash fallback if local neural model weights are still downloading.
        """
        import hashlib
        import math
        vec = []
        for i in range(dim):
            seed_str = f"{text}_{i}"
            h = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:8], 16)
            vec.append((h % 2000 - 1000) / 1000.0)
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [round(x / norm, 6) for x in vec]


# Global embedding manager singleton
embedding_manager = EmbeddingManager()


if __name__ == "__main__":
    print(f"Testing EmbeddingManager (Provider: {settings.embedding_provider})...")
    mgr = EmbeddingManager()
    sample_text = ["How to deploy Streamlit app to Community Cloud?", "Secrets management tutorial"]
    vecs = mgr.embed_documents(sample_text)
    print(f"✓ Embedded {len(vecs)} documents. Dimension: {len(vecs[0])}")
    q_vec = mgr.embed_query("How do I add dependencies?")
    print(f"✓ Embedded query. Dimension: {len(q_vec)}")
