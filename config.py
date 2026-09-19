"""
Configuration module for the Customer Support RAG System.

INTERVIEW RATIONALE & DESIGN CHOICES:
-------------------------------------
1. Temperature = 0.0:
   - For customer support, hallucination is unacceptable. Deterministic outputs
     grounded solely in retrieved context are required.

2. Hybrid Search (BM25 + Dense Vectors):
   - Dense vector embeddings capture semantic meaning (e.g., "how do I cancel?" matches "subscription termination").
   - Sparse BM25 keyword search is superior for exact terms, error codes (e.g., "ERR_404"), product SKUs, and exact URLs.
   - Hybrid search combines the strengths of both, preventing semantic drift on exact keyword queries.

3. Configurable Embedding Provider:
   - Supports local SentenceTransformers (e.g., all-MiniLM-L6-v2) for 100% free/offline operation,
     and OpenAI (text-embedding-3-small) for higher dimensional dense retrieval.

4. Chunk Size (500) & Overlap (100):
   - 500 characters (~100 tokens) provides enough context for self-contained support answers without
     diluting the vector embedding with irrelevant neighbouring topic details.
   - Overlap of 100 characters prevents context loss across chunk boundaries (especially mid-explanation).
"""

import os
from pathlib import Path
from typing import Literal

# Try loading python-dotenv if available, else manual lightweight parser
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


class Settings:
    """Centralized configuration object for the RAG pipeline."""
    def __init__(self):
        # LLM Settings
        self.llm_provider: str = os.getenv("LLM_PROVIDER", "openai")  # "openai", "groq", "ollama", "openrouter", "local"
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.groq_api_key: str = os.getenv("GROQ_API_KEY", "")
        self.llm_api_key: str = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", os.getenv("GROQ_API_KEY", "")))
        self.llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.llm_base_url: str = os.getenv("LLM_BASE_URL", "")
        self.llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

        # Embedding Settings
        self.embedding_provider: Literal["huggingface", "openai"] = os.getenv("EMBEDDING_PROVIDER", "huggingface")  # type: ignore
        self.embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))

        # Typesense Settings
        self.typesense_host: str = os.getenv("TYPESENSE_HOST", "localhost")
        self.typesense_port: str = os.getenv("TYPESENSE_PORT", "8108")
        self.typesense_protocol: str = os.getenv("TYPESENSE_PROTOCOL", "http")
        self.typesense_api_key: str = os.getenv("TYPESENSE_API_KEY", "xyz")
        self.typesense_collection_name: str = os.getenv("TYPESENSE_COLLECTION_NAME", "customer_support_docs")

        # Chunking Settings
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))

        # Retrieval & Search Settings
        self.top_k: int = int(os.getenv("TOP_K", "4"))
        self.search_mode: Literal["hybrid", "vector", "keyword"] = os.getenv("SEARCH_MODE", "hybrid")  # type: ignore
        self.hybrid_alpha: float = float(os.getenv("HYBRID_ALPHA", "0.7"))

        # Path Settings
        self.base_dir: Path = Path(__file__).resolve().parent
        self.raw_data_dir: Path = Path(__file__).resolve().parent / "data" / "raw"
        self.processed_data_dir: Path = Path(__file__).resolve().parent / "data" / "processed"
        self.eval_data_dir: Path = Path(__file__).resolve().parent / "data" / "eval"

        # Ensure runtime directories exist
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.eval_data_dir.mkdir(parents=True, exist_ok=True)


# Global settings singleton
settings = Settings()


if __name__ == "__main__":
    print("✓ Configuration initialized successfully:")
    print(f"  • LLM Model: {settings.llm_model} (Temp: {settings.llm_temperature})")
    print(f"  • Embedding: {settings.embedding_provider} ({settings.embedding_model_name}, dim={settings.embedding_dimension})")
    print(f"  • Typesense: {settings.typesense_protocol}://{settings.typesense_host}:{settings.typesense_port}")
    print(f"  • Search Mode: {settings.search_mode} (Top-K: {settings.top_k}, Alpha: {settings.hybrid_alpha})")
    print(f"  • Chunking: Size={settings.chunk_size}, Overlap={settings.chunk_overlap}")
    print(f"  • Raw Data Dir: {settings.raw_data_dir}")
