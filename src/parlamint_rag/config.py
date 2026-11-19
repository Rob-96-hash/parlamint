"""Central settings: paths, data source and cleaning thresholds."""

import os
from pathlib import Path

# Project root = two levels above this file (src/parlamint_rag/config.py)
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# ParlaMint-AT 4.x, CLARIN.SI handle 11356/1912, licence CC BY 4.0
PARLAMINT_URL = (
    "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1912/"
    "ParlaMint-AT.tgz?sequence=1&isAllowed=y"
)
TGZ_PATH = RAW_DIR / "ParlaMint-AT.tgz"

YEARS = range(2018, 2023)  # 2018 to 2022 inclusive
MIN_WORDS = 50

SPEECHES_PARQUET = PROCESSED_DIR / "speeches.parquet"
DUCKDB_PATH = PROCESSED_DIR / "parlamint.duckdb"

# Chunking (in words). 250 words are roughly 350 to 400 e5 tokens, below the 512 limit.
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
CHUNKS_PARQUET = PROCESSED_DIR / "chunks.parquet"

# Embeddings
EMBED_MODEL = "intfloat/multilingual-e5-small"  # 384 dimensions
EMBEDDINGS_PARQUET = PROCESSED_DIR / "embeddings.parquet"

# Vector database (environment variable lets docker-compose point to the qdrant service)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = "parlamint_chunks"

# Generation: local LLM via Ollama, no external API
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:1.7b")
TOP_K = 5
