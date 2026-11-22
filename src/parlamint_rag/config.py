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

# Reranking: a cross-encoder re-sorts the top RERANK_CANDIDATES vector search hits
# (see rerank.py). RERANK_MODEL="" switches it off, e.g. for before/after comparisons.
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
RERANK_CANDIDATES = 30
# Relevance gate: if even the best reranked chunk scores below this cross-encoder score,
# the system answers NO_ANSWER without asking the LLM. Only used with the reranker
# (vector search scores are on another scale). RELEVANCE_THRESHOLD="" switches it off.
# 0.0 was chosen on the evaluation set (eval/results/gate_rerank.csv): it blocks two
# unanswerable questions and one answerable question whose answer was wrong anyway,
# and no correct answer. Higher values start to block correct answers.
_threshold = os.environ.get("RELEVANCE_THRESHOLD", "0.0")
RELEVANCE_THRESHOLD = float(_threshold) if _threshold else None
# Fixed sentence for "the sources do not answer this"; also used to measure abstention
NO_ANSWER = "Dazu finde ich in den Quellen keine Antwort."

# Evaluation
EVAL_DIR = ROOT / "eval"
QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
