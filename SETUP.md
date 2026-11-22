# Setup and Usage

This file explains how to run the system and how the repository is organised. The
[README](README.md) describes the approach and the results.

## Where each step runs

The pipeline has three stages, and each runs on hardware that suits it:

| Stage | Where I ran it | Why |
|---|---|---|
| ETL and chunking | laptop | pandas on about 60 MB of text |
| Embeddings | Google Colab (T4 GPU) | 46k chunks take 7 minutes on a GPU |
| Qdrant, Ollama, app, evaluation | GitHub Codespaces (2 cores, 8 GB RAM, 32 GB disk) | Docker is preinstalled |

## 1. Data and chunks

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
python -m parlamint_rag.etl       # downloads ~330 MB, writes data/processed/speeches.parquet
python -m parlamint_rag.chunks    # writes data/processed/chunks.parquet
```

## 2. Embeddings

Open [notebooks/colab_runner.ipynb](notebooks/colab_runner.ipynb) in Colab with a GPU runtime.
The notebook is a thin runner without its own logic. It does four things:
1. clones the repository
2. installs the requirements
3. runs `etl`, `chunks` and `embed`
4. copies `embeddings.parquet` to Google Drive

Put the file into `data/processed/` on the machine that runs Docker.

## 3. Start the stack

```bash
docker compose up -d --build
docker compose exec ollama ollama pull qwen3:1.7b               # once, ~1.4 GB
docker compose run --rm app python -m parlamint_rag.index       # once, loads 46k vectors
```

Open http://localhost:8501. In Codespaces, open the forwarded port 8501 in the *Ports* tab.
The first question takes longer, because the embedding model and the reranker are downloaded
once into a volume.

## 4. Configuration

All settings live in [src/parlamint_rag/config.py](src/parlamint_rag/config.py). The following
can be changed through environment variables, for example in front of `docker compose up`:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_MODEL` | `qwen3:1.7b` | Ollama model for the answers |
| `RERANK_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | empty = vector search only |
| `RELEVANCE_THRESHOLD` | `0.0` | minimum cross-encoder score; empty = no relevance gate |
| `QDRANT_URL`, `OLLAMA_URL` | set by Compose | addresses of the two services |

## 5. Evaluation

```bash
# retrieval only (minutes)
docker compose run --rm -v "$PWD/eval:/app/eval" app python -m parlamint_rag.evaluate --retrieval-only --tag rerank

# retrieval and answers (about 50 minutes on 2 CPU cores)
docker compose run --rm -v "$PWD/eval:/app/eval" app python -m parlamint_rag.evaluate --tag rerank

# summary after rating the answers by hand (columns faithful and correct, 1 or 0)
python -m parlamint_rag.evaluate --summarize eval/results/answers_qwen3-1.7b_rerank.csv

# best reranker score per question and the simulated relevance gate
docker compose run --rm -v "$PWD/eval:/app/eval" app python -m parlamint_rag.evaluate --gate eval/results/answers_qwen3-1.7b_rerank.csv --tag rerank
```

The figures in the README are made from these result files, without any model:

```bash
python -m parlamint_rag.plots     # writes docs/recall_at_k.png and docs/gate_scores.png
```

The questions are in [eval/questions.jsonl](eval/questions.jsonl), and all result files are in
[eval/results/](eval/results/):

| Files | Content |
|---|---|
| `retrieval*.csv` | rank of the first gold speech per question, Recall@k and MRR |
| `answers_*.csv` | the generated answers with automatic checks and manual ratings: `answers_qwen3-1.7b.csv` (baseline), `_fix` (stricter prompt, rejected), `_rerank` (with reranker) |
| `top_scores_rerank.csv`, `gate_rerank.csv` | the relevance gate measurement |

## Repository layout

```
src/parlamint_rag/
  config.py     paths, model names, thresholds (environment variables for Docker)
  etl.py        download, extract, clean, validate, store (Parquet + DuckDB)
  chunks.py     overlapping word windows with speech metadata
  embed.py      E5 embeddings with "query: " / "passage: " prefixes
  index.py      Qdrant collection, payload indexes, filtered search, sanity checks
  rerank.py     cross-encoder second stage
  rag.py        retrieve -> relevance gate -> prompt -> cited answer
  verify.py     citation and number checks
  evaluate.py   Recall@k, MRR, answer metrics, relevance gate simulation
  plots.py      README figures from the result files
app/streamlit_app.py                   user interface
eval/questions.jsonl, eval/results/    evaluation set and all result files
notebooks/colab_runner.ipynb           GPU runner for the embeddings
Dockerfile, docker-compose.yml         app image and the three-container setup
```

## Practical notes

- **Disk space.** The Ollama image alone takes 8.5 GB on disk, because it ships GPU libraries. The app image and its build cache need several GB more. On a 32 GB Codespace, run `docker builder prune -af` after a build. For the same reason, Compose mounts `src/` and `app/` into the app container: a code change needs only `docker compose up -d`, not a rebuild.
- **Ollama crashes.** On two cores, the Ollama runner crashed about three times in 25 answers ("model runner has unexpectedly stopped"). `evaluate.py` saves after every answer, retries after 15 seconds, and continues an interrupted run with `--resume`.
- **Ports.** Qdrant and Ollama are bound to `127.0.0.1`, so they are not reachable from other machines. In Codespaces, keep the visibility of port 8501 on *Private*.
