"""Embed all chunks with multilingual-e5-small. Meant to run on a GPU (Colab).

Run with:  python -m parlamint_rag.embed
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from parlamint_rag import config

# E5 was trained with these prefixes. Queries and passages must use them,
# otherwise retrieval quality drops noticeably.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


def load_model(device: str | None = None) -> SentenceTransformer:
    """Load the embedding model (device=None picks the GPU if there is one)."""
    return SentenceTransformer(config.EMBED_MODEL, device=device)


def embed_passages(model: SentenceTransformer, texts: list[str], batch_size: int = 128) -> np.ndarray:
    # normalize_embeddings=True: every vector has length 1, so cosine similarity == dot product
    return model.encode(
        [PASSAGE_PREFIX + t for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)


def embed_query(model: SentenceTransformer, question: str) -> np.ndarray:
    return model.encode(QUERY_PREFIX + question, normalize_embeddings=True).astype(np.float32)


def truncation_report(model: SentenceTransformer, texts: list[str]) -> None:
    """How many chunks are longer than the model's token limit (and get cut off)?"""
    token_counts = np.array(
        [len(ids) for ids in model.tokenizer([PASSAGE_PREFIX + t for t in texts])["input_ids"]]
    )
    limit = model.max_seq_length
    print(f"[embed] tokens per chunk: median {np.median(token_counts):.0f}, "
          f"p99 {np.percentile(token_counts, 99):.0f}, max {token_counts.max()}")
    print(f"[embed] chunks over {limit} tokens (truncated): {(token_counts > limit).mean():.2%}")


def main() -> None:
    chunks = pd.read_parquet(config.CHUNKS_PARQUET)
    model = load_model()
    print(f"[embed] {config.EMBED_MODEL} on {model.device}, {len(chunks)} chunks")

    texts = chunks["text"].tolist()
    truncation_report(model, texts)
    vectors = embed_passages(model, texts)

    assert vectors.shape == (len(chunks), model.get_embedding_dimension())
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3), "vectors not normalised"

    # One self-contained file: chunk text + metadata + vector (payload for Qdrant)
    chunks["vector"] = list(vectors)
    chunks.to_parquet(config.EMBEDDINGS_PARQUET, index=False)
    print(f"[embed] {vectors.shape} -> {config.EMBEDDINGS_PARQUET}")


if __name__ == "__main__":
    main()
