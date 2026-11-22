"""Second retrieval stage: re-sort vector search candidates with a cross-encoder.

The embedding model (bi-encoder) turns question and chunk into one vector each, separately,
and compares the vectors. That is fast enough for all 46,254 chunks, but a single vector
loses details such as numbers, names and negations.

A cross-encoder reads question and chunk together in one pass, so every question word can
attend to every chunk word. That is more accurate but far too slow for the whole corpus,
so it only re-sorts the top candidates of the vector search.
"""

from sentence_transformers import CrossEncoder

from parlamint_rag import config


def load_reranker(device: str = "cpu") -> CrossEncoder:
    """Load the cross-encoder model (multilingual, trained on question/passage pairs)."""
    # max_length in tokens: question + chunk (~370 tokens) fit, longer pairs are cut at the end
    return CrossEncoder(config.RERANK_MODEL, device=device, max_length=512)


def rerank(model: CrossEncoder, question: str, hits: list, k: int) -> list:
    """Sort vector search hits by cross-encoder relevance and return the best k.

    The `score` of every returned hit is replaced by the cross-encoder score
    (higher = more relevant; only the order matters, not the absolute value).
    """
    if not hits:
        return []
    # One (question, chunk text) pair per candidate: the model reads both texts together
    pairs = [(question, hit.payload["text"]) for hit in hits]
    # One relevance score per pair, computed in batches of 16 pairs
    scores = model.predict(pairs, batch_size=16)
    # Highest score first, keep the best k
    ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)[:k]
    for hit, score in ranked:
        hit.score = float(score)
    return [hit for hit, _ in ranked]
