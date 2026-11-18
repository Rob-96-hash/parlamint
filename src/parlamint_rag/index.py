"""Load chunk embeddings into Qdrant and check search with filters.

Run with:  python -m parlamint_rag.index      (Qdrant must be running on QDRANT_URL)
"""

import uuid

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models

from parlamint_rag import config

PAYLOAD_COLS = ["chunk_id", "speech_id", "chunk_index", "date", "year",
                "speaker", "party", "orientation", "text"]


def point_id(chunk_id: str) -> str:
    """Qdrant ids must be integers or UUIDs. uuid5 turns a chunk_id into a stable
    UUID, so running the upload twice overwrites points instead of duplicating them."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def load_embeddings(path=config.EMBEDDINGS_PARQUET) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (chunk metadata, vectors as an n x 384 float32 matrix)."""
    df = pd.read_parquet(path)
    vectors = np.stack(df.pop("vector").to_numpy()).astype(np.float32)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")  # JSON has no date type
    return df, vectors


def create_collection(client: QdrantClient, dim: int) -> None:
    """(Re)create the collection with cosine distance and payload indexes."""
    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)
    client.create_collection(
        config.COLLECTION,
        # HNSW graph with default settings (m=16, ef_construct=100) is fine for ~46k vectors
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    # Payload indexes make filtered search fast, like an index on a SQL column
    client.create_payload_index(config.COLLECTION, "party", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(config.COLLECTION, "year", models.PayloadSchemaType.INTEGER)


def to_payloads(df: pd.DataFrame) -> list[dict]:
    """DataFrame rows -> JSON-ready dicts (pandas NA -> None, numpy ints -> int)."""
    records = df[PAYLOAD_COLS].to_dict("records")
    return [
        {k: (None if pd.isna(v) else v.item() if isinstance(v, np.generic) else v)
         for k, v in r.items()}
        for r in records
    ]


def upload(client: QdrantClient, df: pd.DataFrame, vectors: np.ndarray, batch_size: int = 256) -> None:
    points = (
        models.PointStruct(id=point_id(p["chunk_id"]), vector=v.tolist(), payload=p)
        for p, v in zip(to_payloads(df), vectors)
    )
    client.upload_points(config.COLLECTION, points=points, batch_size=batch_size)


def build_filter(party: str | None = None, year: int | None = None) -> models.Filter | None:
    must = []
    if party:
        must.append(models.FieldCondition(key="party", match=models.MatchValue(value=party)))
    if year:
        must.append(models.FieldCondition(key="year", match=models.MatchValue(value=year)))
    return models.Filter(must=must) if must else None


def search(client: QdrantClient, query_vector: np.ndarray, k: int = 5,
           party: str | None = None, year: int | None = None) -> list[models.ScoredPoint]:
    return client.query_points(
        config.COLLECTION,
        query=query_vector,
        query_filter=build_filter(party, year),
        limit=k,
    ).points


def sanity_check(client: QdrantClient, df: pd.DataFrame, vectors: np.ndarray) -> None:
    n_points = client.count(config.COLLECTION, exact=True).count
    assert n_points == len(df), f"{n_points} points in Qdrant, but {len(df)} chunks"

    # Self-retrieval: searching with a chunk's own vector must give score ~1.0
    sample = np.random.default_rng(0).choice(len(df), size=min(20, len(df)), replace=False)
    top_scores = [search(client, vectors[i], k=1)[0].score for i in sample]
    assert min(top_scores) > 0.999, f"self-retrieval failed, min score {min(top_scores):.4f}"

    # Filters: every hit must match party and year
    hits = search(client, vectors[0], k=10, party="Grüne", year=2021)
    assert hits and all(h.payload["party"] == "Grüne" and h.payload["year"] == 2021 for h in hits)

    print(f"[index] OK: {n_points} points, self-retrieval and filters work")
    print(f"[index] example: chunks most similar to {df['chunk_id'].iloc[0]} (Grüne, 2021):")
    for h in hits[:3]:
        print(f"  {h.score:.3f}  {h.payload['speaker']:<25} {h.payload['date']}  {h.payload['text'][:70]}...")


def main() -> None:
    df, vectors = load_embeddings()
    client = QdrantClient(url=config.QDRANT_URL, timeout=60)
    print(f"[index] {len(df)} chunks -> {config.QDRANT_URL}, collection '{config.COLLECTION}'")
    create_collection(client, vectors.shape[1])
    upload(client, df, vectors)
    sanity_check(client, df, vectors)


if __name__ == "__main__":
    main()
