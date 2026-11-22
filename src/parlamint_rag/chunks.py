"""Split speeches into overlapping word windows (chunks) for embedding.

Run with:  python -m parlamint_rag.chunks
"""

import pandas as pd

from parlamint_rag import config

# Metadata copied from the speech to every chunk (needed for filters and citations)
META_COLS = ["speech_id", "date", "year", "speaker", "party", "orientation"]


def split_words(
    words: list[str],
    size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[list[str]]:
    """Split a list of words into windows of `size` words that overlap by `overlap` words.

    If fewer than `overlap` words would be left for a last window, they are
    appended to the current window instead (no tiny tail chunks). So a chunk
    has at most size + overlap words.

    Example with size=4, overlap=1:  a b c d e f g  ->  [a b c d] [d e f g]
    """
    stride = size - overlap
    windows = []
    start = 0
    while True:
        end = start + size
        if len(words) - end <= overlap:  # rest is small: take everything up to the end
            windows.append(words[start:])
            return windows
        windows.append(words[start:end])
        start += stride


def make_chunks(speeches: pd.DataFrame) -> pd.DataFrame:
    """One row per chunk, with the speech metadata and a stable chunk_id."""
    rows = []
    for speech in speeches.itertuples(index=False):
        windows = split_words(speech.text.split())
        for i, window in enumerate(windows):
            row = {col: getattr(speech, col) for col in META_COLS}
            row["chunk_id"] = f"{speech.speech_id}_c{i:03d}"
            row["chunk_index"] = i
            row["n_chunks"] = len(windows)
            row["word_count"] = len(window)
            row["text"] = " ".join(window)
            rows.append(row)
    chunks = pd.DataFrame(rows)
    return chunks[["chunk_id", "chunk_index", "n_chunks", *META_COLS, "word_count", "text"]]


def validate(chunks: pd.DataFrame, speeches: pd.DataFrame) -> None:
    """Checks: unique ids, every speech covered, no empty or oversized chunks."""
    assert chunks["chunk_id"].is_unique, "duplicate chunk_id"
    assert set(chunks["speech_id"]) == set(speeches["speech_id"]), "speech without chunks"
    assert chunks["word_count"].min() > 0, "empty chunk"
    max_words = config.CHUNK_SIZE + config.CHUNK_OVERLAP
    assert chunks["word_count"].max() <= max_words, f"chunk longer than {max_words} words"

    per_speech = chunks.groupby("speech_id").size()
    print(f"[chunks] OK: {len(speeches)} speeches -> {len(chunks)} chunks")
    print(f"[chunks] chunks per speech: median {per_speech.median():.0f}, max {per_speech.max()}")
    print(f"[chunks] words per chunk: median {chunks['word_count'].median():.0f}, "
          f"min {chunks['word_count'].min()}, max {chunks['word_count'].max()}")


def main() -> None:
    """Split all speeches into chunks, validate them and save chunks.parquet."""
    speeches = pd.read_parquet(config.SPEECHES_PARQUET)
    chunks = make_chunks(speeches)
    validate(chunks, speeches)
    chunks.to_parquet(config.CHUNKS_PARQUET, index=False)
    print(f"[chunks] -> {config.CHUNKS_PARQUET}")


if __name__ == "__main__":
    main()
