"""ETL for ParlaMint-AT: download, extract, read, clean, validate, store.

Run with:  python -m parlamint_rag.etl
"""

import csv
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd

from parlamint_rag import config

# Archive members we need, e.g.
#   .../ParlaMint-AT.txt/2019/ParlaMint-AT_2019-01-30-...txt
#   .../ParlaMint-AT.txt/2019/ParlaMint-AT_2019-01-30-...-meta-en.tsv
MEMBER_RE = re.compile(r"ParlaMint-AT\.txt/(\d{4})/[^/]+(\.txt|meta-en\.tsv)$")

PARTIES = {"ÖVP", "SPÖ", "FPÖ", "Grüne", "NEOS", "JETZT"}
COLUMNS = ["speech_id", "date", "year", "speaker", "party", "orientation", "word_count", "text"]


# --- Extract ----------------------------------------------------------------

def download(url: str = config.PARLAMINT_URL, dest: Path = config.TGZ_PATH) -> Path:
    """Download the archive once. Skips the download if the file already exists."""
    if dest.exists():
        print(f"[download] {dest.name} already exists, skipping")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"[download] {url}\n        -> {dest} (~330 MB, takes a few minutes)")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.rename(dest)  # only a complete download gets the final name
    return dest


def extract(
    tgz: Path = config.TGZ_PATH,
    out_dir: Path = config.RAW_DIR,
    years=config.YEARS,
) -> Path:
    """Extract only plain-text speeches and English metadata for the given years."""
    wanted_years = {str(y) for y in years}
    n_files = 0
    with tarfile.open(tgz, "r:gz") as tar:
        for member in tar:  # streams through the archive once
            match = MEMBER_RE.search(member.name)
            if member.isfile() and match and match.group(1) in wanted_years:
                # filter="data" blocks absolute paths, "../" and links (path traversal)
                tar.extract(member, out_dir, filter="data")
                n_files += 1
    if n_files == 0:
        raise RuntimeError("No matching files found. Has the archive layout changed?")
    print(f"[extract] {n_files} files -> {out_dir}")
    return out_dir


def read_raw(raw_dir: Path = config.RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read all metadata TSVs and speech TXTs into two DataFrames: (meta, texts)."""
    meta_files = sorted(raw_dir.rglob("*meta-en.tsv"))
    # rglob("*.txt") also matches the folder "ParlaMint-AT.txt", so keep files only
    txt_files = sorted(p for p in raw_dir.rglob("*.txt") if p.is_file())

    # Metadata rows end with an extra tab. index_col=False stops pandas from
    # turning the first column into the index, which would shift all columns.
    # QUOTE_NONE: a '"' inside a speech is text, not the start of a quoted field.
    meta = pd.concat(
        [pd.read_csv(p, sep="\t", index_col=False, quoting=csv.QUOTE_NONE) for p in meta_files],
        ignore_index=True,
    )
    texts = pd.concat(
        [
            pd.read_csv(p, sep="\t", header=None, names=["ID", "text"],
                        encoding="utf-8-sig", quoting=csv.QUOTE_NONE)
            for p in txt_files
        ],
        ignore_index=True,
    )
    print(f"[read] {len(meta_files)} meta files -> {len(meta)} rows")
    print(f"[read] {len(txt_files)} txt files  -> {len(texts)} rows")
    return meta, texts


# --- Transform --------------------------------------------------------------

def transform(meta: pd.DataFrame, texts: pd.DataFrame) -> pd.DataFrame:
    """Merge and clean. Returns one row per speech with the columns in COLUMNS."""

    def log(step: str, df: pd.DataFrame) -> None:
        print(f"[transform] {step:<30} {len(df):>6} rows")

    # validate="one_to_one" raises an error if an ID appears twice on either side
    df = meta.merge(texts, on="ID", how="inner", validate="one_to_one")
    log("merge meta + text", df)

    df = df[df["Speaker_role"] != "Chairperson"].copy()
    log("drop chairperson", df)

    # [[...]] holds interjections, applause and timestamps. Kept in, other MPs'
    # words would be attributed to the speaker.
    df["text"] = (
        df["text"]
        .str.replace(r"\[\[.*?\]\]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["word_count"] = df["text"].str.split().str.len()
    df = df[df["word_count"] >= config.MIN_WORDS].copy()
    log(f"keep >= {config.MIN_WORDS} words", df)

    # "-" means "no party in the metadata" (mostly ministers without a seat).
    # Keep these speeches, but store the party as missing instead of "-".
    df["party"] = df["Speaker_party"].replace({"GRÜNE": "Grüne", "-": pd.NA})
    df["orientation"] = df["Party_orientation"].replace({"-": pd.NA})
    print(f"[transform] {'speeches without party':<30} {df['party'].isna().sum():>6} rows")

    df["date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df["year"] = df["date"].dt.year
    df = df.rename(columns={"ID": "speech_id", "Speaker_name": "speaker"})
    return df[COLUMNS].reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    """Data-quality checks: fail loudly instead of silently producing bad data."""
    assert df["speech_id"].is_unique, "duplicate speech_id"
    key_cols = ["speech_id", "date", "year", "speaker", "word_count", "text"]
    assert df[key_cols].notna().all().all(), "missing values in key columns"
    unknown = set(df["party"].dropna()) - PARTIES
    assert not unknown, f"unexpected party values: {unknown}"
    assert df["year"].between(min(config.YEARS), max(config.YEARS)).all(), "year out of range"
    assert not df["text"].str.contains(r"\[\[|\]\]").any(), "comment markers left in text"
    assert (df["word_count"] >= config.MIN_WORDS).all(), "speech below MIN_WORDS"

    print(f"[validate] OK: {len(df)} speeches, {df['word_count'].sum():,} words, "
          f"{df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
    print(df["party"].value_counts(dropna=False).to_string())


# --- Load -------------------------------------------------------------------

def load(df: pd.DataFrame) -> None:
    """Write speeches.parquet (with text) and a DuckDB table `speeches`
    (metadata only, no text) for quick SQL checks."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.SPEECHES_PARQUET, index=False)

    with duckdb.connect(str(config.DUCKDB_PATH)) as con:
        con.register("speeches_df", df)
        con.execute("CREATE OR REPLACE TABLE speeches AS SELECT * EXCLUDE (text) FROM speeches_df")
    print(f"[load] {config.SPEECHES_PARQUET.name} and {config.DUCKDB_PATH.name} -> {config.PROCESSED_DIR}")


def main() -> None:
    download()
    extract()
    meta, texts = read_raw()
    df = transform(meta, texts)
    validate(df)
    load(df)


if __name__ == "__main__":
    main()
