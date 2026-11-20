"""Evaluate retrieval and answers on the hand-checked questions in eval/questions.jsonl.

Run with:
    python -m parlamint_rag.evaluate --retrieval-only     # fast: Qdrant + query embeddings only
    python -m parlamint_rag.evaluate                      # also generates all answers (slow on CPU)
    python -m parlamint_rag.evaluate --resume             # continue an interrupted run
    python -m parlamint_rag.evaluate --tag fix            # write answers_<model>_fix.csv instead
    python -m parlamint_rag.evaluate --summarize eval/results/answers_qwen3-1.7b.csv
                                                          # after rating the answers by hand

Retrieval metrics (answerable questions only, relevance on speech level):
    Recall@k  share of questions with at least one gold speech among the top k chunks
              (also called hit rate; with one gold speech it equals classic recall)
    MRR       mean of 1/rank of the first gold speech (0 if not in the top 10)

Answer metrics:
    automatic  abstention on unanswerable questions, citations present and valid,
               does the answer cite a gold speech
    manual     columns `faithful` and `correct` in the answers CSV, filled in by hand (1/0)
"""

import argparse
import json
import time
from pathlib import Path

import ollama
import pandas as pd

from parlamint_rag import config

KS = (1, 5, 10)
ABSTENTION_MARKER = "in den Quellen keine Antwort"
# Semicolon + UTF-8 with BOM: opens correctly in German Excel for manual rating
CSV_OPTIONS = {"sep": ";", "encoding": "utf-8-sig", "index": False}


def load_questions(path: Path = config.QUESTIONS_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def first_relevant_rank(retrieved_speech_ids: list[str], gold: set[str]) -> int | None:
    """Rank (1-based) of the first retrieved chunk that belongs to a gold speech."""
    for rank, speech_id in enumerate(retrieved_speech_ids, start=1):
        if speech_id in gold:
            return rank
    return None


# --- Retrieval --------------------------------------------------------------

def evaluate_retrieval(rag, questions: list[dict]) -> pd.DataFrame:
    rows = []
    for q in questions:
        if q["type"] == "unanswerable":
            continue
        hits = rag.retrieve(q["question"], k=max(KS), party=q["party"], year=q["year"])
        retrieved = [h.payload["speech_id"] for h in hits]
        rank = first_relevant_rank(retrieved, set(q["gold_speech_ids"]))
        rows.append({"id": q["id"], "type": q["type"], "question": q["question"],
                     "rank": rank, "top1_speech_id": retrieved[0] if retrieved else None})
    df = pd.DataFrame(rows)
    for k in KS:
        df[f"Recall@{k}"] = df["rank"].le(k)          # missing rank -> False
    df["MRR"] = (1 / df["rank"]).fillna(0.0)
    return df


def retrieval_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [f"Recall@{k}" for k in KS] + ["MRR"]
    summary = pd.concat([df[metrics].mean().to_frame("all").T, df.groupby("type")[metrics].mean()])
    summary.insert(0, "n", [len(df)] + df.groupby("type").size().tolist())
    return summary.round(2)


# --- Answers ----------------------------------------------------------------

def is_abstention(answer: str) -> bool:
    """Lenient: the answer contains the no-answer sentence (possibly after other text)."""
    return ABSTENTION_MARKER in answer


def is_clean_abstention(answer: str) -> bool:
    """Strict: the answer is only the no-answer sentence, as the prompt asks."""
    return answer.strip().strip("\"„“'") == config.NO_ANSWER


def answer_with_retry(rag, q: dict, attempts: int = 3, wait: int = 15) -> dict:
    """The Ollama runner occasionally crashes on the small CPU machine ("unexpected EOF").
    Ollama restarts it on the next request, so waiting and asking again usually works."""
    for attempt in range(1, attempts + 1):
        try:
            return rag.answer(q["question"], party=q["party"], year=q["year"])
        except ollama.ResponseError as error:
            if attempt == attempts:
                raise
            print(f"[evaluate] {q['id']} failed ({error}), retry {attempt}/{attempts - 1} in {wait} s", flush=True)
            time.sleep(wait)


def answer_row(q: dict, result: dict) -> dict:
    sources, cited = result["sources"], result["cited"]
    gold = set(q["gold_speech_ids"])
    cited_speech_ids = {s["speech_id"] for s in sources if s["n"] in cited}
    return {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "reference_answer": q["reference_answer"],
        "answer": result["answer"],
        "abstained": is_abstention(result["answer"]),
        "n_citations": len(cited),
        "citations_valid": all(1 <= n <= len(sources) for n in cited),
        "gold_retrieved": any(s["speech_id"] in gold for s in sources),
        "cites_gold": bool(cited_speech_ids & gold),
        "sources": " | ".join(f"[{s['n']}] {s['speaker']}, {s['date']}, {s['speech_id']}" for s in sources),
        "seconds": result.get("stats", {}).get("seconds"),
        # filled in by hand: 1 = yes, 0 = no
        "faithful": "",   # every statement is supported by the cited source
        "correct": "",    # the answer matches the reference answer
        "notes": "",
    }


def evaluate_answers(rag, questions: list[dict], path: Path, resume: bool = False) -> pd.DataFrame:
    """Answer all questions. Saves after every answer, so a crash loses nothing;
    with resume=True, questions already in `path` are skipped."""
    done = read_answers_csv(path) if resume and path.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    done_ids = set(done["id"]) if len(done) else set()
    if done_ids:
        print(f"[evaluate] resuming, {len(done_ids)} answers already in {path.name}", flush=True)
    for i, q in enumerate(questions, start=1):
        if q["id"] in done_ids:
            continue
        rows.append(answer_row(q, answer_with_retry(rag, q)))
        pd.DataFrame(rows).to_csv(path, **CSV_OPTIONS)
        print(f"[evaluate] {i}/{len(questions)} {q['id']} ({q['type']}) {rows[-1]['seconds']} s", flush=True)
    order = {q["id"]: i for i, q in enumerate(questions)}
    return pd.DataFrame(rows).sort_values("id", key=lambda s: s.map(order), ignore_index=True)


def answer_summary(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["answer"] = df["answer"].fillna("").astype(str)
    # Recomputed from the answer text, so older CSVs are summarised with the same definitions
    df["abstained"] = df["answer"].map(is_abstention)
    df["clean"] = df["answer"].map(is_clean_abstention)
    answerable = df[df["type"] != "unanswerable"]
    unanswerable = df[df["type"] == "unanswerable"]
    summary = {
        "answerable: answered (no abstention)": 1 - answerable["abstained"].mean(),
        "answerable: gold speech among sources": answerable["gold_retrieved"].mean(),
        "answerable: answer cites a gold speech": answerable["cites_gold"].mean(),
        "answerable: has at least one citation": (answerable["n_citations"] > 0).mean(),
        "all: citations valid": df["citations_valid"].mean(),
        "all: mixed (text plus no-answer sentence)": (df["abstained"] & ~df["clean"]).mean(),
        "unanswerable: abstained (contains no-answer sentence)": unanswerable["abstained"].mean(),
        "unanswerable: clean abstention (only that sentence)": unanswerable["clean"].mean(),
        "mean seconds per answer": pd.to_numeric(df["seconds"]).mean(),
    }
    # Manual ratings, only where filled in
    groups = {"answered answerable": answerable[~answerable["abstained"]], "unanswerable": unanswerable}
    for name, group in groups.items():
        for col in ("faithful", "correct"):
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            if len(values):
                summary[f"manual {name}: {col} (n={len(values)})"] = values.mean()
    return pd.Series(summary).round(2)


def read_answers_csv(path: Path) -> pd.DataFrame:
    """Read the (hand-rated) answers CSV. Falls back to cp1252 if Excel saved it as ANSI."""
    try:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype={"faithful": str, "correct": str})
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=";", encoding="cp1252", dtype={"faithful": str, "correct": str})
    for col in ("abstained", "citations_valid", "gold_retrieved", "cites_gold"):
        df[col] = df[col].astype(str).str.lower().isin(["true", "wahr", "1"])
    return df


# --- Main -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers.")
    parser.add_argument("--retrieval-only", action="store_true", help="skip answer generation")
    parser.add_argument("--summarize", type=Path, help="summarise an existing (rated) answers CSV")
    parser.add_argument("--resume", action="store_true",
                        help="keep answers already in the answers CSV and only generate the missing ones")
    parser.add_argument("--tag", help="suffix for the answers CSV, e.g. 'fix' -> answers_<model>_fix.csv")
    args = parser.parse_args()

    if args.summarize:
        print(answer_summary(read_answers_csv(args.summarize)).to_string())
        return

    from parlamint_rag.rag import RAG  # imported here: loads torch, not needed for --summarize

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_questions()
    rag = RAG()

    retrieval = evaluate_retrieval(rag, questions)
    retrieval.to_csv(config.RESULTS_DIR / "retrieval.csv", **CSV_OPTIONS)
    summary = retrieval_summary(retrieval)
    summary.to_csv(config.RESULTS_DIR / "retrieval_summary.csv", sep=";", encoding="utf-8-sig")
    print("\nRetrieval\n" + summary.to_string())
    missed = retrieval[retrieval["rank"].isna()]
    if len(missed):
        print("\nNot in the top 10:", ", ".join(missed["id"]))

    if args.retrieval_only:
        return

    model = config.LLM_MODEL.replace(":", "-")
    path = config.RESULTS_DIR / f"answers_{model}{'_' + args.tag if args.tag else ''}.csv"
    answers = evaluate_answers(rag, questions, path, resume=args.resume)
    answers.to_csv(path, **CSV_OPTIONS)
    print(f"\nAnswers ({config.LLM_MODEL})\n" + answer_summary(answers).to_string())
    print(f"\nRate the answers by hand in {path} (columns faithful, correct: 1 or 0), then run:\n"
          f"python -m parlamint_rag.evaluate --summarize {path.relative_to(config.ROOT).as_posix()}")


if __name__ == "__main__":
    main()
