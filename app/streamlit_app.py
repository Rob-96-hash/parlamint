"""Streamlit front end: ask a question, get a cited answer and the speech excerpts behind it.

Run with:  streamlit run app/streamlit_app.py      (Qdrant and Ollama must be running)
"""

import ollama
import streamlit as st

from parlamint_rag import config
from parlamint_rag.rag import NO_ANSWER, RAG, to_sources
from parlamint_rag.verify import uncited_sentences, unsupported_numbers

PARTIES = ["ÖVP", "SPÖ", "FPÖ", "Grüne", "NEOS", "JETZT"]
EXAMPLES = [
    "Wie stark ging die Zahl neu diagnostizierter Brustkrebsfälle während des Corona-Lockdowns zurück?",
    "Mit wie viel Geld pro Jahr beteiligt sich der Bund am Ausbau der Wiener U-Bahn?",
    "Wie stark sind die Importe Russlands laut den Grünen wegen der Sanktionen eingebrochen?",
]


@st.cache_resource(show_spinner="Lade Embedding-Modell und Reranker …")
def get_rag() -> RAG:
    """Load the models once per server process (Streamlit reruns this script on every click)."""
    return RAG()


def show_sources(sources: list[dict], cited: list[int]) -> None:
    """One expandable box per source; sources the answer cites are marked and opened."""
    st.subheader("Quellen")
    for s in sources:
        is_cited = s["n"] in cited
        mark = " · ✅ zitiert" if is_cited else ""
        label = (f"[{s['n']}] {s['speaker']} ({s['party'] or 'ohne Partei'}), {s['date']}"
                 f" · Score {s['score']:.2f}{mark}")
        with st.expander(label, expanded=is_cited):
            st.write(s["text"])
            st.caption(f"Rede-ID: {s['speech_id']}")


def show_checks(result: dict, question: str) -> None:
    """Automatic checks of the answer, shown as warnings (they flag, they do not correct)."""
    answer, sources = result["answer"], result["sources"]
    if result.get("gated"):
        st.info(f"Kein Redeabschnitt ist relevant genug (bester Score {sources[0]['score']:.2f} "
                f"unter der Schwelle {config.RELEVANCE_THRESHOLD}). Das Sprachmodell wurde nicht gefragt.")
        return
    invalid = [n for n in result["cited"] if n > len(sources)]
    if invalid:
        st.warning(f"Die Antwort zitiert Quellen, die es nicht gibt: {invalid}")
    if not result["cited"] and answer != NO_ANSWER:
        st.warning("Die Antwort enthält keine Quellenangaben.")
    else:
        # Single sentences without a source number (the prompt asks for one per sentence)
        uncited = uncited_sentences(answer, NO_ANSWER)
        if uncited:
            st.warning("Ohne Quellenangabe, bitte prüfen:\n" + "\n".join(f"- „{s}“" for s in uncited))
    # Text plus the no-answer sentence: the model is unsure (20 % of answers in the evaluation)
    if NO_ANSWER in answer and answer != NO_ANSWER:
        st.info("Die Antwort enthält Text und zugleich den Hinweis, dass die Quellen die Frage "
                "nicht beantworten. Bitte die Quellen prüfen.")
    # Numbers that are not in the cited sources: the most common invented detail
    for p in unsupported_numbers(answer, {s["n"]: s["text"] for s in sources}, question):
        where = (f"steht in Quelle {', '.join(f'[{n}]' for n in p['found_in'])}, nicht in der zitierten"
                 if p["found_in"] else "steht in keiner Quelle")
        st.warning(f"Zahl **{p['number']}** {where}: „{p['sentence']}“")


st.set_page_config(page_title="ParlaMint-AT RAG", page_icon="🏛️", layout="wide")

with st.sidebar:
    st.header("Filter")
    party = st.selectbox("Partei", ["alle", *PARTIES])
    year = st.selectbox("Jahr", ["alle", *config.YEARS])
    # More sources = more context for the model, but a longer prompt (4,096 token window)
    k = st.slider("Anzahl Quellen", min_value=3, max_value=6, value=config.TOP_K)
    st.divider()
    st.caption(f"Sprachmodell: {config.LLM_MODEL} (lokal über Ollama)  \n"
               f"Reranker: {config.RERANK_MODEL or 'aus'}  \n"
               f"Relevanzschwelle: {config.RELEVANCE_THRESHOLD if config.RELEVANCE_THRESHOLD is not None else 'aus'}  \n"
               f"Embeddings: {config.EMBED_MODEL}")

st.title("🏛️ Was wurde im Nationalrat gesagt?")
st.write("Fragen zu Reden im österreichischen Nationalrat 2018 bis 2022. Das System sucht passende "
         "Redeabschnitte und lässt ein lokales Sprachmodell daraus eine Antwort mit Quellen schreiben.")
# Transparency (EU AI Act): users must know that the answer is AI-generated
st.info("Die Antworten erzeugt ein kleines KI-Modell. Sie können falsch sein: Bitte prüfe die "
        "zitierten Quellen. Auf einer CPU dauert eine Antwort etwa 1 bis 2 Minuten.")

with st.form("ask"):
    question = st.text_input("Frage", placeholder=EXAMPLES[0])
    st.caption("Beispiele: " + " · ".join(EXAMPLES[1:]))
    submitted = st.form_submit_button("Fragen", type="primary")

# Nothing more to do until a question is submitted
if not submitted or not question.strip():
    st.stop()

rag = get_rag()
with st.spinner("Suche passende Redeabschnitte …"):
    hits = rag.retrieve(question, k=k, party=None if party == "alle" else party,
                        year=None if year == "alle" else year)
if not hits:
    st.warning(NO_ANSWER + " (Keine Rede passt zu den Filtern.)")
    st.stop()

# Placeholders: the answer appears above the sources, which are shown while the model writes
answer_slot = st.empty()
sources_slot = st.empty()
with sources_slot.container():
    show_sources(to_sources(hits), cited=[])

with answer_slot.container():
    with st.spinner(f"{config.LLM_MODEL} schreibt die Antwort …"):
        try:
            result = rag.generate(question, hits)
        except (ollama.ResponseError, ConnectionError) as err:
            st.error(f"Das Sprachmodell ist nicht erreichbar oder abgestürzt: {err}")
            st.stop()

with answer_slot.container():
    st.subheader("Antwort")
    st.markdown(result["answer"])
    show_checks(result, question)
    stats = result.get("stats")
    if stats:
        st.caption(f"{stats['prompt_tokens']} Prompt-Tokens, {stats['answer_tokens']} Antwort-Tokens, "
                   f"{stats['seconds']} s")

# Redraw the sources, now with the cited ones marked
with sources_slot.container():
    show_sources(result["sources"], result["cited"])

st.divider()
st.caption("Daten: ParlaMint-AT 4.x (CLARIN.SI, CC BY 4.0). Alle Komponenten laufen lokal: "
           "Qdrant, Ollama, keine externe API.")
