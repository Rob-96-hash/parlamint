"""Cheap check of a generated answer: is every number in a source that the sentence cites?

Small models often state the main fact correctly but add details that are not in the
sources (a wrong year, an extra percentage) or cite the wrong source number. Checking
every claim would need a second model (NLI or an LLM judge). Numbers are the most
frequent and most harmful claims in this corpus, and they can be checked with plain
string matching, so this check is fast, deterministic and easy to explain.

Limits: claims in words ("dreimal", "im Sommer") are not checked, and a number that is
in the source but used in the wrong context ("68 million" for the wrong programme) passes.
A flag means "check this sentence", not "this sentence is wrong".
"""

import re

# A number as written in German texts: 7, 2023, 16 000, 13.420, 2,3. Thousands are
# separated by a space, a no-break space or a dot; the decimal separator is a comma.
# Grouped numbers start with 1 to 3 digits, so "2022 250 Millionen" stays two numbers.
NUMBER_RE = re.compile(r"\d{1,3}(?:[ .  ]\d{3}(?!\d))+(?:,\d+)?|\d+(?:,\d+)?")
CITATION_RE = re.compile(r"\[(\d+)\]")
# Sentence end: . ! or ? plus whitespace, but not when a citation follows ("Euro. [2]")
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?!\[)")


def numbers(text: str) -> set[str]:
    """All numbers in a text, normalised ("16 000" and "16.000" both become "16000")."""
    # Citation markers like [2] are not claims
    text = CITATION_RE.sub(" ", text)
    return {re.sub(r"[ .  ]", "", m) for m in NUMBER_RE.findall(text)}


def unsupported_numbers(answer: str, sources: dict[int, str], question: str = "") -> list[dict]:
    """Numbers in the answer that do not appear in the sources their sentence cites.

    `sources` maps the source number shown to the model to its text. Returns one dict
    per problem: {"number", "sentence", "found_in"}. `found_in` lists other sources that
    contain the number: empty means the number is in no source (probably invented),
    otherwise the sentence cites the wrong source. Numbers from the question are skipped,
    because the model may repeat them ("bis 2030").
    """
    source_numbers = {n: numbers(text) for n, text in sources.items()}
    from_question = numbers(question)
    problems = []
    for sentence in SENTENCE_END_RE.split(answer.strip()):
        cited = {int(n) for n in CITATION_RE.findall(sentence)} & source_numbers.keys()
        # A sentence without a valid citation is checked against all sources
        allowed = set().union(*(source_numbers[n] for n in (cited or source_numbers)))
        for number in sorted(numbers(sentence) - from_question - allowed):
            found_in = [n for n, nums in source_numbers.items() if number in nums]
            problems.append({"number": number, "sentence": sentence, "found_in": found_in})
    return problems
