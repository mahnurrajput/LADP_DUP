"""
vqa_eval.py — LADP
==================
One place for everything that defines a VizWiz VQA result, shared by
``scripts/run_baseline.py`` (the 0.528 baseline), ``scripts/run_diagnostic.py``
and the LADP evaluation, so their numbers stay comparable.

* ``PROMPT`` / ``GENERATION``  — the exact prompt and greedy decoding settings of
  the baseline (including the "respond with 'Unanswerable'" instruction, which
  matters on VizWiz: 448 of the first 1,000 val questions are unanswerable).
* ``norm`` / ``vqa_acc``       — the standard VQA-eval answer normalisation and the
  official accuracy: the mean over the 10 leave-one-out subsets of
  min(1, #matches / 3).  Moved here unchanged from run_baseline.py.
* ``answer_strings``           — VizWiz annotations store answers as
  ``{"answer": str, "answer_confidence": str}``; this extracts the strings and
  rejects anything else loudly.
* ``any_match``                — a separate, clearly named binary label
  ("at least one annotator gave this answer").  It is NOT VQA accuracy.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Mapping, Union

PROMPT = ("USER: <image>\n{q} When the provided information is insufficient, "
          "respond with 'Unanswerable'. Answer the question using a single word or phrase. ASSISTANT:")

# Greedy decoding exactly as in run_baseline.py.
GENERATION = {"max_new_tokens": 16, "do_sample": False}

# --- VQA answer normalisation (standard VQA-eval style) ---------------------
_NUM = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ART = {"a", "an", "the"}
_PUNCT = re.compile(r"[;/\[\]\"{}()=+\\_\-><@`,?!*#%^&$:|~]")


def build_prompt(question: str) -> str:
    return PROMPT.format(q=question)


def norm(a: str) -> str:
    a = a.replace("\n", " ").replace("\t", " ").strip().lower()
    a = _PUNCT.sub(" ", a)
    a = re.sub(r"(?<!\d)\.(?!\d)", "", a)  # keep decimal points
    words = [_NUM.get(w, w) for w in a.split() if w not in _ART]
    return " ".join(words)


def answer_strings(answers: Iterable[Union[str, Mapping]]) -> List[str]:
    """Answer strings from a VizWiz ``answers`` list (dicts with "answer", or str)."""
    out: List[str] = []
    for a in answers:
        if isinstance(a, str):
            out.append(a)
        elif isinstance(a, Mapping) and isinstance(a.get("answer"), str):
            out.append(a["answer"])
        else:
            raise TypeError(f"Unsupported VizWiz answer entry {a!r}: expected a str or a dict "
                            "with a string 'answer' field.")
    return out


def vqa_acc(pred: str, answers: List[str]) -> float:
    """Official VQA accuracy: mean over the 10 leave-one-out subsets of min(1, #match/3)."""
    p = norm(pred)
    gts = [norm(a) for a in answers]
    accs = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1:]
        accs.append(min(1.0, sum(g == p for g in others) / 3.0))
    return sum(accs) / len(accs)


def any_match(pred: str, answers: List[str]) -> bool:
    """True if at least one annotator's answer equals the prediction (after ``norm``).

    A lenient binary label, kept separate from (and never reported as) VQA accuracy.
    """
    p = norm(pred)
    return any(norm(a) == p for a in answers)
