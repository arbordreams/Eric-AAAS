from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
from rouge_score import rouge_scorer
import sacrebleu


def accuracy(preds: List[str], labels: List[str]) -> float:
    correct = 0
    total = 0
    for p, y in zip(preds, labels):
        if y is None or y == "":
            continue
        total += 1
        if str(p).strip().lower() == str(y).strip().lower():
            correct += 1
    return (correct / total) if total > 0 else 0.0


def extract_letter(text: str) -> str:
    m = re.search(r"\b([ABCD])\b", text.strip(), flags=re.IGNORECASE)
    return m.group(1).upper() if m else ""


def ner_span_f1(pred_lists: List[List[str]], gold_lists: List[List[str]]) -> Dict[str, float]:
    tp = fp = fn = 0
    per_doc_f1 = []
    for pred, gold in zip(pred_lists, gold_lists):
        pred_norm = {normalize_entity(x) for x in pred}
        gold_norm = {normalize_entity(x) for x in gold}
        tpi = len(pred_norm & gold_norm)
        fpi = len(pred_norm - gold_norm)
        fni = len(gold_norm - pred_norm)
        tp += tpi; fp += fpi; fn += fni
        denom = (2*tpi + fpi + fni)
        per_doc_f1.append((2*tpi/denom) if denom else 1.0)
    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = (2*micro_p*micro_r/(micro_p+micro_r)) if (micro_p+micro_r) else 0.0
    macro_f1 = float(np.mean(per_doc_f1)) if per_doc_f1 else 0.0
    return {"micro_p": micro_p, "micro_r": micro_r, "micro_f1": micro_f1, "macro_f1": macro_f1}


def normalize_entity(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def perplexity(nlls: Iterable[float]) -> float:
    nlls = [x for x in nlls if x is not None and not math.isnan(x)]
    if not nlls:
        return float("inf")
    return float(math.exp(np.mean(nlls)))


def bleu(preds: List[str], refs: List[str]) -> float:
    preds = [p.strip() for p in preds]
    refs = [[r.strip() for r in refs]]
    return float(sacrebleu.corpus_bleu(preds, refs).score)


def rouge_l(preds: List[str], refs: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    return float(np.mean(scores)) if scores else 0.0


def medical_term_precision_recall(pred_texts: List[str], ref_texts: List[str]) -> Dict[str, float]:
    try:
        import spacy
        nlp = spacy.load("en_core_sci_lg")
    except Exception:
        # fallback: naive term extraction via capitalized words
        def extract_naive(t: str) -> List[str]:
            return re.findall(r"[A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)*", t)
        pred_terms = [set(extract_naive(t)) for t in pred_texts]
        ref_terms = [set(extract_naive(t)) for t in ref_texts]
    else:
        def ents(doc):
            return {e.text for e in doc.ents}
        pred_terms = [ents(nlp(t)) for t in pred_texts]
        ref_terms = [ents(nlp(t)) for t in ref_texts]

    tp = fp = fn = 0
    for pset, rset in zip(pred_terms, ref_terms):
        tp += len(pset & rset)
        fp += len(pset - rset)
        fn += len(rset - pset)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


