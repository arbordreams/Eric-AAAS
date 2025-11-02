import json
import os
from typing import Dict, List, Tuple

from datasets import load_dataset


def _load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_pubmedqa(bench_dir: str) -> Dict[str, List[dict]]:
    data = {}
    for split in ["train", "validation", "test"]:
        p = os.path.join(bench_dir, f"pubmedqa_{split}.jsonl")
        if os.path.isfile(p):
            data[split] = _load_jsonl(p)
    return data


def load_medqa(bench_dir: str) -> Dict[str, List[dict]]:
    data = {}
    for split in ["train", "validation", "test"]:
        p = os.path.join(bench_dir, f"medqa_{split}.jsonl")
        if os.path.isfile(p):
            data[split] = _load_jsonl(p)
    return data


def load_mednli(bench_dir: str) -> Dict[str, List[dict]]:
    data = {}
    for split in ["train", "validation", "test"]:
        p = os.path.join(bench_dir, f"mednli_{split}.jsonl")
        if os.path.isfile(p):
            data[split] = _load_jsonl(p)
    return data


def load_ncbi_disease(bench_dir: str) -> Dict[str, List[dict]]:
    data = {}
    for split in ["train", "validation", "test"]:
        p = os.path.join(bench_dir, f"ncbi_disease_{split}.jsonl")
        if os.path.isfile(p):
            data[split] = _load_jsonl(p)
    return data


def load_bc5cdr(bench_dir: str) -> Dict[str, List[dict]]:
    data = {}
    for split in ["train", "validation", "test"]:
        p = os.path.join(bench_dir, f"bc5cdr_{split}.jsonl")
        if os.path.isfile(p):
            data[split] = _load_jsonl(p)
    return data


def load_perplexity_corpus(proc_dir: str, source: str = "pubmed_rct") -> List[str]:
    if source == "pubmed_rct":
        p = os.path.join(proc_dir, "pubmed_rct_test.jsonl")
        if os.path.isfile(p):
            exs = _load_jsonl(p)
            texts = []
            for ex in exs:
                # bigbio/pubmed_rct uses fields like: abstract, title, sentences
                t = ex.get("text") or ex.get("abstract") or ex.get("sentence") or ""
                if isinstance(t, list):
                    t = "\n".join(t)
                texts.append(str(t))
            return texts
    # fallback empty
    return []


def format_pubmedqa_prompt(example: dict) -> Tuple[str, str]:
    # PubMedQA uses dict with fields: question, context(s), final_decision (yes/no/maybe)
    q = example.get("question") or example.get("question_concept") or ""
    ctx = example.get("context") or example.get("long_answer") or ""
    if isinstance(ctx, list):
        ctx = "\n".join(ctx)
    prompt = (
        "You are a medical QA assistant. Answer yes/no/maybe.\n"
        f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer: "
    )
    label = (example.get("final_decision") or example.get("answer") or "").strip().lower()
    return prompt, label


def format_medqa_prompt(example: dict) -> Tuple[str, str]:
    # Expect fields question + 4 options + correct answer letter or text
    q = example.get("question") or example.get("question_text") or ""
    opts = [
        example.get("A") or example.get("option_a") or "",
        example.get("B") or example.get("option_b") or "",
        example.get("C") or example.get("option_c") or "",
        example.get("D") or example.get("option_d") or "",
    ]
    prompt = (
        "You are a medical exam assistant. Choose A, B, C, or D and justify briefly.\n"
        f"Question: {q}\n"
        f"A) {opts[0]}\nB) {opts[1]}\nC) {opts[2]}\nD) {opts[3]}\n"
        "Answer (just the letter): "
    )
    ans = (example.get("answer") or example.get("label") or "").strip().upper()
    if ans and ans[0] in {"A","B","C","D"}:
        label = ans[0]
    else:
        # fallback: match answer text to options
        label = ""
    return prompt, label


def format_mednli_prompt(example: dict) -> Tuple[str, str]:
    # Fields: premise, hypothesis, label in {entailment, contradiction, neutral}
    prem = example.get("premise") or ""
    hyp = example.get("hypothesis") or ""
    prompt = (
        "You are a clinical NLI assistant. Predict entailment, contradiction, or neutral.\n"
        f"Premise: {prem}\nHypothesis: {hyp}\nLabel: "
    )
    label = (example.get("label") or example.get("gold_label") or "").strip().lower()
    return prompt, label


def extract_ner_gold_spans(example: dict) -> List[str]:
    # bigbio_ner style: entities with text spans
    ents = example.get("entities") or []
    gold = []
    for e in ents:
        txt = e.get("text")
        if isinstance(txt, list):
            txt = " ".join(txt)
        if txt:
            gold.append(str(txt))
    return gold


def format_ner_prompt(text: str, entity_type: str = "Disease") -> str:
    return (
        f"Extract all {entity_type} entities from the text. One per line, no bullets, exact surface forms.\n"
        f"Text: {text}\nEntities:\n"
    )


