"""Thin orchestrator around original TokenAlign components.

This module exposes convenience wrappers that call the original TokenAlign
scripts (ported into medical_tokalign/src):
 - cal_trans_matrix.py (GloVe + relative representation -> 1-1 mapping)
 - convert.py (apply mapping to model/tokenizer)

Note: The full end-to-end pipeline is executed via the shell script
`scripts/run_vocab_adaptation.sh`. These Python wrappers are provided for
programmatic use if needed.
"""

from typing import Dict, Tuple
import json
import os

from .cal_trans_matrix import (
    load_glove_model,
    convert2matrix,
    convert2relative_rep,
)
from .convert import trans2switch


def build_alignment_matrix(
    source_glove_vector_path: str,
    target_glove_vector_path: str,
    gold_target_to_source_path: str,
    source_vocab_size: int,
    target_vocab_size: int,
    use_relative: bool = True,
    pivotal_token_number: int = 300,
) -> Dict[str, int]:
    with open(gold_target_to_source_path, "r") as f:
        t2l_supl = json.load(f)

    # Load GloVe vectors
    embed1 = load_glove_model(target_glove_vector_path)
    embed2 = load_glove_model(source_glove_vector_path)

    if use_relative:
        ids1, rep1, ids2, rep2 = convert2relative_rep(
            embed1=embed1,
            embed2=embed2,
            gold=t2l_supl,
            num_pivot=pivotal_token_number,
        )
    else:
        ids1, rep1 = convert2matrix(embed1)
        ids2, rep2 = convert2matrix(embed2)

    import numpy as np
    sim = np.matmul(rep1, rep2.T)

    td: Dict[str, int] = {}
    tids = [str(tid) for tid in range(target_vocab_size)]
    for tid in tids:
        if tid in t2l_supl:
            td[tid] = t2l_supl[tid]
            continue
        if tid not in ids1:
            td[tid] = 0
            continue
        id1_idx = ids1.index(tid)
        lix = int(np.argmax(sim[id1_idx]))
        lid = ids2[lix]
        if str(lid) in ('unk','<unk>'):
            # pick the second best
            top2 = np.argpartition(sim[id1_idx], -2)[-2:]
            lid = ids2[int(top2[0] if top2[1] == lix else top2[1])]
        # Prefer numeric token id; fall back to similarity index when non-numeric
        try:
            td[tid] = int(lid)
        except (ValueError, TypeError):
            td[tid] = int(lix)
    return td


def adapt_tokenizer(
    mapping: Dict[str, int],
    source_model_path: str,
    target_tokenizer_path: str,
    output_model_path: str,
) -> Tuple[str, str]:
    os.makedirs(output_model_path, exist_ok=True)
    tmp_map = os.path.join(output_model_path, "align_matrix.json")
    with open(tmp_map, "w") as f:
        json.dump(mapping, f)
    trans2switch(
        trans_path=tmp_map,
        src_clm_path=source_model_path,
        tgt_clm_path=output_model_path,
        tgt_tok_path=target_tokenizer_path,
        random_shuffle=-1,
    )
    return output_model_path, target_tokenizer_path


def save_adapted_tokenizer(model_dir: str, tokenizer_dir: str) -> Dict[str, str]:
    return {"model_dir": model_dir, "tokenizer_dir": tokenizer_dir}


